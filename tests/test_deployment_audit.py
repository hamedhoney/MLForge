from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from mlforge.registry.deployment_audit import (
    DeploymentAudit,
    DeploymentAuditEvent,
    DeploymentAuditStore,
)

COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40


@pytest.fixture
def store(tmp_path: Path):
    database = tmp_path / "audit.db"
    audit = DeploymentAuditStore(f"sqlite:///{database}")
    audit.initialize()
    yield audit
    audit.close()


def start(
    store: DeploymentAuditStore,
    *,
    run_id: int = 12345,
    commit: str = COMMIT,
    reviewer: str = "reviewer",
) -> int:
    return store.start(
        github_run_id=run_id,
        model_version="v0.4.4",
        previous_version="v0.3.1",
        training_commit=commit,
        quality_gate_run_id=67890,
        approved_by=reviewer,
    )


def test_start_creates_audit(store):
    audit_id = start(store)

    assert audit_id > 0
    assert store.history()[0]["status"] == "approved"
    assert store.history()[0]["approved_by"] == "reviewer"
    assert len(store.events(12345)) == 1


def test_start_is_idempotent(store):
    first = start(store)
    second = start(store)

    assert first == second
    assert len(store.history()) == 1
    assert len(store.events(12345)) == 1


def test_start_rejects_conflicting_provenance(store):
    start(store)

    with pytest.raises(ValueError, match="different provenance"):
        start(store, commit=OTHER_COMMIT)


def test_start_rejects_different_reviewer(store):
    start(store)

    with pytest.raises(ValueError, match="reviewer mismatch"):
        start(store, reviewer="another-reviewer")


@pytest.mark.parametrize(
    ("version", "commit", "run_id"),
    [
        ("invalid", COMMIT, 12345),
        ("v0.4.4", "invalid", 12345),
        ("v0.4.4", COMMIT, 0),
    ],
)
def test_invalid_identity(store, version, commit, run_id):
    with pytest.raises(ValueError):
        store.start(
            github_run_id=run_id,
            model_version=version,
            previous_version="v0.3.1",
            training_commit=commit,
            quality_gate_run_id=67890,
        )


def test_valid_deployment_lifecycle(store):
    start(store)

    store.record(
        github_run_id=12345,
        status="deploying",
        render_deployment_id="dep-123",
    )

    store.record(
        github_run_id=12345,
        status="verified",
        render_deployment_id="dep-123",
    )

    assert store.history()[0]["status"] == "verified"
    assert [event["status"] for event in store.events(12345)] == [
        "approved",
        "deploying",
        "verified",
    ]


def test_repeated_status_is_idempotent(store):
    start(store)

    for _ in range(2):
        store.record(
            github_run_id=12345,
            status="deploying",
            render_deployment_id="dep-123",
        )

    assert len(store.events(12345)) == 2


def test_conflicting_duplicate_is_rejected(store):
    start(store)

    store.record(
        github_run_id=12345,
        status="deploying",
        render_deployment_id="dep-123",
    )

    with pytest.raises(ValueError, match="Conflicting duplicate"):
        store.record(
            github_run_id=12345,
            status="deploying",
            render_deployment_id="dep-456",
        )


def test_invalid_transition_is_rejected(store):
    start(store)

    with pytest.raises(ValueError, match="Invalid transition"):
        store.record(
            github_run_id=12345,
            status="verified",
        )


def test_deploying_requires_deployment_id(store):
    start(store)

    with pytest.raises(ValueError, match="Deployment ID required"):
        store.record(
            github_run_id=12345,
            status="deploying",
        )


def test_failed_deployment_can_roll_back(store):
    start(store)

    store.record(
        github_run_id=12345,
        status="deploying",
        render_deployment_id="dep-123",
    )

    store.record(
        github_run_id=12345,
        status="failed",
    )

    store.record(
        github_run_id=12345,
        status="rollback_started",
    )

    store.record(
        github_run_id=12345,
        status="rolled_back",
        render_deployment_id="dep-rollback",
    )

    history = store.history()[0]

    assert history["status"] == "rolled_back"
    assert history["render_deployment_id"] == "dep-123"

    events = store.events(12345)

    assert events[-1]["render_deployment_id"] == "dep-rollback"


def test_verified_rollback_requires_authorization(store):
    start(store)

    store.record(
        github_run_id=12345,
        status="deploying",
        render_deployment_id="dep-123",
    )

    store.record(
        github_run_id=12345,
        status="verified",
        render_deployment_id="dep-123",
    )

    with pytest.raises(ValueError, match="authorization"):
        store.record(
            github_run_id=12345,
            status="rollback_started",
        )

    store.record(
        github_run_id=12345,
        status="rollback_started",
        rollback_authorized=True,
    )

    assert store.history()[0]["status"] == "rollback_started"


def test_terminal_state_is_protected(store):
    start(store)

    store.record(
        github_run_id=12345,
        status="failed",
    )
    store.record(
        github_run_id=12345,
        status="rollback_started",
    )
    store.record(
        github_run_id=12345,
        status="rollback_failed",
    )

    with pytest.raises(ValueError, match="terminal state"):
        store.record(
            github_run_id=12345,
            status="deploying",
            render_deployment_id="dep-new",
        )


def test_invalid_status_and_unknown_run(store):
    start(store)

    with pytest.raises(ValueError, match="Invalid deployment status"):
        store.record(
            github_run_id=12345,
            status="unknown",
        )

    with pytest.raises(ValueError, match="Unknown deployment run"):
        store.record(
            github_run_id=99999,
            status="verified",
        )


def test_history_order_and_limit(store):
    for run_id in range(1, 6):
        start(store, run_id=run_id)

    assert len(store.history(limit=2)) == 2
    assert store.history()[0]["github_run_id"] == 5

    with pytest.raises(ValueError):
        store.history(limit=0)


def test_concurrent_start_is_idempotent(store):
    def worker(_):
        return start(store)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, range(8)))

    assert len(set(results)) == 1
    assert len(store.history()) == 1
    assert len(store.events(12345)) == 1


def test_concurrent_status_updates(store):
    start(store)

    def worker(_):
        return store.record(
            github_run_id=12345,
            status="deploying",
            render_deployment_id="dep-123",
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, range(8)))

    assert len(set(results)) == 1
    assert len(store.events(12345)) == 2


def test_event_count_matches_history(store):
    start(store)

    store.record(
        github_run_id=12345,
        status="deploying",
        render_deployment_id="dep-123",
    )

    with Session(store.engine) as session:
        audit_count = session.scalar(select(func.count(DeploymentAudit.id)))
        event_count = session.scalar(select(func.count(DeploymentAuditEvent.id)))

    assert audit_count == 1
    assert event_count == 2

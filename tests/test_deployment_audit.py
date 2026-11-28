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
    """Create an isolated audit database."""
    database = tmp_path / "audit.db"
    audit = DeploymentAuditStore(f"sqlite:///{database}")
    audit.initialize()

    yield audit

    audit.close()


def start_deployment(
    store: DeploymentAuditStore,
    *,
    run_id: int = 12345,
    version: str = "v0.4.4",
    commit: str = COMMIT,
) -> int:
    """Create a deployment audit with valid provenance."""
    return store.start(
        github_run_id=run_id,
        model_version=version,
        previous_version="v0.3.1",
        training_commit=commit,
        quality_gate_run_id=67890,
        approved_by="reviewer",
    )


def test_start_creates_audit(store):
    audit_id = start_deployment(store)

    assert audit_id > 0

    history = store.history()

    assert len(history) == 1
    assert history[0]["model_version"] == "v0.4.4"
    assert history[0]["previous_version"] == "v0.3.1"
    assert history[0]["status"] == "approved"
    assert history[0]["approved_by"] == "reviewer"


def test_start_is_idempotent(store):
    first = start_deployment(store)
    second = start_deployment(store)

    assert first == second
    assert len(store.history()) == 1

    with Session(store.engine) as session:
        count = session.scalar(select(func.count(DeploymentAuditEvent.id)))

    assert count == 1


def test_start_rejects_conflicting_provenance(store):
    start_deployment(store)

    with pytest.raises(ValueError, match="different provenance"):
        start_deployment(
            store,
            commit=OTHER_COMMIT,
        )


def test_start_rejects_different_reviewer(store):
    start_deployment(store)

    with pytest.raises(ValueError, match="reviewer mismatch"):
        store.start(
            github_run_id=12345,
            model_version="v0.4.4",
            previous_version="v0.3.1",
            training_commit=COMMIT,
            quality_gate_run_id=67890,
            approved_by="different-reviewer",
        )


@pytest.mark.parametrize(
    ("version", "commit", "run_id"),
    [
        ("invalid", COMMIT, 12345),
        ("v0.4.4", "invalid", 12345),
        ("v0.4.4", COMMIT, 0),
    ],
)
def test_start_rejects_invalid_identity(
    store,
    version,
    commit,
    run_id,
):
    with pytest.raises(ValueError):
        store.start(
            github_run_id=run_id,
            model_version=version,
            previous_version="v0.3.1",
            training_commit=commit,
            quality_gate_run_id=67890,
        )


def test_record_deployment_status(store):
    audit_id = start_deployment(store)

    result = store.record(
        github_run_id=12345,
        status="deploying",
        render_deployment_id="dep-123",
    )

    assert result == audit_id

    history = store.history()

    assert history[0]["status"] == "deploying"
    assert history[0]["render_deployment_id"] == "dep-123"


def test_record_is_idempotent(store):
    start_deployment(store)

    store.record(
        github_run_id=12345,
        status="deploying",
        render_deployment_id="dep-123",
    )

    store.record(
        github_run_id=12345,
        status="deploying",
        render_deployment_id="dep-123",
    )

    with Session(store.engine) as session:
        count = session.scalar(select(func.count(DeploymentAuditEvent.id)))

    assert count == 2


def test_record_rejects_invalid_status(store):
    start_deployment(store)

    with pytest.raises(ValueError, match="Invalid deployment status"):
        store.record(
            github_run_id=12345,
            status="unknown",
        )


def test_record_rejects_unknown_run(store):
    with pytest.raises(ValueError, match="Unknown deployment run"):
        store.record(
            github_run_id=99999,
            status="verified",
        )


def test_record_rejects_invalid_deployment_id(store):
    start_deployment(store)

    with pytest.raises(ValueError, match="Invalid Render deployment ID"):
        store.record(
            github_run_id=12345,
            status="deploying",
            render_deployment_id="",
        )


def test_history_returns_newest_first(store):
    start_deployment(store, run_id=100)
    start_deployment(store, run_id=200)

    history = store.history()

    assert len(history) == 2
    assert history[0]["github_run_id"] == 200
    assert history[1]["github_run_id"] == 100


def test_history_limit(store):
    for run_id in range(1, 6):
        start_deployment(store, run_id=run_id)

    assert len(store.history(limit=2)) == 2

    with pytest.raises(ValueError):
        store.history(limit=0)

    with pytest.raises(ValueError):
        store.history(limit=501)


def test_verified_deployment_is_persisted(store):
    start_deployment(store)

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

    with Session(store.engine) as session:
        audit = session.scalar(
            select(DeploymentAudit).where(DeploymentAudit.github_run_id == 12345)
        )

        events = session.scalars(
            select(DeploymentAuditEvent)
            .where(DeploymentAuditEvent.audit_id == audit.id)
            .order_by(DeploymentAuditEvent.id)
        ).all()

    assert audit.status == "verified"
    assert [event.status for event in events] == [
        "approved",
        "deploying",
        "verified",
    ]


def test_rollback_events_are_persisted(store):
    start_deployment(store)

    for status in (
        "deploying",
        "failed",
        "rollback_started",
        "rolled_back",
    ):
        store.record(
            github_run_id=12345,
            status=status,
        )

    assert store.history()[0]["status"] == "rolled_back"

    with Session(store.engine) as session:
        events = session.scalars(
            select(DeploymentAuditEvent).order_by(DeploymentAuditEvent.id)
        ).all()

    assert [event.status for event in events] == [
        "approved",
        "deploying",
        "failed",
        "rollback_started",
        "rolled_back",
    ]

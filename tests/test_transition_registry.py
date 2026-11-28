import pytest

from mlforge.registry.repository import ModelRegistry
from mlforge.registry.transitions import (
    InvalidTransitionError,
    ModelStatus,
)

SHA = "a" * 64
COMMIT = "b" * 40


@pytest.fixture
def registry(tmp_path):
    instance = ModelRegistry(f"sqlite:///{tmp_path / 'registry.db'}")
    instance.initialize()
    yield instance
    instance.close()


def register_candidate(registry):
    return registry.register(
        version="v0.3.1",
        sha256=SHA,
        source_commit=COMMIT,
    )


def test_candidate_registration(registry):
    model_id = register_candidate(registry)

    registry.transition(
        model_id=model_id,
        target=ModelStatus.CANDIDATE,
    )

    assert registry.history()[0]["status"] == "candidate"


def test_direct_deployment_rejected(registry):
    model_id = register_candidate(registry)

    with pytest.raises(ValueError):
        registry.transition(
            model_id=model_id,
            target=ModelStatus.DEPLOYED,
            deployment_verified=True,
        )


def test_failed_quality_gate_rejected(registry):
    model_id = register_candidate(registry)

    registry.transition(
        model_id=model_id,
        target=ModelStatus.CANDIDATE,
    )

    with pytest.raises(InvalidTransitionError):
        registry.transition(
            model_id=model_id,
            target=ModelStatus.EVALUATED,
            quality_gate_passed=False,
        )


def test_approval_requires_identity(registry):
    model_id = register_candidate(registry)

    registry.transition(
        model_id=model_id,
        target=ModelStatus.CANDIDATE,
    )

    registry.transition(
        model_id=model_id,
        target=ModelStatus.EVALUATED,
        quality_gate_passed=True,
    )

    with pytest.raises(InvalidTransitionError):
        registry.transition(
            model_id=model_id,
            target=ModelStatus.APPROVED,
        )


def test_complete_lifecycle(registry):
    model_id = register_candidate(registry)

    registry.transition(
        model_id=model_id,
        target=ModelStatus.CANDIDATE,
    )

    registry.transition(
        model_id=model_id,
        target=ModelStatus.EVALUATED,
        quality_gate_passed=True,
    )

    registry.transition(
        model_id=model_id,
        target=ModelStatus.APPROVED,
        approved_by="test-approver",
    )

    registry.transition(
        model_id=model_id,
        target=ModelStatus.DEPLOYED,
        deployment_verified=True,
    )

    statuses = [event["status"] for event in reversed(registry.history())]

    assert statuses == [
        "candidate",
        "evaluated",
        "approved",
        "deployed",
    ]

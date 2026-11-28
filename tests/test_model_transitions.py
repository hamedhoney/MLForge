import pytest

from mlforge.registry.transitions import (
    InvalidTransitionError,
    ModelStatus,
    validate_transition,
)


def test_candidate_can_be_evaluated():
    validate_transition(
        ModelStatus.CANDIDATE,
        ModelStatus.EVALUATED,
        quality_gate_passed=True,
    )


def test_failed_quality_gate_is_rejected():
    with pytest.raises(InvalidTransitionError):
        validate_transition(
            ModelStatus.CANDIDATE,
            ModelStatus.EVALUATED,
            quality_gate_passed=False,
        )


def test_approval_requires_identity():
    with pytest.raises(InvalidTransitionError):
        validate_transition(
            ModelStatus.EVALUATED,
            ModelStatus.APPROVED,
        )


def test_approved_model_can_be_deployed():
    validate_transition(
        ModelStatus.APPROVED,
        ModelStatus.DEPLOYED,
        deployment_verified=True,
    )


def test_deployment_without_verification_is_rejected():
    with pytest.raises(InvalidTransitionError):
        validate_transition(
            ModelStatus.APPROVED,
            ModelStatus.DEPLOYED,
        )


def test_rollback_requires_authorization():
    with pytest.raises(InvalidTransitionError):
        validate_transition(
            ModelStatus.DEPLOYED,
            ModelStatus.ROLLED_BACK,
        )


def test_direct_candidate_deployment_is_rejected():
    with pytest.raises(InvalidTransitionError):
        validate_transition(
            ModelStatus.CANDIDATE,
            ModelStatus.DEPLOYED,
            deployment_verified=True,
        )


def test_valid_rollback():
    validate_transition(
        ModelStatus.DEPLOYED,
        ModelStatus.ROLLED_BACK,
        rollback_authorized=True,
    )

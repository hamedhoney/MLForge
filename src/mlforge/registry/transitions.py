from enum import StrEnum


class ModelStatus(StrEnum):
    CANDIDATE = "candidate"
    EVALUATED = "evaluated"
    APPROVED = "approved"
    DEPLOYED = "deployed"
    ROLLED_BACK = "rolled_back"


ALLOWED_TRANSITIONS = {
    ModelStatus.CANDIDATE: {ModelStatus.EVALUATED},
    ModelStatus.EVALUATED: {ModelStatus.APPROVED},
    ModelStatus.APPROVED: {ModelStatus.DEPLOYED},
    ModelStatus.DEPLOYED: {ModelStatus.ROLLED_BACK},
    ModelStatus.ROLLED_BACK: set(),
}


class InvalidTransitionError(ValueError):
    """Raised when a model lifecycle transition is forbidden."""


def validate_transition(
    current: ModelStatus,
    target: ModelStatus,
    *,
    quality_gate_passed: bool = False,
    approved_by: str | None = None,
    deployment_verified: bool = False,
    rollback_authorized: bool = False,
) -> None:
    """Validate a proposed model lifecycle transition."""
    current = ModelStatus(current)
    target = ModelStatus(target)

    if target not in ALLOWED_TRANSITIONS[current]:
        raise InvalidTransitionError(
            f"Transition {current.value} -> {target.value} is not allowed"
        )

    if target == ModelStatus.EVALUATED and not quality_gate_passed:
        raise InvalidTransitionError(
            "Candidate must pass quality gates before evaluation"
        )

    if target == ModelStatus.APPROVED and not approved_by:
        raise InvalidTransitionError("Human approval is required")

    if target == ModelStatus.DEPLOYED and not deployment_verified:
        raise InvalidTransitionError("Deployment verification is required")

    if target == ModelStatus.ROLLED_BACK and not rollback_authorized:
        raise InvalidTransitionError("Rollback authorization is required")

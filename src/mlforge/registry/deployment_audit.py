import re
from datetime import UTC, datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
)

VERSION_RE = re.compile(r"^v\d+\.\d+\.\d+$")
SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")

STATUSES = {
    "approved",
    "deploying",
    "verified",
    "failed",
    "rollback_started",
    "rolled_back",
    "rollback_failed",
}


class AuditBase(DeclarativeBase):
    """Base for deployment audit tables."""


class DeploymentAudit(AuditBase):
    """One auditable production promotion attempt."""

    __tablename__ = "model_deployment_audit"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    github_run_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    model_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    previous_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    training_commit: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
    )

    quality_gate_run_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    approved_by: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    render_deployment_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "github_run_id",
            name="uq_deployment_audit_github_run",
        ),
        CheckConstraint(
            "status IN ("
            "'approved', 'deploying', 'verified', "
            "'failed', 'rollback_started', "
            "'rolled_back', 'rollback_failed')",
            name="ck_deployment_audit_status",
        ),
    )


class DeploymentAuditEvent(AuditBase):
    """Append-only record of a deployment state change."""

    __tablename__ = "model_deployment_audit_events"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    audit_id: Mapped[int] = mapped_column(
        ForeignKey("model_deployment_audit.id"),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    render_deployment_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )


class DeploymentAuditStore:
    """Write deployment audit records to PostgreSQL."""

    def __init__(self, database_url: str):
        self.engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_recycle=300,
        )

    def initialize(self) -> None:
        """Create audit tables for development and initial setup."""
        AuditBase.metadata.create_all(self.engine)

    def start(
        self,
        *,
        github_run_id: int,
        model_version: str,
        previous_version: str,
        training_commit: str,
        quality_gate_run_id: int,
        approved_by: str | None = None,
    ) -> int:
        """Create or validate an existing promotion audit."""
        if github_run_id <= 0 or quality_gate_run_id <= 0:
            raise ValueError("Invalid GitHub run ID")

        if not VERSION_RE.fullmatch(model_version):
            raise ValueError("Invalid model version")

        if not VERSION_RE.fullmatch(previous_version):
            raise ValueError("Invalid previous version")

        if not SHA_RE.fullmatch(training_commit):
            raise ValueError("Invalid training commit")

        if approved_by is not None and (not approved_by or len(approved_by) > 128):
            raise ValueError("Invalid reviewer")

        with Session(self.engine) as session:
            with session.begin():
                existing = session.scalar(
                    select(DeploymentAudit)
                    .where(DeploymentAudit.github_run_id == github_run_id)
                    .with_for_update()
                )

                if existing is not None:
                    identity = (
                        existing.model_version,
                        existing.previous_version,
                        existing.training_commit.lower(),
                        existing.quality_gate_run_id,
                    )

                    expected = (
                        model_version,
                        previous_version,
                        training_commit.lower(),
                        quality_gate_run_id,
                    )

                    if identity != expected:
                        raise ValueError("Existing audit has different provenance")

                    if approved_by is not None and existing.approved_by != approved_by:
                        raise ValueError("Existing audit reviewer mismatch")

                    return existing.id

                audit = DeploymentAudit(
                    github_run_id=github_run_id,
                    model_version=model_version,
                    previous_version=previous_version,
                    training_commit=training_commit.lower(),
                    quality_gate_run_id=quality_gate_run_id,
                    approved_by=approved_by,
                    status="approved",
                )

                session.add(audit)
                session.flush()

                session.add(
                    DeploymentAuditEvent(
                        audit_id=audit.id,
                        status="approved",
                    )
                )

                audit_id = audit.id

            return audit_id

    def record(
        self,
        *,
        github_run_id: int,
        status: str,
        render_deployment_id: str | None = None,
    ) -> int:
        """Record a state change without duplicating identical events."""
        if status not in STATUSES:
            raise ValueError("Invalid deployment status")

        if render_deployment_id is not None and (
            not render_deployment_id or len(render_deployment_id) > 128
        ):
            raise ValueError("Invalid Render deployment ID")

        with Session(self.engine) as session:
            with session.begin():
                audit = session.scalar(
                    select(DeploymentAudit)
                    .where(DeploymentAudit.github_run_id == github_run_id)
                    .with_for_update()
                )

                if audit is None:
                    raise ValueError("Unknown deployment run")

                if (
                    audit.status == status
                    and audit.render_deployment_id == render_deployment_id
                ):
                    return audit.id

                audit.status = status
                audit.render_deployment_id = render_deployment_id

                session.add(
                    DeploymentAuditEvent(
                        audit_id=audit.id,
                        status=status,
                        render_deployment_id=render_deployment_id,
                    )
                )

                return audit.id

    def history(self, limit: int = 50) -> list[dict]:
        """Return recent promotion attempts."""
        if not 1 <= limit <= 500:
            raise ValueError("Invalid history limit")

        with Session(self.engine) as session:
            rows = session.scalars(
                select(DeploymentAudit)
                .order_by(
                    DeploymentAudit.created_at.desc(),
                    DeploymentAudit.id.desc(),
                )
                .limit(limit)
            ).all()

            return [
                {
                    "id": row.id,
                    "github_run_id": row.github_run_id,
                    "model_version": row.model_version,
                    "previous_version": row.previous_version,
                    "training_commit": row.training_commit,
                    "quality_gate_run_id": row.quality_gate_run_id,
                    "approved_by": row.approved_by,
                    "render_deployment_id": (row.render_deployment_id),
                    "status": row.status,
                    "created_at": row.created_at.isoformat(),
                    "updated_at": row.updated_at.isoformat(),
                }
                for row in rows
            ]

    def close(self) -> None:
        """Dispose database connections."""
        self.engine.dispose()

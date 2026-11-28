"""Concurrency-safe deployment lifecycle auditing."""

import re
import time
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
    update,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
)

VERSION_RE = re.compile(r"^v\d+\.\d+\.\d+$")
SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")

STATUSES = frozenset(
    {
        "approved",
        "deploying",
        "verified",
        "failed",
        "rollback_started",
        "rolled_back",
        "rollback_failed",
    }
)

TRANSITIONS = {
    "approved": {"deploying", "failed"},
    "deploying": {"verified", "failed"},
    "verified": {"rollback_started"},
    "failed": {"rollback_started"},
    "rollback_started": {"rolled_back", "rollback_failed"},
    "rolled_back": set(),
    "rollback_failed": set(),
}

TERMINAL_STATUSES = frozenset({"rolled_back", "rollback_failed"})

MAX_RETRIES = 10


class AuditBase(DeclarativeBase):
    """Declarative base for deployment auditing."""


class DeploymentAudit(AuditBase):
    """One promotion identity with its current lifecycle state."""

    __tablename__ = "model_deployment_audit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    github_run_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    previous_version: Mapped[str] = mapped_column(String(64), nullable=False)
    training_commit: Mapped[str] = mapped_column(String(40), nullable=False)
    quality_gate_run_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    approved_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    render_deployment_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
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
    """Append-only deployment transition event."""

    __tablename__ = "model_deployment_audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    audit_id: Mapped[int] = mapped_column(
        ForeignKey("model_deployment_audit.id"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    render_deployment_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )


class DeploymentAuditStore:
    """Persist deployment identities and atomic state transitions."""

    def __init__(self, database_url: str):
        self.engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_recycle=300,
            connect_args=(
                {"timeout": 30}
                if database_url.startswith("sqlite")
                else (
                    {"connect_timeout": 5}
                    if database_url.startswith("postgresql")
                    else {}
                )
            ),
        )

    def initialize(self) -> None:
        """Create audit tables for development and initial setup."""
        AuditBase.metadata.create_all(self.engine)

    @staticmethod
    def _validate_version(version: str) -> None:
        if not isinstance(version, str) or not VERSION_RE.fullmatch(version):
            raise ValueError("Invalid model version")

    @staticmethod
    def _validate_deployment_id(deployment_id: str | None) -> None:
        if deployment_id is not None and (
            not isinstance(deployment_id, str)
            or not deployment_id
            or len(deployment_id) > 128
        ):
            raise ValueError("Invalid Render deployment ID")

    @staticmethod
    def _retry_delay(attempt: int) -> None:
        time.sleep(min(0.02 * (2**attempt), 0.5))

    def _insert_if_absent(self, values: dict) -> None:
        """Atomically insert an identity using its unique run ID."""
        dialect = self.engine.dialect.name

        if dialect == "postgresql":
            statement = pg_insert(DeploymentAudit).values(**values)
        elif dialect == "sqlite":
            statement = sqlite_insert(DeploymentAudit).values(**values)
        else:
            raise RuntimeError(f"Unsupported audit database: {dialect}")

        statement = statement.on_conflict_do_nothing(index_elements=["github_run_id"])

        for attempt in range(MAX_RETRIES):
            try:
                with Session(self.engine) as session:
                    with session.begin():
                        result = session.execute(statement)

                        if result.rowcount == 1:
                            audit = session.scalar(
                                select(DeploymentAudit).where(
                                    DeploymentAudit.github_run_id
                                    == values["github_run_id"]
                                )
                            )

                            session.add(
                                DeploymentAuditEvent(
                                    audit_id=audit.id,
                                    status="approved",
                                )
                            )
                return
            except OperationalError:
                if attempt == MAX_RETRIES - 1:
                    raise
                self._retry_delay(attempt)

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
        """Create an audit once and reject conflicting provenance."""
        if (
            type(github_run_id) is not int
            or github_run_id <= 0
            or type(quality_gate_run_id) is not int
            or quality_gate_run_id <= 0
        ):
            raise ValueError("Invalid GitHub run ID")

        self._validate_version(model_version)
        self._validate_version(previous_version)

        if not isinstance(training_commit, str) or not SHA_RE.fullmatch(
            training_commit
        ):
            raise ValueError("Invalid training commit")

        if approved_by is not None and (
            not isinstance(approved_by, str)
            or not approved_by
            or len(approved_by) > 128
        ):
            raise ValueError("Invalid reviewer")

        values = {
            "github_run_id": github_run_id,
            "model_version": model_version,
            "previous_version": previous_version,
            "training_commit": training_commit.lower(),
            "quality_gate_run_id": quality_gate_run_id,
            "approved_by": approved_by,
            "status": "approved",
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }

        self._insert_if_absent(values)

        with Session(self.engine) as session:
            audit = session.scalar(
                select(DeploymentAudit).where(
                    DeploymentAudit.github_run_id == github_run_id
                )
            )

            if audit is None:
                raise RuntimeError("Audit insertion failed")

            identity = (
                audit.model_version,
                audit.previous_version,
                audit.training_commit.lower(),
                audit.quality_gate_run_id,
                audit.approved_by,
            )

            expected = (
                model_version,
                previous_version,
                training_commit.lower(),
                quality_gate_run_id,
                approved_by,
            )

            if identity != expected:
                raise ValueError(
                    "Existing audit has different provenance or reviewer mismatch"
                )

            return audit.id

    def _record_once(
        self,
        *,
        github_run_id: int,
        status: str,
        render_deployment_id: str | None,
        rollback_authorized: bool,
    ) -> int:
        """Apply one compare-and-swap lifecycle transition."""
        with Session(self.engine) as session:
            with session.begin():
                audit = session.scalar(
                    select(DeploymentAudit).where(
                        DeploymentAudit.github_run_id == github_run_id
                    )
                )

                if audit is None:
                    raise ValueError("Unknown deployment run")

                previous_status = audit.status
                audit_id = audit.id

                if previous_status == status:
                    latest = session.scalar(
                        select(DeploymentAuditEvent)
                        .where(DeploymentAuditEvent.audit_id == audit_id)
                        .order_by(DeploymentAuditEvent.id.desc())
                        .limit(1)
                    )

                    if (
                        latest is not None
                        and latest.render_deployment_id == render_deployment_id
                    ):
                        return audit_id

                    raise ValueError("Conflicting duplicate deployment event")

                if previous_status in TERMINAL_STATUSES:
                    raise ValueError("Cannot transition from terminal state")

                if status not in TRANSITIONS[previous_status]:
                    raise ValueError(
                        f"Invalid transition: {previous_status} -> {status}"
                    )

                if (
                    previous_status == "verified"
                    and status == "rollback_started"
                    and not rollback_authorized
                ):
                    raise ValueError("Rollback requires explicit authorization")

                values = {
                    "status": status,
                    "updated_at": datetime.now(UTC),
                }

                if status == "deploying":
                    values["render_deployment_id"] = render_deployment_id

                # Compare-and-swap:
                # update only if the previously read state
                # is still the current database state.
                statement = (
                    update(DeploymentAudit)
                    .where(
                        DeploymentAudit.id == audit_id,
                        DeploymentAudit.status == previous_status,
                    )
                    .values(**values)
                )

                result = session.execute(statement)

                if result.rowcount != 1:
                    # Another transaction won the transition.
                    # Roll back this transaction and retry.
                    raise _ConcurrentTransition()

                session.add(
                    DeploymentAuditEvent(
                        audit_id=audit_id,
                        status=status,
                        render_deployment_id=render_deployment_id,
                    )
                )

                return audit_id

    def record(
        self,
        *,
        github_run_id: int,
        status: str,
        render_deployment_id: str | None = None,
        rollback_authorized: bool = False,
    ) -> int:
        """Atomically validate and append a lifecycle transition."""
        if type(github_run_id) is not int or github_run_id <= 0:
            raise ValueError("Invalid GitHub run ID")

        if status not in STATUSES:
            raise ValueError("Invalid deployment status")

        self._validate_deployment_id(render_deployment_id)

        if status == "deploying" and render_deployment_id is None:
            raise ValueError("Deployment ID required")

        for attempt in range(MAX_RETRIES):
            try:
                return self._record_once(
                    github_run_id=github_run_id,
                    status=status,
                    render_deployment_id=render_deployment_id,
                    rollback_authorized=rollback_authorized,
                )
            except _ConcurrentTransition:
                if attempt == MAX_RETRIES - 1:
                    raise RuntimeError(
                        "Concurrent deployment transition retry limit exceeded"
                    ) from None
                self._retry_delay(attempt)
            except OperationalError:
                if attempt == MAX_RETRIES - 1:
                    raise
                self._retry_delay(attempt)

        raise RuntimeError("Deployment transition failed")

    def history(self, limit: int = 50) -> list[dict]:
        """Return recent deployment attempts."""
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
                    "render_deployment_id": row.render_deployment_id,
                    "status": row.status,
                    "created_at": row.created_at.isoformat(),
                    "updated_at": row.updated_at.isoformat(),
                }
                for row in rows
            ]

    def events(self, github_run_id: int) -> list[dict]:
        """Return append-only lifecycle events for a run."""
        with Session(self.engine) as session:
            audit = session.scalar(
                select(DeploymentAudit).where(
                    DeploymentAudit.github_run_id == github_run_id
                )
            )

            if audit is None:
                return []

            rows = session.scalars(
                select(DeploymentAuditEvent)
                .where(DeploymentAuditEvent.audit_id == audit.id)
                .order_by(DeploymentAuditEvent.id)
            ).all()

            return [
                {
                    "id": event.id,
                    "status": event.status,
                    "render_deployment_id": (event.render_deployment_id),
                    "timestamp": event.timestamp.isoformat(),
                }
                for event in rows
            ]

    def close(self) -> None:
        """Dispose database connections."""
        self.engine.dispose()


class _ConcurrentTransition(Exception):
    """Internal signal to retry a stale state transition."""

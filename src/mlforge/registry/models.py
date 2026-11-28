from datetime import UTC, datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class RegistryBase(DeclarativeBase):
    """Declarative base for registry tables."""


class RegisteredModel(RegistryBase):
    """Immutable identity and provenance of a model artifact."""

    __tablename__ = "registered_models"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
    )

    sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    source_commit: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )


class ModelDeployment(RegistryBase):
    """Historical record of model deployment decisions."""

    __tablename__ = "model_deployments"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    model_id: Mapped[int] = mapped_column(
        ForeignKey("registered_models.id"),
        nullable=False,
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

    __table_args__ = (
        CheckConstraint(
            "status IN ('candidate', 'approved', 'deployed', 'rolled_back')",
            name="ck_model_deployment_status",
        ),
    )

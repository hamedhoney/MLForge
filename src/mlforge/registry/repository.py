import re

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from mlforge.registry.models import (
    ModelDeployment,
    RegisteredModel,
    RegistryBase,
)

SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
VERSION_PATTERN = re.compile(r"^v\d+\.\d+\.\d+$")


class ModelRegistry:
    """Store immutable model identities and deployment records."""

    def __init__(self, database_url: str):
        self.engine = create_engine(
            database_url,
            pool_pre_ping=True,
        )

    def initialize(self) -> None:
        """Create registry tables for initial development."""
        RegistryBase.metadata.create_all(self.engine)

    def register(
        self,
        *,
        version: str,
        sha256: str,
        source_commit: str,
    ) -> int:
        """Register a model without overwriting existing provenance."""
        if not VERSION_PATTERN.fullmatch(version):
            raise ValueError("Invalid model version")

        if not SHA256_PATTERN.fullmatch(sha256):
            raise ValueError("Invalid model SHA-256")

        if not COMMIT_PATTERN.fullmatch(source_commit):
            raise ValueError("Invalid source commit")

        with Session(self.engine) as session:
            existing = session.scalar(
                select(RegisteredModel).where(RegisteredModel.version == version)
            )

            if existing is not None:
                if (
                    existing.sha256.lower() != sha256.lower()
                    or existing.source_commit.lower() != source_commit.lower()
                ):
                    raise ValueError(
                        "Model version already registered with different provenance"
                    )

                return existing.id

            model = RegisteredModel(
                version=version,
                sha256=sha256.lower(),
                source_commit=source_commit.lower(),
            )

            session.add(model)
            session.commit()
            session.refresh(model)

            return model.id

    def record_status(
        self,
        *,
        model_id: int,
        status: str,
    ) -> int:
        """Append a deployment status event."""
        allowed = {
            "candidate",
            "approved",
            "deployed",
            "rolled_back",
        }

        if status not in allowed:
            raise ValueError("Invalid deployment status")

        with Session(self.engine) as session:
            model = session.get(RegisteredModel, model_id)

            if model is None:
                raise ValueError("Unknown model ID")

            event = ModelDeployment(
                model_id=model_id,
                status=status,
            )

            session.add(event)
            session.commit()
            session.refresh(event)

            return event.id

    def history(self) -> list[dict]:
        """Return deployment events ordered newest first."""
        with Session(self.engine) as session:
            rows = session.execute(
                select(
                    ModelDeployment.id,
                    RegisteredModel.version,
                    ModelDeployment.status,
                    ModelDeployment.created_at,
                )
                .join(
                    RegisteredModel,
                    ModelDeployment.model_id == RegisteredModel.id,
                )
                .order_by(
                    ModelDeployment.created_at.desc(),
                    ModelDeployment.id.desc(),
                )
            ).all()

        return [
            {
                "id": row.id,
                "version": row.version,
                "status": row.status,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]

    def close(self) -> None:
        """Dispose database connections."""
        self.engine.dispose()

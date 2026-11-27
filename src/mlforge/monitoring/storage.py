from datetime import UTC, datetime, timedelta

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class InferenceRecord(Base):
    """Privacy-conscious inference telemetry record."""

    __tablename__ = "inference_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    prediction: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        CheckConstraint("latency_ms >= 0"),
        CheckConstraint("probability >= 0 AND probability <= 1"),
        CheckConstraint("prediction IN (0, 1)"),
        Index("ix_inference_events_timestamp", "timestamp"),
    )


class TelemetryStore:
    """Persist inference events and query historical aggregates."""

    def __init__(self, database_url: str):
        self.engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_recycle=300,
            connect_args=(
                {"connect_timeout": 5} if database_url.startswith("postgresql") else {}
            ),
        )

    def initialize(self) -> None:
        """Create the initial telemetry schema."""
        Base.metadata.create_all(self.engine)

    def record(
        self,
        *,
        model_version: str,
        latency_ms: float,
        probability: float,
        prediction: int,
    ) -> None:
        """Persist one prediction event."""
        import math

        if not model_version or len(model_version) > 64:
            raise ValueError("Invalid model version")
        if not math.isfinite(latency_ms) or latency_ms < 0:
            raise ValueError("Invalid latency")
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid probability")
        if prediction not in (0, 1):
            raise ValueError("Invalid prediction")

        with Session(self.engine) as session:
            session.add(
                InferenceRecord(
                    timestamp=datetime.now(UTC),
                    model_version=model_version,
                    latency_ms=latency_ms,
                    probability=probability,
                    prediction=prediction,
                )
            )
            session.commit()

    def summary(self, days: int = 30) -> dict:
        """Return historical telemetry aggregates."""
        if not 1 <= days <= 365:
            raise ValueError("days must be between 1 and 365")

        cutoff = datetime.now(UTC) - timedelta(days=days)

        with Session(self.engine) as session:
            statement = select(
                func.count(InferenceRecord.id),
                func.avg(InferenceRecord.latency_ms),
                func.avg(InferenceRecord.probability),
                func.avg(InferenceRecord.prediction),
            ).where(InferenceRecord.timestamp >= cutoff)

            count, latency, probability, churn_rate = session.execute(statement).one()

        return {
            "window_days": days,
            "total_predictions": count,
            "mean_latency_ms": float(latency) if latency is not None else None,
            "mean_probability": (
                float(probability) if probability is not None else None
            ),
            "predicted_churn_rate": (
                float(churn_rate) if churn_rate is not None else None
            ),
        }

    def delete_older_than(self, days: int = 90) -> int:
        """Delete expired telemetry records."""
        from sqlalchemy import delete

        if days < 1:
            raise ValueError("days must be positive")

        cutoff = datetime.now(UTC) - timedelta(days=days)

        with Session(self.engine) as session:
            result = session.execute(
                delete(InferenceRecord).where(InferenceRecord.timestamp < cutoff)
            )
            session.commit()
            return result.rowcount or 0

    def close(self) -> None:
        """Dispose database connections."""
        self.engine.dispose()

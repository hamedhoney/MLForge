import pytest

from mlforge.registry.repository import ModelRegistry

SHA = "a" * 64
COMMIT = "b" * 40


@pytest.fixture
def registry(tmp_path):
    database = tmp_path / "registry.db"

    instance = ModelRegistry(f"sqlite:///{database}")
    instance.initialize()

    yield instance

    instance.close()


def test_register_model(registry):
    model_id = registry.register(
        version="v0.3.1",
        sha256=SHA,
        source_commit=COMMIT,
    )

    assert model_id > 0


def test_registration_is_idempotent(registry):
    first = registry.register(
        version="v0.3.1",
        sha256=SHA,
        source_commit=COMMIT,
    )

    second = registry.register(
        version="v0.3.1",
        sha256=SHA,
        source_commit=COMMIT,
    )

    assert first == second


def test_provenance_cannot_change(registry):
    registry.register(
        version="v0.3.1",
        sha256=SHA,
        source_commit=COMMIT,
    )

    with pytest.raises(ValueError):
        registry.register(
            version="v0.3.1",
            sha256="c" * 64,
            source_commit=COMMIT,
        )


def test_record_status(registry):
    model_id = registry.register(
        version="v0.3.1",
        sha256=SHA,
        source_commit=COMMIT,
    )

    registry.record_status(
        model_id=model_id,
        status="candidate",
    )

    history = registry.history()

    assert len(history) == 1
    assert history[0]["version"] == "v0.3.1"
    assert history[0]["status"] == "candidate"


def test_unknown_model_rejected(registry):
    with pytest.raises(ValueError):
        registry.record_status(
            model_id=999,
            status="approved",
        )


def test_invalid_status_rejected(registry):
    model_id = registry.register(
        version="v0.3.1",
        sha256=SHA,
        source_commit=COMMIT,
    )

    with pytest.raises(ValueError):
        registry.record_status(
            model_id=model_id,
            status="invalid",
        )

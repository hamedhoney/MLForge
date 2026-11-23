import hashlib
import os
from pathlib import Path
from urllib.request import urlopen

REPO = "hamedhoney/MLForge"
VERSION = "v0.1.0"

MODEL_DIR = Path("artifacts/tuning")


def download_model() -> None:
    """Download and verify an approved model release."""
    expected = os.environ["MLFORGE_MODEL_SHA256"]

    if len(expected) != 64:
        raise ValueError("Invalid SHA-256 configuration")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    filename = "best_model.joblib"
    destination = MODEL_DIR / filename

    base_url = f"https://github.com/{REPO}/releases/download/{VERSION}"

    with urlopen(f"{base_url}/{filename}", timeout=60) as response:
        content = response.read()

    actual = hashlib.sha256(content).hexdigest()

    if actual.lower() != expected.lower():
        raise ValueError("Model checksum verification failed")

    destination.write_bytes(content)

    with urlopen(f"{base_url}/metadata.json", timeout=30) as response:
        metadata = response.read()

    # Metadata should also be pinned and verified.
    metadata_expected = os.environ["MLFORGE_METADATA_SHA256"]
    if hashlib.sha256(metadata).hexdigest() != metadata_expected.lower():
        raise ValueError("Metadata checksum verification failed")

    (MODEL_DIR / "metadata.json").write_bytes(metadata)

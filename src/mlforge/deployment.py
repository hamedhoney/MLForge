import hashlib
import json
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

from mlforge.model_registry import get_latest_model_version

REPO = "hamedhoney/MLForge"
MODEL_DIR = Path(
    os.getenv("MLFORGE_MODEL_DIR", "artifacts/tuning")
)

REQUIRED_FILES = (
    "best_model.joblib",
    "metadata.json",
)


def _download(url: str) -> bytes:
    """Download a GitHub Release asset."""
    request = Request(
        url,
        headers={"User-Agent": "MLForge"},
    )

    with urlopen(request, timeout=60) as response:
        return response.read()


def _sha256(data: bytes) -> str:
    """Return SHA-256 checksum."""
    return hashlib.sha256(data).hexdigest()


def _parse_checksums(data: bytes) -> dict[str, str]:
    """Parse GitHub Release SHA256SUMS.txt."""
    checksums = {}

    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue

        parts = line.split(maxsplit=1)

        if len(parts) != 2:
            raise ValueError("Invalid checksum manifest")

        digest, filename = parts
        filename = filename.lstrip("*")

        if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError("Invalid SHA-256 digest")

        if filename in checksums:
            raise ValueError("Duplicate checksum entry")

        checksums[filename] = digest.lower()

    return checksums


def download_model() -> str:
    """Download and verify the selected model release.

    Returns:
        The resolved model version.
    """
    version = get_latest_model_version()

    base_url = (
        f"https://github.com/{REPO}/releases/download/{version}"
    )

    manifest = _download(
        f"{base_url}/SHA256SUMS.txt"
    )

    # Optional independent manifest pin.
    expected_manifest = os.getenv(
        "MLFORGE_MANIFEST_SHA256"
    )

    if expected_manifest:
        if _sha256(manifest) != expected_manifest.lower():
            raise ValueError(
                "Release manifest checksum mismatch"
            )

    checksums = _parse_checksums(manifest)

    verified_files = {}

    for filename in REQUIRED_FILES:
        expected = checksums.get(filename)

        if expected is None:
            raise ValueError(
                f"Missing checksum for {filename}"
            )

        content = _download(
            f"{base_url}/{filename}"
        )

        if _sha256(content) != expected:
            raise ValueError(
                f"Checksum mismatch: {filename}"
            )

        verified_files[filename] = content

    metadata = json.loads(
        verified_files["metadata.json"]
    )

    if not isinstance(metadata, dict):
        raise ValueError("Invalid model metadata")

    if "features" not in metadata:
        raise ValueError("Missing model feature schema")

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Stage verified artifacts before replacing existing files.
    with TemporaryDirectory(
        dir=MODEL_DIR
    ) as temporary:
        staging = Path(temporary)

        for filename, content in verified_files.items():
            (staging / filename).write_bytes(content)

        (staging / "version.txt").write_text(
            version,
            encoding="utf-8",
        )

        # Replace metadata and model after verification.
        # A concurrent reader should be coordinated separately.
        for filename in REQUIRED_FILES:
            os.replace(
                staging / filename,
                MODEL_DIR / filename,
            )

        os.replace(
            staging / "version.txt",
            MODEL_DIR / "version.txt",
        )

    return version

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

from mlforge.model_registry import get_latest_model_version

logger = logging.getLogger(__name__)

REPO = "hamedhoney/MLForge"

MODEL_DIR = Path(os.getenv("MLFORGE_MODEL_DIR", "artifacts/tuning"))

REQUIRED_FILES = (
    "best_model.joblib",
    "metadata.json",
)

SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def _download(url: str) -> bytes:
    """Download a GitHub Release artifact."""
    request = Request(
        url,
        headers={
            "User-Agent": "MLForge",
            "Accept": "application/octet-stream",
        },
    )

    with urlopen(request, timeout=60) as response:
        return response.read()


def _sha256(content: bytes) -> str:
    """Calculate SHA-256 digest."""
    return hashlib.sha256(content).hexdigest()


def _parse_checksums(content: bytes) -> dict[str, str]:
    """Parse and validate a SHA256SUMS.txt file."""
    checksums = {}

    for line in content.decode("utf-8").splitlines():
        if not line.strip():
            continue

        parts = line.split(maxsplit=1)

        if len(parts) != 2:
            raise ValueError("Invalid checksum manifest entry")

        digest, filename = parts
        filename = filename.lstrip("*")

        if not SHA256_PATTERN.fullmatch(digest):
            raise ValueError("Invalid SHA-256 digest")

        if not filename or "/" in filename or "\\" in filename:
            raise ValueError("Invalid artifact filename")

        if filename in checksums:
            raise ValueError(f"Duplicate checksum entry: {filename}")

        checksums[filename] = digest.lower()

    if not checksums:
        raise ValueError("Empty checksum manifest")

    return checksums


def download_model() -> str:
    """Download and verify the configured MLForge model release.

    Uses MLFORGE_MODEL_VERSION when configured. Otherwise,
    resolves the latest eligible stable model release.

    Returns:
        The downloaded model version.
    """
    version = get_latest_model_version()

    base_url = f"https://github.com/{REPO}/releases/download/{version}"

    logger.info("Downloading MLForge model release %s", version)

    # Download checksum manifest.
    manifest = _download(f"{base_url}/SHA256SUMS.txt")

    # Optional independently configured manifest checksum.
    expected_manifest = os.getenv("MLFORGE_MANIFEST_SHA256")

    if expected_manifest:
        if not SHA256_PATTERN.fullmatch(expected_manifest):
            raise ValueError("Invalid manifest SHA-256 configuration")

        if _sha256(manifest) != expected_manifest.lower():
            raise ValueError("Release manifest checksum verification failed")

    checksums = _parse_checksums(manifest)

    # Download and verify all required model files.
    verified_files: dict[str, bytes] = {}

    for filename in REQUIRED_FILES:
        expected = checksums.get(filename)

        if expected is None:
            raise ValueError(f"Missing checksum for {filename}")

        content = _download(f"{base_url}/{filename}")

        actual = _sha256(content)

        if actual != expected:
            raise ValueError(f"Checksum verification failed: {filename}")

        verified_files[filename] = content

        logger.info("Verified model artifact: %s", filename)

    # Validate model metadata before writing files.
    metadata = json.loads(verified_files["metadata.json"])

    if not isinstance(metadata, dict):
        raise ValueError("Invalid model metadata")

    features = metadata.get("features")

    if (
        not isinstance(features, list)
        or not features
        or not all(isinstance(f, str) for f in features)
    ):
        raise ValueError("Invalid model feature schema")

    threshold = float(metadata["threshold"])

    if not 0 <= threshold <= 1:
        raise ValueError("Invalid model threshold")

    # Create the destination directory.
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Stage verified artifacts before replacing existing files.
    with TemporaryDirectory(dir=MODEL_DIR) as temporary:
        staging = Path(temporary)

        for filename, content in verified_files.items():
            (staging / filename).write_bytes(content)

        (staging / "version.txt").write_text(
            version,
            encoding="utf-8",
        )

        # Replace files only after successful verification.
        for filename in REQUIRED_FILES:
            os.replace(
                staging / filename,
                MODEL_DIR / filename,
            )

        os.replace(
            staging / "version.txt",
            MODEL_DIR / "version.txt",
        )

    logger.info(
        "MLForge model %s downloaded and verified successfully",
        version,
    )

    return version

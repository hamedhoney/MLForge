import hashlib
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def get_git_commit() -> str:
    """Return the Git commit associated with training."""
    if commit := os.getenv("GITHUB_SHA"):
        return commit

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def get_provenance(dataset_path: Path) -> dict:
    """Collect reproducibility metadata."""
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "mlflow",
        "optuna",
        "kagglehub",
    ]

    versions = {}

    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"

    return {
        "dataset_sha256": sha256_file(dataset_path),
        "dataset_filename": dataset_path.name,
        "dataset_size_bytes": dataset_path.stat().st_size,
        "git_commit": get_git_commit(),
        "python_version": platform.python_version(),
        "packages": versions,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


def save_provenance(
    dataset_path: Path,
    output_path: Path,
) -> dict:
    """Save dataset and environment provenance."""
    provenance = get_provenance(dataset_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )

    return provenance

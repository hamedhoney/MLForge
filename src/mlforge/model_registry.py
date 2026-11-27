import json
import os
import re
from urllib.request import Request, urlopen

REPO = "hamedhoney/MLForge"
VERSION_PATTERN = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")

REQUIRED_ASSETS = {
    "best_model.joblib",
    "metadata.json",
    "SHA256SUMS.txt",
}

def get_latest_model_version() -> str:
    """Return the newest stable release with required model assets."""
    pinned = os.getenv("MLFORGE_MODEL_VERSION")

    if pinned:
        if not VERSION_PATTERN.fullmatch(pinned):
            raise ValueError("Invalid model version")
        return pinned

    releases = []
    page = 1

    while True:
        url = (
            f"https://api.github.com/repos/{REPO}/releases"
            f"?per_page=100&page={page}"
        )

        request = Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "MLForge",
            },
        )

        with urlopen(request, timeout=30) as response:
            batch = json.load(response)

        if not batch:
            break

        releases.extend(batch)

        if len(batch) < 100:
            break

        page += 1

    candidates = []

    for release in releases:
        version = release.get("tag_name", "")

        if not VERSION_PATTERN.fullmatch(version):
            continue

        if release.get("draft") or release.get("prerelease"):
            continue

        assets = {
            asset["name"]
            for asset in release.get("assets", [])
        }

        if not REQUIRED_ASSETS.issubset(assets):
            continue

        version_tuple = tuple(
            int(part) for part in version[1:].split(".")
        )

        candidates.append((version_tuple, version))

    if not candidates:
        raise RuntimeError("No valid MLForge model release found")

    return max(candidates)[1]

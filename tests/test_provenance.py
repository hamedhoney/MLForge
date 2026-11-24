import hashlib
import json

from mlforge.data.provenance import (
    save_provenance,
    sha256_file,
)


def test_sha256_file(tmp_path):
    path = tmp_path / "sample.csv"
    content = b"feature,target\n1,0\n2,1\n"
    path.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()

    assert sha256_file(path) == expected


def test_save_provenance(tmp_path):
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("x,y\n1,2\n")

    output = tmp_path / "provenance.json"

    result = save_provenance(dataset, output)

    assert output.exists()

    stored = json.loads(output.read_text())

    assert stored["dataset_sha256"] == result["dataset_sha256"]
    assert stored["dataset_filename"] == "dataset.csv"
    assert "git_commit" in stored
    assert "python_version" in stored
    assert "packages" in stored

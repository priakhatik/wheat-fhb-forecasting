"""Regression checks for portable, provenance-preserving FHB label generation."""

import csv
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "build_labels.py"


def test_label_generation_is_import_safe_and_runs_outside_checkout(tmp_path):
    project = tmp_path / "checkout"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    copied = scripts / "build_labels.py"
    shutil.copy2(SOURCE, copied)

    # Importing a helper must not create a dataset or touch the author's
    # machine-specific home directory.
    namespace = runpy.run_path(str(copied), run_name="imported_for_test")
    output = project / "raw" / "labels" / "fhb_labels.csv"
    assert not output.exists()
    assert len(namespace["build_label_grid"]()) == 60

    # Running the CLI from another working directory still writes only into
    # the checkout, not into the caller's current directory.
    result = subprocess.run(
        [sys.executable, str(copied)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.is_file()
    assert not (tmp_path / "raw").exists()

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 60
    assert len({(row["state"], row["year"]) for row in rows}) == 60
    known = [row for row in rows if row["severity"]]
    unknown = [row for row in rows if not row["severity"]]
    assert len(known) == 16
    assert len(unknown) == 44
    assert all(row["notes"] == "NEEDS RESEARCH" and not row["source"]
               for row in unknown)
    assert any(row["state"] == "ND" and row["year"] == "2009"
               and float(row["severity"]) == 3 for row in known)


def test_label_generation_is_deterministic(tmp_path):
    project = tmp_path / "checkout"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    copied = scripts / "build_labels.py"
    shutil.copy2(SOURCE, copied)
    subprocess.run([sys.executable, str(copied)], check=True, capture_output=True)
    output = project / "raw" / "labels" / "fhb_labels.csv"
    first = output.read_bytes()
    subprocess.run([sys.executable, str(copied)], check=True, capture_output=True)
    assert output.read_bytes() == first

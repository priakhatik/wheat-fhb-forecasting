"""The label audit must not confuse weather-derived labels with held-out evidence."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.audit_label_evaluation import DEFAULT_DATASET, audit_observed_labels


def test_bundled_recorded_labels_cannot_support_all_year_holdouts():
    report = audit_observed_labels(pd.read_csv(DEFAULT_DATASET))
    assert report["total_rows"] == 60
    assert report["recorded_label_rows"] == 16
    assert report["missing_label_rows"] == 44
    assert report["recorded_non_outbreak_rows"] == 3
    assert report["recorded_outbreak_rows"] == 13
    assert report["leave_one_year_out_training_feasible"] is False
    assert 2023 in report["years_with_single_class_training"]
    fold = next(row for row in report["leave_one_year_out_training_folds"]
                if row["held_out_year"] == 2023)
    assert fold["train_negative_rows"] == 0
    assert fold["train_positive_rows"] == 13


def test_feasible_year_holdouts_are_distinguished_from_confounded_labels():
    frame = pd.DataFrame({
        "state": ["MN", "ND", "MN", "ND", "SD"],
        "year": [2020, 2020, 2021, 2021, 2022],
        "severity": [0, 3, 1, 2, np.nan],
    })
    report = audit_observed_labels(frame)
    assert report["recorded_label_rows"] == 4
    assert report["missing_label_rows"] == 1
    assert report["leave_one_year_out_training_feasible"] is True
    assert report["years_with_single_class_training"] == []
    assert {r["held_out_year"] for r in report["leave_one_year_out_training_folds"]} == {
        2020, 2021
    }


@pytest.mark.parametrize("severity", [4, -1, 1.5, float("inf"), "unknown"])
def test_invalid_recorded_severity_fails_closed(severity):
    frame = pd.DataFrame({
        "state": ["MN", "ND"],
        "year": [2020, 2021],
        "severity": [0, severity],
    })
    with pytest.raises(ValueError, match="severity"):
        audit_observed_labels(frame)


def test_invalid_year_and_missing_columns_are_rejected():
    frame = pd.DataFrame({"state": ["MN"], "year": [np.nan], "severity": [2]})
    with pytest.raises(ValueError, match="year"):
        audit_observed_labels(frame)
    with pytest.raises(ValueError, match="missing required columns"):
        audit_observed_labels(frame.drop(columns="state"))

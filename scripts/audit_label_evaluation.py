"""Audit observed FHB labels before interpreting weather-imputed CV scores.

The project's imputed severity labels are functions of the weather features.
Cross-validating a weather model against those labels does not measure
independent outbreak prediction. This script inspects *recorded* nonmissing
labels and whether leave-one-year-out validation could train both classes.
It does not verify the external provenance of those recorded labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "processed" / "fhb_final_dataset.csv"


def audit_observed_labels(frame: pd.DataFrame) -> dict:
    required = {"state", "year", "severity"}
    if not required.issubset(frame.columns):
        raise ValueError("missing required columns: " + ", ".join(sorted(required - set(frame))))
    if frame.empty:
        raise ValueError("label dataset must contain at least one row")

    try:
        years = pd.to_numeric(frame["year"], errors="raise").to_numpy(dtype=float)
        labels = pd.to_numeric(frame["severity"], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("year and severity must be numeric or severity missing") from exc

    if not np.isfinite(years).all() or (years != np.floor(years)).any():
        raise ValueError("year must be a finite integer")
    known_mask = ~np.isnan(labels)
    known_labels = labels[known_mask]
    if (not np.isfinite(known_labels).all()
            or ((known_labels != np.floor(known_labels))
                | (known_labels < 0) | (known_labels > 3)).any()):
        raise ValueError("recorded severity must be an integer from 0 to 3")

    observed_years = years[known_mask].astype(int)
    observed_outbreaks = known_labels >= 2
    negative_count = int(np.count_nonzero(~observed_outbreaks))
    positive_count = int(np.count_nonzero(observed_outbreaks))
    folds = []
    for year in sorted(set(observed_years.tolist())):
        test_mask = observed_years == year
        train_negatives = negative_count - int(np.count_nonzero(~observed_outbreaks & test_mask))
        train_positives = positive_count - int(np.count_nonzero(observed_outbreaks & test_mask))
        folds.append({
            "held_out_year": int(year),
            "test_rows": int(np.count_nonzero(test_mask)),
            "train_negative_rows": train_negatives,
            "train_positive_rows": train_positives,
            "train_has_both_classes": train_negatives > 0 and train_positives > 0,
        })

    unscorable = [fold["held_out_year"] for fold in folds
                  if not fold["train_has_both_classes"]]
    return {
        "total_rows": int(len(frame)),
        "recorded_label_rows": int(np.count_nonzero(known_mask)),
        "missing_label_rows": int(np.count_nonzero(~known_mask)),
        "recorded_non_outbreak_rows": negative_count,
        "recorded_outbreak_rows": positive_count,
        "leave_one_year_out_training_folds": folds,
        "years_with_single_class_training": unscorable,
        "leave_one_year_out_training_feasible": (
            bool(folds) and negative_count > 0 and positive_count > 0 and not unscorable
        ),
        "interpretation": (
            "Recorded labels have not been independently source-verified. "
            "Weather-imputed labels must not be treated as independent ground truth. "
            "Fold feasibility is not a measured prediction score."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    report = audit_observed_labels(pd.read_csv(args.dataset))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

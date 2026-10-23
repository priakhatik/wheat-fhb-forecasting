"""Offline tests for portable, import-safe FHB prediction."""

import importlib
import sys

import numpy as np
import pandas as pd
import pytest


def test_import_does_not_train_or_prompt(monkeypatch):
    # Importing the documented API must never prompt or train a model.
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("prompted on import"))
    sys.modules.pop("scripts.predict", None)
    module = importlib.import_module("scripts.predict")
    assert module.DATA_PATH.name == "fhb_final_imputed.csv"
    assert module.DATA_PATH.parent.name == "processed"
    assert module.load_model.cache_info().currsize == 0


def test_model_training_works_with_portable_dataset(tmp_path):
    from scripts.predict import load_model

    data_path = tmp_path / "fhb_final_imputed.csv"
    pd.DataFrame({
        "rainy_days": [2, 3, 10, 12],
        "avg_humidity": [40, 45, 75, 80],
        "high_humidity_days": [1, 2, 9, 11],
        "avg_tmax_c": [30, 29, 22, 21],
        "avg_tmin_c": [18, 17, 13, 12],
        "severity": [0, 1, 2, 3],
    }).to_csv(data_path, index=False)

    model = load_model(data_path)
    assert list(model.classes_) == [0, 1]
    assert load_model(data_path) is model


def test_single_class_data_is_rejected(tmp_path):
    from scripts.predict import load_model

    data_path = tmp_path / "single_class.csv"
    pd.DataFrame({
        "rainy_days": [2, 3],
        "avg_humidity": [40, 45],
        "high_humidity_days": [1, 2],
        "avg_tmax_c": [30, 29],
        "avg_tmin_c": [18, 17],
        "severity": [0, 1],
    }).to_csv(data_path, index=False)
    with pytest.raises(ValueError, match="both outbreak classes"):
        load_model(data_path)


def test_missing_features_are_rejected(tmp_path):
    from scripts.predict import load_model

    data_path = tmp_path / "missing.csv"
    pd.DataFrame({"severity": [0, 3]}).to_csv(data_path, index=False)
    with pytest.raises(ValueError, match="missing required training columns"):
        load_model(data_path)


def test_nonfinite_prediction_inputs_are_rejected():
    from scripts.predict import predict_fhb_risk

    with pytest.raises(ValueError, match="finite"):
        predict_fhb_risk(4, np.nan, 3, 24, 15)

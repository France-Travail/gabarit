import json
import os
from pathlib import Path

import pandas as pd
from IPython import display
from sklearn.compose import ColumnTransformer
from {{package_name}}.utils import get_data_path, get_models_path

DATA_PATH = Path(get_data_path())
MODELS_PATH = Path(get_models_path())

DATASET_NAME = "wine"


def verify_exercice_1():
    """Verify first exercice"""
    files = {
        kind: DATA_PATH / f"{DATASET_NAME}_{kind}.csv"
        for kind in ("train", "valid", "test")
    }
    expected_results = {
        "train": {"alcohol": 13.64, "length": 107},
        "valid": {"alcohol": 13.72, "length": 36},
        "test": {"alcohol": 14.37, "length": 35},
    }

    # Verify that files exist
    for file_path in files.values():
        try:
            assert os.path.exists(file_path)
        except AssertionError:
            raise FileNotFoundError(
                f"{file_path} not found. "
                f"Did you use {{package_name}}-scripts/utils/0_split_train_valid_test.py ?"
            )

    # Verify files content
    for file, file_path in files.items():
        df = pd.read_csv(file_path, sep=";")
        try:
            assert df.shape[0] == expected_results[file]["length"]
        except AssertionError:
            raise ValueError(
                f"Unexpected length for {file_path}.  "
                f"Did you correctly use a 0.6 / 0.2 / 0.2 split ?"
            )
        try:
            assert df["alcohol"].iloc[0] == expected_results[file]["alcohol"]
        except AssertionError:
            raise ValueError(
                f"Unexpected value in {file_path}.  "
                f"Did you correctly use 42 as seed ?"
            )

    print("Exercice 1 : OK ✔")


def verify_exercice_2():
    """Verify second exercice"""
    assert os.path.exists(DATA_PATH / f"{DATASET_NAME}_10_samples.csv")

    print("Exercice 2 : OK ✔")


def verify_exercice_3():
    """Verify third exercice"""
    file_path = DATA_PATH / f"{DATASET_NAME}_preprocess_P1.csv"
    assert os.path.exists(file_path), "Did you run 1_preprocess_data.py ?"

    df = pd.read_csv(file_path, sep=";", skiprows=1)
    try:
        assert df["target"].apply(lambda x: x % 1 == 0).all()
    except AssertionError:
        raise ValueError(
            "target should only contain int. Did you use '--target_cols target' ?"
        )

    print("Exercice 3 : OK ✔")


def verify_exercice_4():
    """Verify fourth exercice"""
    try:
        from {{package_name}}.preprocessing.preprocess import preprocess_P2
    except ImportError:
        raise ImportError(
            "Did you create preprocess_P2 in {{package_name}}.preprocessing.preprocess ?"
        )

    from {{package_name}}.preprocessing.preprocess import get_pipelines_dict

    try:
        assert isinstance(get_pipelines_dict()["preprocess_P2"], ColumnTransformer)
    except KeyError:
        raise KeyError("Did you add 'preprocess_P2' to get_pipelines_dict ?")
    except AssertionError:
        raise AssertionError(
            "Value associated to 'preprocess_P2' in get_pipelines_dict should be 'preprocess_P2()'"
        )

    file_path = DATA_PATH / f"{DATASET_NAME}_preprocess_P2.csv"
    try:
        assert os.path.exists(file_path)
    except AssertionError:
        raise AssertionError(
            "Did you run 1_preprocess_data.py with argument '-p preprocess_P2' ?"
        )

    df = pd.read_csv(file_path, sep=";", skiprows=1)
    try:
        assert df["target"].apply(lambda x: x % 1 == 0).all()
    except AssertionError:
        raise ValueError(
            "target should only contain int. Did you use '--target_cols target' ?"
        )

    print("Exercice 4 : OK ✔")


def verify_exercice_5():
    """Verify sixth exercice"""
    train_P1 = DATA_PATH / f"{DATASET_NAME}_train_preprocess_P1.csv"
    valid_P1 = DATA_PATH / f"{DATASET_NAME}_valid_preprocess_P1.csv"

    for file, error_msg in (
        (train_P1, "Did you run 1_preprocess_data.py on wine_train.csv ? "),
        (valid_P1, "Did you run 2_apply_existing_pipeline.py on wine_valid.csv ? "),
    ):
        try:
            assert os.path.exists(file)
        except AssertionError:
            raise FileNotFoundError(error_msg)

    expected_results = {
        train_P1: 0.858,
        valid_P1: 0.964,
    }

    for file, alcohol_first_value in expected_results.items():
        df = pd.read_csv(file, sep=";", skiprows=1)

        try:
            assert abs(df["alcohol"].iloc[0] - alcohol_first_value) < 1e-3
        except AssertionError:
            raise ValueError(
                f"Unexpected value in {file}. Did you use 1_preprocess_data.py on "
                f"{DATASET_NAME}_train.csv and 2_apply_existing_pipeline.py on "
                f"{DATASET_NAME}_valid.csv and {DATASET_NAME}_test.csv ?"
            )

        try:
            assert df["target"].apply(lambda x: x % 1 == 0).all()
        except AssertionError:
            raise ValueError(
                f"Target column in {file} should only contain int. "
                f"Did you use '--target_cols target' ?"
            )

    print("Exercice 5 : OK ✔")


def verify_exercice_6():
    """Verify sixth exercice"""

    models_folders = sorted(
        MODELS_PATH.glob("model_ridge_classifier/model_ridge_classifier_*")
    )

    if not models_folders:
        raise AssertionError(
            "No model_ridge_classifier found. Did you run 3_training_classification.py ?"
        )

    last_model = models_folders[-1]
    last_model_config = last_model / "configurations.json"

    with last_model_config.open("r") as f:
        config = json.load(f)

    filename_valid = config.get("filename_valid", None)
    if filename_valid != "wine_valid_preprocess_P1.csv":
        raise AssertionError(
            f"'filename_valid' is {filename_valid} but should be 'wine_valid_preprocess_P1.csv'. "
            f"Did you use '--filename_valid  wine_valid_preprocess_P1.csv' ?"
        )

    print("Exercice 6 : OK ✔")


def verify_exercice_7():
    """Verify seventh exercice"""

    models_folders = sorted(
        MODELS_PATH.glob("model_lgbm_classifier/model_lgbm_classifier_*")
    )

    if not models_folders:
        raise AssertionError(
            "No model_lgbm_classifier found. Did you properly uncomment ModelLGBMClassifier model ?"
        )

    last_model = models_folders[-1]
    last_model_config = last_model / "configurations.json"

    with last_model_config.open("r") as f:
        config = json.load(f)

    filename_valid = config.get("filename_valid", None)
    if filename_valid != "wine_valid_preprocess_P1.csv":
        raise AssertionError(
            f"'filename_valid' is {filename_valid} but should be 'wine_valid_preprocess_P1.csv'. "
            f"Did you use '--filename_valid  wine_valid_preprocess_P1.csv' ?"
        )

    print("Exercice 7 : OK ✔")


def verify_exercice_8():
    """Verify eighth exercice"""
    predictions_folder = DATA_PATH / "predictions" / "wine_test"

    if not predictions_folder.exists():
        raise AssertionError(
            f"No folder {predictions_folder}. Did you run 4_predict.py on wine_test ?"
        )

    predictions = sorted(predictions_folder.glob("predictions_*"))

    if not predictions:
        raise AssertionError(
            "No prediction found. Did you run 4_predict.py on wine_test ?"
        )

    predictions_found = False
    for last_predictions in predictions:
        last_predictions_config = last_predictions / "configurations.json"

        with last_predictions_config.open("r") as f:
            config = json.load(f)

        if config["model_name"] == "model_ridge_classifier":
            predictions_found = True
            break

    if not predictions_found:
        raise AssertionError(
            "No predictions found for model_ridge_classifier. "
            "Did you use this model to make your predictions ?"
        )

    print("Exercice 8 : OK ✔")

    display.Image(last_predictions / "plots" / "with_y_true_confusion_matrix.png")

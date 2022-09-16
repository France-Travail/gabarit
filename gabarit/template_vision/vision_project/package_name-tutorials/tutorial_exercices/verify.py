import json
import re
from pathlib import Path

from PIL import Image
from {{package_name}}.utils import get_data_path, get_models_path

DATA_PATH = Path(get_data_path())
MODELS_PATH = Path(get_models_path())

DATASET_CLASSIF = "dataset_v3"
DATASET_OBJ_DETECT = "dataset_object_detection"


def verify_exercice_1():
    """Verify first exercice"""
    dataset_path = DATA_PATH / DATASET_CLASSIF

    dataset_train_path = DATA_PATH / (DATASET_CLASSIF + "_train")
    dataset_valid_path = DATA_PATH / (DATASET_CLASSIF + "_valid")
    dataset_test_path = DATA_PATH / (DATASET_CLASSIF + "_test")

    for path in (
        dataset_path,
        dataset_train_path,
        dataset_valid_path,
        dataset_test_path,
    ):
        assert (
            path.exists()
        ), f"{path} not found. Did you run 0_split_train_valid_test.py ?"

    # Verify folders content by checking ratios
    expected_ratios = {
        dataset_train_path: 0.6,
        dataset_valid_path: 0.2,
        dataset_test_path: 0.2,
    }

    img_counts = {path: sum(1 for _ in path.glob("*/*")) for path in expected_ratios}
    img_ratios = {path: n / sum(img_counts.values()) for path, n in img_counts.items()}

    try:
        for path, expected_ratio in expected_ratios.items():
            assert abs(expected_ratio - img_ratios[path]) < 1e-2

    except AssertionError:
        raise AssertionError(
            f"{path} contains {img_ratios[path]:.0%} of images whereas it "
            f"should contains {expected_ratio:.0%}. Did you use '--perc_train 0.6' "
            f"'--perc_valid 0.2' and '--perc_test 0.2' ?"
        )

    print("Exercice 1 : OK ✔")


def verify_exercice_2():
    """Verify second exercice"""
    dataset_train_sample_path = DATA_PATH / (DATASET_CLASSIF + "_train_3_samples")
    dataset_test_sample_path = DATA_PATH / (DATASET_CLASSIF + "_test_3_samples")

    for path in (dataset_train_sample_path, dataset_test_sample_path):
        assert path.exists(), f"{path} not found. Did you run 0_create_samples.py ?"

    print("Exercice 2 : OK ✔")


def verify_exercice_3():
    """Verify second exercice"""
    dataset_train_sample_path = DATA_PATH / (
        DATASET_CLASSIF + "_train_preprocess_convert_rgb"
    )
    dataset_valid_sample_path = DATA_PATH / (
        DATASET_CLASSIF + "_valid_preprocess_convert_rgb"
    )

    for path in (
        dataset_train_sample_path,
        dataset_valid_sample_path,
    ):
        assert path.exists(), f"{path} not found. Did you run 1_preprocess_data.py ?"

    print("Exercice 3 : OK ✔")


def verify_exercice_4():
    """Verify second exercice"""
    models_folders = sorted(
        MODELS_PATH.glob("model_cnn_classifier/model_cnn_classifier_*")
    )

    if not models_folders:
        raise AssertionError(
            "No model_cnn_classifier found. Did you properly use ModelCnnClassifier model ?"
        )

    last_model = models_folders[-1]
    last_model_config = last_model / "configurations.json"

    with last_model_config.open("r") as f:
        config = json.load(f)

    directory_valid = config.get("directory_valid", None)
    directory_valid_expected = f"{DATASET_CLASSIF}_valid_preprocess_convert_rgb"

    if directory_valid != directory_valid_expected:
        raise AssertionError(
            f"'directory_valid' is '{directory_valid}' instead of "
            f"'{directory_valid_expected}'. Did you use "
            f"'--directory_valid {directory_valid_expected}' ?"
        )

    print("Exercice 4 : OK ✔")

    confusion_matrix = last_model / "plots" / "valid_confusion_matrix_normalized.png"

    print("A confusion matrix plot has been automatically produced :\n")

    confusion_matrixt_img = Image.open(confusion_matrix)
    confusion_matrixt_img.show()

    print("\n", confusion_matrix)


def verify_exercice_5():
    """Verify second exercice"""
    models_folders = sorted(
        MODELS_PATH.glob(
            "model_transfer_learning_classifier/model_transfer_learning_classifier_*"
        )
    )

    if not models_folders:
        raise AssertionError(
            "No model_transfer_learning_classifier found. Did you properly use ModelTransferLearningClassifier model ?"
        )

    last_model = models_folders[-1]
    last_model_config = last_model / "configurations.json"

    with last_model_config.open("r") as f:
        config = json.load(f)

    directory_valid = config.get("directory_valid", None)
    directory_valid_expected = f"{DATASET_CLASSIF}_valid_preprocess_convert_rgb"

    if directory_valid != directory_valid_expected:
        raise AssertionError(
            f"'directory_valid' is '{directory_valid}' instead of "
            f"'{directory_valid_expected}'. Did you use "
            f"'--directory_valid {directory_valid_expected}' ?"
        )

    print("Exercice 5 : OK ✔")

    confusion_matrix = last_model / "plots" / "valid_confusion_matrix_normalized.png"

    print("A confusion matrix plot has been automatically produced :\n")

    confusion_matrixt_img = Image.open(confusion_matrix)
    confusion_matrixt_img.show()

    print("\n", confusion_matrix)


def verify_exercice_6():
    """Verify sixth exercice"""
    predictions_folder = (
        DATA_PATH / "predictions" / f"{DATASET_CLASSIF}_test"
    )

    if not predictions_folder.exists():
        raise AssertionError(
            f"No folder {predictions_folder}. "
            f"Did you run 3_predict.py on {DATASET_CLASSIF}_test ?"
        )

    predictions = sorted(predictions_folder.glob("predictions_*"))

    if not predictions:
        raise AssertionError(
            f"No prediction found. Did you run 3_predict.py on "
            f"{DATASET_CLASSIF}_test_preprocess_convert_rgb ?"
        )

    predictions_found = False
    for last_predictions in predictions:
        last_predictions_config = last_predictions / "configurations.json"

        with last_predictions_config.open("r") as f:
            config = json.load(f)

        if config["model_name"] == "model_transfer_learning_classifier":
            predictions_found = True
            break

    if not predictions_found:
        raise AssertionError(
            "No predictions found for model_transfer_learning_classifier. "
            "Did you use this model to make your predictions ?"
        )

    print("Exercice 6 : OK ✔")

    confusion_matrix = (
        last_predictions / "plots" / "test_confusion_matrix_normalized.png"
    )

    print("A confusion matrix plot has been automatically produced :\n")

    confusion_matrixt_img = Image.open(confusion_matrix)
    confusion_matrixt_img.show()

    print("\n", confusion_matrix)


def verify_exercice_7():
    """Verify seventh exercice"""
    # Train / Validation / Test splits
    train_path = DATA_PATH / (DATASET_OBJ_DETECT + "_train")
    valid_path = DATA_PATH / (DATASET_OBJ_DETECT + "_valid")
    test_path = DATA_PATH / (DATASET_OBJ_DETECT + "_test")

    for path in (train_path, valid_path, test_path):
        assert (
            path.exists()
        ), f"{path} not found. Did you run 0_split_train_valid_test.py ?"

    # Training
    models_folders = sorted(
        MODELS_PATH.glob(
            "model_detectron_faster_rcnn_object_detector/"
            "model_detectron_faster_rcnn_object_detector_*"
        )
    )

    if not models_folders:
        raise AssertionError(
            "No model_detectron_faster_rcnn_object_detector found. "
            "Did you run 2_training_object_detector.py ?"
        )

    # Prediction
    predictions_folder = DATA_PATH / "predictions" / f"{DATASET_OBJ_DETECT}_test"

    if not predictions_folder.exists():
        raise AssertionError(
            f"No folder {predictions_folder}. "
            f"Did you run 3_predict.py on {DATASET_OBJ_DETECT}_test ?"
        )

    predictions = sorted(predictions_folder.glob("predictions_*"))

    if not predictions:
        raise AssertionError(
            f"No prediction found. Did you run 3_predict.py on {DATASET_OBJ_DETECT}_test ?"
        )

    predictions_found = False
    for last_predictions in predictions:
        last_predictions_config = last_predictions / "configurations.json"

        with last_predictions_config.open("r") as f:
            config = json.load(f)

        if config["model_name"] == "model_detectron_faster_rcnn_object_detector":
            predictions_found = True
            break

    if not predictions_found:
        raise AssertionError(
            "No predictions found for model_detectron_faster_rcnn_object_detector. "
            "Did you use this model to make your predictions ?"
        )

    print("Exercice 7 : OK ✔")

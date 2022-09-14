from IPython import display


def print_answer(answer: str, language: str = "bash"):
    """Print the answer

    Answers are always code
    """
    return display.Code(answer.strip(), language=language)


def answer_exercice_1():
    return print_answer(
        f"""
# do not forget to activate your virtual environment
# source venv_num_template/bin/activate

python {{package_name}}-scripts/utils/0_split_train_valid_test.py -d dataset_v3 --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2
"""
    )


def answer_exercice_2():
    return print_answer(
        f"""
# do not forget to activate your virtual environment
# source venv_num_template/bin/activate

python {{package_name}}-scripts/utils/0_create_samples.py -n 3 -d dataset_v3_train dataset_v3_test
"""
    )


def answer_exercice_3():
    return print_answer(
        f"""
# do not forget to activate your virtual environment
# source venv_num_template/bin/activate

python {{package_name}}-scripts/1_preprocess_data.py -p preprocess_convert_rgb -d dataset_v3_train dataset_v3_valid
"""
    )


def answer_exercice_4():
    return print_answer(
        f"""
# do not forget to activate your virtual environment
# source venv_num_template/bin/activate

python {{package_name}}-scripts/2_training_classifier.py -d dataset_v3_train_preprocess_convert_rgb --directory_valid dataset_v3_valid_preprocess_convert_rgb
"""
    )


def answer_exercice_5():
    return print_answer(
        f"""
if model is None:
    model = model_transfer_learning_classifier.ModelTransferLearningClassifier(
        batch_size=64,
        epochs=5,
        validation_split=0.2,
        patience=10,
        width=224,
        height=224,
        depth=3,
        color_mode="rgb",
        in_memory=False,
        with_fine_tune=False, # fine-tunning chews up memory
        level_save=level_save,
    )
""",
        language="python",
    )


def answer_exercice_5_training():
    return print_answer(
        f"""
# do not forget to activate your virtual environment
# source venv_num_template/bin/activate

python {{package_name}}-scripts/2_training_classifier.py -d dataset_v3_train_preprocess_convert_rgb --directory_valid dataset_v3_valid_preprocess_convert_rgb
"""
    )


def answer_exercice_6():
    return print_answer(
        f"""
# do not forget to activate your virtual environment
# source venv_num_template/bin/activate

model="$(ls {{package_name}}-models/model_transfer_learning_classifier | grep model_transfer_learning_classifier_ | tail -n 1)"

python {{package_name}}-scripts/3_predict.py -m "$model" -d dataset_v3_test
"""
    )


def answer_exercice_7():
    return print_answer(
        f"""
# do not forget to activate your virtual environment
# source venv_num_template/bin/activate

# Train / Validation / Test splits
python {{package_name}}-scripts/utils/0_split_train_valid_test.py -d dataset_object_detection

# Training
python {{package_name}}-scripts/2_training_object_detector.py -d dataset_object_detection_train --directory_valid dataset_object_detection_valid

# Prediction
model="$(ls {{package_name}}-models/model_detectron_faster_rcnn_object_detector | grep model_detectron_faster_rcnn_object_detector_ | tail -n 1)"

python {{package_name}}-scripts/3_predict.py -m "$model" -d dataset_object_detection_test
"""
    )

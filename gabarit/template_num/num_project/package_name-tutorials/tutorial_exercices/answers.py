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

python {{package_name}}-scripts/utils/0_split_train_valid_test.py -f wine.csv --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --seed 42 --overwrite
"""
    )


def answer_exercice_2():
    return print_answer(
        f"""
# do not forget to activate your virtual environment
# source venv_num_template/bin/activate

python {{package_name}}-scripts/utils/0_create_samples.py -f wine.csv -n 10
"""
    )


def answer_exercice_3():
    return print_answer(
        f"""
# do not forget to activate your virtual environment
# source venv_num_template/bin/activate

python {{package_name}}-scripts/utils/0_sweetviz_report.py -s wine_train.csv -c wine_test.csv
"""
    )


def answer_exercice_4():
    return print_answer(
        f"""
# do not forget to activate your virtual environment
# source venv_num_template/bin/activate

python {{package_name}}-scripts/1_preprocess_data.py -f wine.csv --target_cols target
"""
    )


def answer_exercice_5_preprocess_P2():
    return print_answer(
        f"""
from sklearn.preprocessing import MinMaxScaler

def get_pipelines_dict() -> dict:
    '''Gets a dictionary of available preprocessing pipelines

    Returns:
        dict: Dictionary of preprocessing pipelines
    '''
    pipelines_dict = {{ '{{' }}
        'no_preprocess': ColumnTransformer([('identity', FunctionTransformer(lambda x: x), make_column_selector())]),
        'preprocess_P1': preprocess_P1(),
        'preprocess_P2': preprocess_P2()
    {{ '}}' }}
    return pipelines_dict

def preprocess_P2() -> ColumnTransformer:
    '''Gets "default" preprocessing pipeline
    Returns:
        ColumnTransformer: The pipeline
    '''
    numeric_pipeline = make_pipeline(SimpleImputer(strategy='median'), MinMaxScaler())
    transformers = [
        ('num', numeric_pipeline, make_column_selector(dtype_include='number')),
    ]
    pipeline = ColumnTransformer(transformers, sparse_threshold=0, remainder='drop')
    return pipeline
""",
        language="python",
    )


def answer_exercice_5_preprocess_script():
    return print_answer(
        f"""
# do not forget to activate your virtual environment
# source venv_num_template/bin/activate

python {{package_name}}-scripts/1_preprocess_data.py -f wine.csv -p preprocess_P2 --target_cols target
"""
    )


def answer_exercice_6():
    return print_answer(
        f"""
# do not forget to activate your virtual environment
# source venv_num_template/bin/activate

python {{package_name}}-scripts/1_preprocess_data.py -f wine_train.csv -p preprocess_P1 --target_cols target

# Get last pipeline fitted in {{package_name}}-pipelines
pipeline="$(ls {{package_name}}-pipelines | grep preprocess_P1 | tail -n 1)"

python {{package_name}}-scripts/2_apply_existing_pipeline.py -f wine_valid.csv -p "$pipeline" --target_cols target
"""
    )


def answer_exercice_7():
    return print_answer(
        f"""
# do not forget to activate your virtual environment
# source venv_num_template/bin/activate

python {{package_name}}-scripts/3_training_classification.py -f wine_train_preprocess_P1.csv --filename_valid  wine_valid_preprocess_P1.csv -y target
"""
    )


def answer_exercice_8():
    return print_answer(
        f"""
# in {{package_name}}-scripts/3_training_classification.py :

if model is None:
    model = model_lgbm_classifier.ModelLGBMClassifier(
        x_col=x_col,
        y_col=y_col,
        level_save=level_save,
        preprocess_pipeline=preprocess_pipeline,
        lgbm_params={{ '{{' }}
            "num_leaves": 31,
            "max_depth": -1,
            "learning_rate": 0.1,
            "n_estimators": 100,
        {{ '}}' }},
        multi_label=multi_label,
    )
""",
        language="python",
    )


def answer_exercice_9():
    return print_answer(
        f"""
# do not forget to activate your virtual environment
# source venv_num_template/bin/activate

model="$(ls {{package_name}}-models/model_ridge_classifier | grep model_ridge_classifier_ | tail -n 1)"

python {{package_name}}-scripts/4_predict.py -f wine_test.csv -y target -m "$model"
"""
    )


def answer_exercice_10_preprocess_P3():

    return print_answer(
        f"""
def get_pipelines_dict() -> dict:
    '''Gets a dictionary of available preprocessing pipelines

    Returns:
        dict: Dictionary of preprocessing pipelines
    '''
    pipelines_dict = {{ '{{' }}
        'no_preprocess': ColumnTransformer([('identity', FunctionTransformer(lambda x: x), make_column_selector())]),
        'preprocess_P1': preprocess_P1(),
        'preprocess_P2': preprocess_P2(),
        'preprocess_P3': preprocess_P3()
    {{ '}}' }}
    return pipelines_dict


def preprocess_P3() -> ColumnTransformer:
    '''Gets "default" preprocessing pipeline
    Returns:
        ColumnTransformer: The pipeline
    '''
    numeric_pipeline = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())
    cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
    transformers = [
        ("num", numeric_pipeline, make_column_selector(pattern="^(?!cultivar).*$", dtype_include="number")),
        ("cat", cat_pipeline, make_column_selector(pattern="cultivar")),
    ]
    pipeline = ColumnTransformer(transformers, sparse_threshold=0, remainder="drop")
    return pipeline
""",
        language="python",
    )


def answer_exercice_10_scripts():

    return print_answer(
        f"""
# do not forget to activate your virtual environment
# source venv_num_template/bin/activate

### train / valid / test splits
python {{package_name}}-scripts/utils/0_split_train_valid_test.py -f wine_reg.csv --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --overwrite

### preprocessing training data
python {{package_name}}-scripts/1_preprocess_data.py -f wine_reg_train.csv -p preprocess_P3 --target_cols alcohol

### preprocessing validation data
# Get last pipeline fitted in {{package_name}}-pipelines
pipeline="$(ls {{package_name}}-pipelines | grep preprocess_P3 | tail -n 1)"

python {{package_name}}-scripts/2_apply_existing_pipeline.py -f wine_reg_valid.csv -p "$pipeline" --target_cols alcohol

### train a regressor
python {{package_name}}-scripts/3_training_regression.py -f wine_reg_train_preprocess_P3.csv --filename_valid  wine_reg_valid_preprocess_P3.csv -y alcohol

### predict test data
model="$(ls {{package_name}}-models/model_knn_regressor | grep model_knn_regressor_ | tail -n 1)"

python {{package_name}}-scripts/4_predict.py -f wine_reg_test.csv -y alcohol -m "$model"
"""
    )

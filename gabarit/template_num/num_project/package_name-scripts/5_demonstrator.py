# Copyright (C) <2018-2022>  <Agence Data Services, DSI Pôle Emploi>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Ex: streamlit run 5_demonstrator.py


import os
# BY DEFAULT, GPU USAGE IS DISABLED
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import sys
import time
import logging
import numpy as np
import pandas as pd
from PIL import Image
from typing import Union, List, Type, Tuple

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess
from {{package_name}}.models_training import utils_models
from {{package_name}}.models_training.model_class import ModelClass
from {{package_name}}.monitoring.model_explainer import ShapExplainer

# TMP FIX: somehow, a json method prevents us to cache most of our models with Streamlit
# That was not the case before, something must have changed within a third party library ?
# Anyway, we'll just add "hash_funcs={'_json.Scanner': hash}" to st.cache when needed.
# https://docs.streamlit.io/library/advanced-features/caching#the-hash_funcs-parameter
# https://github.com/streamlit/streamlit/issues/4876

# Get logger
logger = logging.getLogger('{{package_name}}.5_demonstrator')


# ---------------------
# Streamlit.io confs
# ---------------------

try:
    import streamlit as st
except ImportError as e:
    logger.error("Can't import streamlit library")
    logger.error("Please install it on your virtual env (check the correct version in the requirements.txt -> `pip install streamlit==...`")
    sys.exit("Can't import streamlit")

try:
    import altair as alt
except ImportError as e:
    logger.error("Can't import altair library")
    logger.error("Please install it on your virtual env (check the correct version in the requirements.txt -> `pip install altair==...`")
    sys.exit("Can't import altair")

if not st._is_running_with_streamlit:
    logger.error('This script should not be run directly with python, but via streamlit')
    logger.error('e.g. "streamlit run 5_demonstrator.py')
    sys.exit("Streamlit not started")


# ---------------------
# Streamlit CSS update
# ---------------------

# We increase the sidebar size
css = '''
<style>
.sidebar.--collapsed .sidebar-content {
    margin-left: -30rem;
}
.sidebar .sidebar-content {
    width: 30rem;
}
code {
    display: block;
    white-space: pre-wrap;
}
</style>
'''
st.markdown(css, unsafe_allow_html=True)


# ---------------------
# Manage session variables
# ---------------------

if 'content' not in st.session_state:
    st.session_state['content'] = None


# ---------------------
# Utils functions
# ---------------------

@st.cache(allow_output_mutation=True, hash_funcs={'_json.Scanner': hash})
def load_model(selected_model: str) -> Tuple[Type[ModelClass], dict]:
    '''Loads a model

    Args:
        selected_model(str): Model to be loaded - directory name
    Returns:
        model (ModelClass): Loaded model
        model_conf (dict): The model's configuration
    '''
    model, model_conf = utils_models.load_model(selected_model)
    # We force a first predict to init. the inference time
    # https://github.com/keras-team/keras/issues/8724
    tmp_df = pd.DataFrame({col:[0] for col in model.x_col})
    model.predict(tmp_df)
    return model, model_conf


@st.cache(allow_output_mutation=True)
def get_available_models() -> List[str]:
    '''Gets all available models

    Returns:
        list<str>: All available models
    '''
    # Start with an empty list
    models_list = []
    # Find models
    models_dir = utils.get_models_path()
    for path, subdirs, files in os.walk(models_dir):
        # Check presence of a .pkl file (should be sufficient)
        if len([f for f in files if f.endswith('.pkl')]) > 0:
            models_list.append(os.path.basename(path))
    models_list = sorted(models_list)
    return models_list


@st.cache
def get_model_conf_text(model_conf: dict, list_classes: List[str]) -> str:
    '''Gets informations to be displayed about a model

    Args:
        model_conf (dict): The model's configuration
        list_classes (list<str>): The model's classes list
    Returns:
        str: Markdown text to be displayed
    '''
    markdown_content = "---  \n"
    markdown_content += f"Task : {model_conf['model_type']}  \n"
    if 'multi_label' in model_conf.keys() and model_conf['multi_label'] is not None:
        markdown_content += f"Multi-label : {model_conf['multi_label']}  \n"
    markdown_content += f"Training date : {model_conf['date']}  \n"
    markdown_content += f"Model type : {model_conf['model_name']}  \n"
    markdown_content += "---  \n"

    # Mandatory columns
    markdown_content += "Model's mandatory columns :\n"
    for col in model_conf['mandatory_columns']:
        markdown_content += f"- {col} \n"
    markdown_content += "---  \n"

    # Classifier
    if model_conf['model_type'] == 'classifier':
        markdown_content += "Model's labels : \n"
        if model_conf['multi_label']:
            for cl in list_classes:
                markdown_content += f"- {cl} \n"
        else:
            markdown_content += f"- {model_conf['y_col']} \n"
            for cl in list_classes:
                markdown_content += f"  - {cl} \n"

    # Return
    return markdown_content


def get_prediction(model: Type[ModelClass], content: pd.DataFrame) -> Tuple[Union[str, np.ndarray, float], np.ndarray, float]:
    '''Gets prediction on a content for a given model

    Args:
        model (ModelClass): Model to use
        content (pd.DataFrame): Input content
    Raises:
        ValueError: If invalid model type
    Returns:
        (str | np.ndarray | float): Prediction on the input content
            str if classifier mono-label
            np.ndarray if multi-labels classifier
            float if regressor
        (np.ndarray): Probabilities if classifier, else None
        (float): Prediction time
    '''
    start_time = time.time()

    # Preprocess
    if model.preprocess_pipeline is not None:
        df_prep = utils_models.apply_pipeline(content, model.preprocess_pipeline)
    else:
        df_prep = content.copy()

    # Get prediction
    if model.model_type == 'classifier':
        predictions, probas = model.predict_with_proba(df_prep)
        prediction = predictions[0]
        probas = probas[0]
    elif model.model_type == 'regressor':
        prediction =  model.predict(df_prep)[0]
        probas = None
    else:
        raise ValueError("Invalid model type")

    # Return with prediction time
    prediction_time = time.time() - start_time
    return prediction, probas, prediction_time


def get_prediction_formatting_text(model: Type[ModelClass], model_conf: dict, prediction: Union[str, np.ndarray, float], probas: np.ndarray) -> str:
    '''Formatting prediction into markdown str

    Args:
        model (ModelClass): Model to use
        model_conf (dict): The model's configuration
        prediction (str | float): Model's prediction
            str if classifier mono-label
            np.ndarray if multi-labels classifier
            float if regressor
        probas (np.ndarray): Probabilities if classifier, else None
    Raises:
        ValueError: If invalid model type
    Returns:
        (str): Markdown text to be displayed
    '''
    markdown_content = ''
    if model.model_type == 'classifier':
        if not model_conf['multi_label']:
            prediction_inversed = model.inverse_transform([prediction])[0]
            markdown_content = f"- {model_conf['y_col']}: **{prediction_inversed}**  \n"
            markdown_content += f"  - Probability : {round(probas[model.list_classes.index(prediction_inversed)] * 100, 2)} %  \n"
        else:
            # TODO: add a maximum limit to the number of classes
            markdown_content = ""
            for i, cl in enumerate(model.list_classes):
                if prediction[i] == 0:
                    markdown_content += f"- ~~{cl}~~  \n"
                else:
                    markdown_content += f"- **{cl}**  \n"
                markdown_content += f"  - Probability : {round(probas[i] * 100, 2)} %  \n"
    elif model.model_type == 'regressor':
        # TODO: later, manage multi-output
        markdown_content = f"- {model_conf['y_col']}: **{prediction}**  \n"
    else:
        raise ValueError("Invalid model type")
    return markdown_content


def get_histogram(probas: np.ndarray, list_classes: List[str], is_multi_label: bool) -> Tuple[pd.DataFrame, alt.LayerChart]:
    '''Gets a probabilities histogram (to be plotted)

    Args:
        probas (np.ndarray): Probabilities
        list_classes (list<str>): The model's classes list
        is_multi_label (bool): If the model is multi-labels or not
    Returns:
        pd.DataFrame: Dataframe with class/probability pairs
        alt.LayerChart: Histogram
    '''
    # Get dataframe
    if is_multi_label:
        predicted = ['Accepted' if proba >= 0.5 else 'Rejected' for proba in probas]
    else:
        max_proba = max(probas)
        predicted = ['Accepted' if proba == max_proba else 'Rejected' for proba in probas]
    df_probabilities = pd.DataFrame({'classes': list_classes, 'probabilities': probas, 'result': predicted})

    # Prepare plot
    domain = ['Accepted', 'Rejected']
    range_ = ['#1f77b4', '#d62728']
    bars = (
        alt.Chart(df_probabilities, width=720, height=80 * len(list_classes))
        .mark_bar()
        .encode(
            x=alt.X('probabilities:Q', scale=alt.Scale(domain=(0, 1))),
            y='classes:O',
            color=alt.Color('result', scale=alt.Scale(domain=domain, range=range_)),
            tooltip=['probabilities:Q', 'classes:O'],
        )
    )
    # Nudges text to the right so it does not appear on top of the bar
    text = bars.mark_text(align='left', baseline='middle', dx=3)\
               .encode(text=alt.Text('probabilities:Q', format='.2f'))

    return df_probabilities, alt.layer(bars + text)


def get_explanation(model: Type[ModelClass], content: pd.DataFrame, class_or_label_index: Union[int, None] = None) -> str:
    '''Explains the model's prediction on a given content

    Args:
        model (ModelClass): Model to use
        content (pd.DataFrame): Input content
    Kwargs:
        class_or_label_index (int): for classification only. Class or label index to be considered.
    Returns:
        (str): HTML content to be rendered
    '''
    # Check for anchor data
    anchor_data_path = os.path.join(model.model_dir, 'original_data_samples.csv')
    if os.path.exists(anchor_data_path):
        anchor_data = pd.read_csv(anchor_data_path, sep='{{default_sep}}', encoding='{{default_encoding}}')
    else:
        return "<p> No anchor data available for this model, can't produce an explanation</p>"
    # Explain via SHAP explainer
    logger.info("Explain results via SHAP")
    explainer = ShapExplainer(model, anchor_data, anchor_preprocessed=False)
    html = explainer.explain_instance_as_html(content, class_or_label_index)
    return html


# ---------------------
# Streamlit.io App
# ---------------------


st.title('Demonstrator for project {{package_name}}')
st.image(Image.open(os.path.join(utils.get_ressources_path(), 'robot.jpg')), width=200)
st.markdown("---  \n")

# Sidebar (model selection)
st.sidebar.title('Model')
selected_model = st.sidebar.selectbox('Model selection', get_available_models(), index=0)

# Add a checkbox to get explanation
checkbox_explanation = st.sidebar.checkbox('With explanation', False)


# Get model
if selected_model is not None:

    # ---------------------
    # Read the model
    # ---------------------

    start_time = time.time()
    model, model_conf = load_model(selected_model)
    model_loading_time = time.time() - start_time
    st.write(f"Model loading time: {round(model_loading_time, 2)}s (warning, can be cached by the application)")
    st.markdown("---  \n")

    # ---------------------
    # Get model confs
    # ---------------------

    list_classes = model.list_classes if hasattr(model, 'list_classes') else None
    markdown_content = get_model_conf_text(model_conf, list_classes)
    st.sidebar.markdown(markdown_content)

    # ---------------------
    # Inputs
    # ---------------------

    form = st.form(key='my-form')

    # ---------------------
    # Data
    # ---------------------


    # TODO TODO TODO TODO TODO
    # TODO TODO TODO TODO TODO
    # TODO: TO BE CHANGED WITH YOUR DATA
    # TODO: Here is some examples with the "wine" dataset from tutorial
    form.write("Input data")
    form_values = {col: form.number_input(col) for col in model.mandatory_columns}
    form.markdown("---  \n")
    # TODO TODO TODO TODO TODO
    # TODO TODO TODO TODO TODO

    # ---------------------
    # GO Button
    # ---------------------

    # Prediction starts by clicking this button
    submit = form.form_submit_button("Predict")
    if submit:
        # Construct content from inputs
        # TODO TODO TODO TODO TODO
        # TODO: TO BE CHANGED WITH YOUR DATA
        content = pd.DataFrame({col: [col_value] for col, col_value in form_values.items()})
        # TODO TODO TODO TODO TODO
        st.session_state.content = content


    # ---------------------
    # Button clear
    # ---------------------

    # Clear everything by clicking this button
    if st.button("Clear"):
        st.session_state.content = None


    # ---------------------
    # Checks
    # ---------------------

    # In some cases, if input form features are automatically generated from the model's mendatory columns,
    # st.session_state.content could still be defined with previous model's input columns
    # Hence, we check if all mendatory columns are in st.session_state.content, and reset it to None if it is not the case
    if st.session_state.content is not None and any([col not in st.session_state.content.columns for col in model.mandatory_columns]):
        st.session_state.content = None
        logger.warning("Input content had been reset because it does not match the model's mendatory columns")
        logger.warning("You probably just changed the model, try to submit a new content")


    # ---------------------
    # Prediction
    # ---------------------

    # Prediction and results diplay
    if st.session_state.content is not None:
        st.write("---  \n")
        st.markdown("## Results  \n")
        st.markdown("  \n")

        # ---------------------
        # Prediction
        # ---------------------

        prediction, probas, prediction_time = get_prediction(model, st.session_state.content)
        st.write(f"Prediction (inference time : {int(round(prediction_time*1000, 0))}ms) :")

        # ---------------------
        # Format prediction
        # ---------------------

        markdown_content = get_prediction_formatting_text(model, model_conf, prediction, probas)
        st.markdown(markdown_content)

        # ---------------------
        # Histogram probabilities - Classifier
        # ---------------------

        if model.model_type == 'classifier':
            df_probabilities, altair_layer = get_histogram(probas, model.list_classes, model.multi_label)
            # Display dataframe probabilities & plot altair
            st.subheader('Probabilities histogram')
            st.write(df_probabilities)
            st.altair_chart(altair_layer)

        # ---------------------
        # Explainer
        # ---------------------

        if checkbox_explanation:

            st.write("---  \n")
            st.subheader('Explanation')

            # Get shap explanations
            if model.model_type == 'classifier':
                # Set form
                form_explanation = st.form(key='my-form-explanation')
                inv_dict = {v: k for k, v in model.dict_classes.items()}
                index_max = probas.argmax()
                if model.multi_label:
                    form_explanation.write("Select the label to be explained")
                    selected_class_or_label = form_explanation.selectbox("Label :", ['Highest prediction score label'] + model.list_classes, index=0)
                    inv_dict['Highest prediction score label'] = index_max
                else:
                    form_explanation.write("Class to be explained")
                    selected_class_or_label = form_explanation.selectbox("Class :", ['Predicted class'] + model.list_classes, index=0)
                    inv_dict['Predicted class'] = index_max
                # Set submit button
                submit_explanation = form_explanation.form_submit_button("Explain")
                # On click, get explanation
                if submit_explanation:
                    class_or_label_index = inv_dict[selected_class_or_label]
                    html = get_explanation(model, st.session_state.content, class_or_label_index)
                else:
                    html = None
            else:
                # Automatically get explanation
                html = get_explanation(model, st.session_state.content, None)

            # If html set ...
            if html is not None:
                # Add some css
                html += "<style>.shap_fig {max-width: 100%}</style>"
                # Chosen class or label
                if model.model_type == 'classifier':
                    st.write(f"Explanation for {'label' if model.multi_label else 'class'} {model.dict_classes[class_or_label_index]}")
                # Display html
                st.components.v1.html(html, height=800, scrolling=True)  # We could pby have a better height

        st.write("---  \n")

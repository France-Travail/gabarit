# Model demonstrator
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
# Ex: streamlit run 4_demonstrateur.py

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
from {{package_name}}.monitoring.model_explainer import LimeExplainer
from {{package_name}}.monitoring.model_explainer import AttentionExplainer
from {{package_name}}.models_training.model_class import ModelClass

# TMP FIX: somehow, a json method prevents us to cache most of our models with Streamlit
# That was not the case before, something must have changed within a third party library ?
# Anyway, we'll just add "hash_funcs={'_json.Scanner': hash}" to st.cache when needed.
# https://docs.streamlit.io/library/advanced-features/caching#the-hash_funcs-parameter
# https://github.com/streamlit/streamlit/issues/4876

# Get logger
logger = logging.getLogger('{{package_name}}.4_demonstrator')


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
    logger.error('e.g. "streamlit run 4_demonstrateur.py')
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

if 'text_areas_content' not in st.session_state:
    st.session_state['text_areas_content'] = None


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
    model.predict(['test'])
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
    markdown_content += f"Multi-label : {model_conf['multi_label']}  \n"
    markdown_content += f"Training date : {model_conf['date']}  \n"
    markdown_content += f"Model type : {model_conf['model_name']}  \n"
    markdown_content += "---  \n"
    markdown_content += "Model's labels : \n"
    if model_conf['multi_label']:
        for cl in list_classes:
            markdown_content += f"- {cl} \n"
    else:
        markdown_content += f"- {model_conf['y_col']} \n"
        for cl in list_classes:
            markdown_content += f"  - {cl} \n"
    return markdown_content


def get_prediction(model: Type[ModelClass], model_conf: dict, content: str) -> Tuple[Union[str, np.ndarray], np.ndarray, str, float]:
    '''Gets prediction on a content for a given model

    Args:
        model (ModelClass): Model to use
        model_conf (dict): The model's configuration
        content (str): Input content
    Returns:
        (str | np.ndarray): Prediction on the input content
            str if classifier mono-label
            np.ndarray if multi-labels classifier
        (np.ndarray): Probabilities
        (str): Preprocessed content
        (float): Prediction time
    '''
    start_time = time.time()

    # Get preprocessor
    if 'preprocess_str' in model_conf.keys():
        preprocess_str = model_conf['preprocess_str']
    else:
        preprocess_str = 'no_preprocess'
    preprocessor = preprocess.get_preprocessor(preprocess_str)

    # Preprocess
    content_preprocessed = preprocessor(content)

    # Get prediction (some models need an iterable)
    predictions, probas = model.predict_with_proba([content_preprocessed])

    # Get prediction
    prediction = predictions[0]
    probas = probas[0]

    # Return with prediction time
    prediction_time = time.time() - start_time
    return prediction, probas, content_preprocessed, prediction_time


def get_prediction_formatting_text(model: Type[ModelClass], model_conf: dict, prediction: Union[str, np.ndarray], probas: np.ndarray) -> str:
    '''Formatting prediction into markdown str

    Args:
        model (ModelClass): Model to use
        model_conf (dict): The model's configuration
        prediction (str | np.ndarray): Model's prediction
            str if classifier mono-label
            np.ndarray if multi-labels classifier
        probas (np.ndarray): Probabilities if classifier, else None
    Returns:
        (str): Markdown text to be displayed
    '''
    markdown_content = ''
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


def get_explanation(model: Type[ModelClass], model_conf: dict, content: str, is_multi_label: bool, probas: np.ndarray) -> str:
    '''Explains the model's prediction on a given content

    Args:
        model (ModelClass): Model to use
        model_conf (dict): The model's configuration
        content (str): Input content
        is_multi_label (bool): If the model is multi-labels or not
        probas (np.ndarray): Probabilities
    Returns:
        (str): HTML content to be rendered
    '''
    if hasattr(model, 'explain'):
        logger.info("Explain results via the model's own explain function")
        exp = AttentionExplainer(model)
    else:
        logger.info("Explain results via LIME")
        exp = LimeExplainer(model, model_conf)
    if is_multi_label:  # multi-labels : compare the two main probas
        classes = [model.list_classes[i] for i in np.argsort(probas)[-2:][::-1]]
    else:
        classes = model.list_classes
    html = exp.explain_instance_as_html(text=content, classes=classes)
    return html


# ---------------------
# Streamlit.io App
# ---------------------


st.title('Demonstrator for project {{package_name}}')
st.image(Image.open(os.path.join(utils.get_ressources_path(), 'nlp.jpg')), width=200)
st.markdown("---  \n")

# Sidebar (model selection)
st.sidebar.title('Model')
selected_model = st.sidebar.selectbox('Model selection', get_available_models(), index=0)

# Add a button to have multiple lines input
checkbox_multilines = st.sidebar.checkbox('Multiple lines', False)

# If not checked, we allow several entries at once
if not checkbox_multilines:
    nb_entries = st.sidebar.slider("Number of entries", 1, 5)

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

    markdown_content = get_model_conf_text(model_conf, model.list_classes)
    st.sidebar.markdown(markdown_content)

    # ---------------------
    # Inputs
    # ---------------------

    form = st.form(key='my-form')

    # ---------------------
    # Text input
    # ---------------------

    if checkbox_multilines:
        text_areas = [form.text_area('Input text')]
    else:
        text_areas = [form.text_input(f'Input text n° {i}') for i in range(nb_entries)]
    form.markdown("---  \n")

    # ---------------------
    # GO Button
    # ---------------------

    # Prediction starts by clicking this button
    submit = form.form_submit_button("Predict")
    if submit:
        st.session_state.text_areas_content = text_areas


    # ---------------------
    # Button clear
    # ---------------------

    # Clear everything by clicking this button
    if st.button("Clear"):
        st.session_state.text_areas_content = None


    # ---------------------
    # Prediction
    # ---------------------

    # Prediction and results diplay
    if st.session_state.text_areas_content is not None and st.session_state.text_areas_content != '':
        st.write("---  \n")
        st.markdown("## Results  \n")
        st.markdown("  \n")

        # Process contents one by one
        for content in st.session_state.text_areas_content:
            if len(content) > 0:

                # ---------------------
                # Prediction
                # ---------------------

                prediction, probas, content_preprocessed, prediction_time = get_prediction(model, model_conf, content)

                st.write(f"Original text: `{content}`")
                st.write(f"Preprocessed text: `{content_preprocessed}`")
                st.write(f"Prediction (inference time : {int(round(prediction_time*1000, 0))}ms) :")

                # ---------------------
                # Format prediction
                # ---------------------

                markdown_content = get_prediction_formatting_text(model, model_conf, prediction, probas)
                st.markdown(markdown_content)

                # ---------------------
                # Histogram probabilities
                # ---------------------

                df_probabilities, altair_layer = get_histogram(probas, model.list_classes, model.multi_label)

                # Display dataframe probabilities & plot altair
                st.subheader('Probabilities histogram')
                st.write(df_probabilities)
                st.altair_chart(altair_layer)

                # ---------------------
                # Explainer
                # ---------------------

                if checkbox_explanation:
                    html = get_explanation(model, model_conf, content, model.multi_label, probas)
                    st.components.v1.html(html, height=500)

                st.write("---  \n")

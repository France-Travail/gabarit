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
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
from lime import lime_image
from matplotlib import pyplot as plt
from typing import Union, List, Type, Tuple
from skimage.segmentation import mark_boundaries

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess
from {{package_name}}.models_training import utils_models
from {{package_name}}.models_training.model_class import ModelClass
from {{package_name}}.models_training.object_detectors import utils_object_detectors

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

if 'input_image_file_area' not in st.session_state:
    st.session_state['input_image_file_area'] = None


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
    def_image_path = os.path.join(utils.get_ressources_path(), 'robot.jpg')
    model.predict(pd.DataFrame({'file_path': [def_image_path]}))
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
        list_classes (list): The model's classes list
    Returns:
        str: Markdown text to be displayed
    '''
    markdown_content = "---  \n"
    markdown_content += f"Task : {model_conf['model_type']}  \n"
    markdown_content += f"Training date : {model_conf['date']}  \n"
    markdown_content += f"Model type : {model_conf['model_name']}  \n"
    markdown_content += "---  \n"
    markdown_content += "Model's labels : \n"
    for cl in list_classes:
        markdown_content += f"  - {cl} \n"
    return markdown_content


def get_prediction(model: Type[ModelClass], model_conf: dict, img: Image.Image) -> Tuple[Union[str, list], np.ndarray, Image.Image, float]:
    '''Gets prediction on an image for a given model

    Args:
        model (ModelClass): Model to use
        model_conf (dict): The model's configuration
        img (Image.Image): input image
    Raises:
        ValueError: If invalid model type
    Returns:
        (str | list): prediction on the input image
            str if classifier
            list of bboxes if object_detector
        (np.ndarray): Probabilities if classifier, else None
        (Image.Image): preprocessed image
        (float): Prediction time
    '''
    start_time = time.time()

    # Get preprocessor
    if 'preprocess_str' in model_conf.keys():
        preprocess_str = model_conf['preprocess_str']
    else:
        preprocess_str = "no_preprocess"
    preprocessor = preprocess.get_preprocessor(preprocess_str)

    # Preprocess
    img_preprocessed = preprocessor([img])[0]

    # We'll create a temporary folder to save preprocessed images
    with tempfile.TemporaryDirectory(dir=utils.get_data_path()) as tmp_folder:
        # Save preprocessed image
        im_path = os.path.join(tmp_folder, 'image.png')
        img_preprocessed.save(im_path, format='PNG')
        # Get prediction
        df = pd.DataFrame({'file_path': [im_path]})
        if model_conf['model_type'] == 'classifier':
            predictions, probas = model.predict_with_proba(df)
            prediction = predictions[0]
            probas = probas[0]
        elif model_conf['model_type'] == 'object_detector':
            prediction = model.predict(df)[0]
            probas = None
        else:
            raise ValueError("Invalid model type")

    # Getting out of the context, all temporary data is deleted

    # Return with prediction time
    prediction_time = time.time() - start_time
    return prediction, probas, img_preprocessed, prediction_time


def get_prediction_formatting_text(model: Type[ModelClass], model_conf: dict, prediction: Union[str, list], probas: np.ndarray) -> str:
    '''Formatting prediction into markdown str

    Args:
        model (ModelClass): Model to use
        model_conf (dict): The model's configuration
        prediction (str | list): model'sprediction
            str if classifier
            list of bboxes if object_detector
        probas (np.ndarray): Probabilities if classifier, else None
    Raises:
        ValueError: If invalid model type
    Returns:
        (str): Markdown text to be displayed
    '''
    markdown_content = ''
    if model_conf['model_type'] == 'classifier':
        prediction_inversed = model.inverse_transform([prediction])[0]
        markdown_content = f"- **{prediction_inversed}**  \n"
        markdown_content += f"  - Probability : {round(probas[model.list_classes.index(prediction_inversed)] * 100, 2)} %  \n"
    elif model_conf['model_type'] == 'object_detector':
        for bbox in prediction:
            markdown_content += f"- **{bbox['class']}** - Probability : {round(bbox['proba'] * 100, 2)} % \n"
    else:
        raise ValueError("Invalid model type")

    # Return
    return markdown_content


def get_img_with_bbox(img: Image.Image, bboxes: List[dict]) -> Image.Image:
    '''Adds predicted bboxes on original image

    Args:
        img (Image.Image): input Image
        bboxes (list<dict>): list of predicted bboxes
    Returns:
        Original image with predicted bboxes drawn on it
    '''
    mode = img.mode
    if mode not in {'RGB', 'RGBA'}:
        img = img.convert('RGB')
        mode = 'RGB'
    img_2 = np.array(img)
    if mode == 'RGB':
        color = (0, 255, 0)
    else:
        color = (0, 255, 0, 255)
    for bbox in bboxes:
        for coord in ['x1', 'x2', 'y1', 'y2']:
            bbox[coord] = int(bbox[coord])
        utils_object_detectors.draw_rectangle_from_bbox(img_2, bbox, color=color, thickness=3, with_center=True)
    return Image.fromarray(img_2, mode=mode)


def explain_image(model: Type[ModelClass], model_conf: dict, img: Image.Image) -> plt.Figure:
    '''Explains the model's prediction on a given image

    Args:
        model (ModelClass): Model to use
        model_conf (dict): The model's configuration
        img (Image.Image): input Image
    Returns:
        (plt.Figure): Figure to be displayed
    '''
    # Get lime explainer
    explainer = lime_image.LimeImageExplainer()

    # Set predict proba function
    def classifier_fn_lime(images: np.ndarray) -> np.ndarray:
        '''Function to be used by Lime, returns probas per classes

        Args:
            images (np.ndarray):array of images
        Returns:
            np.array: probabilities
        '''
        # Preprocess images
        images = [Image.fromarray(img, 'RGB') for img in images]
        if 'preprocess_str' in model_conf.keys():
            preprocess_str = model_conf['preprocess_str']
        else:
            preprocess_str = "no_preprocess"
        preprocessor = preprocess.get_preprocessor(preprocess_str)
        images_preprocessed = preprocessor(images)
        # Temporary folder
        with tempfile.TemporaryDirectory(dir=utils.get_data_path()) as tmp_folder:
            # Save images
            images_path = [os.path.join(tmp_folder, f'image_{i}.png') for i in range(len(images_preprocessed))]
            for i, img_preprocessed in enumerate(images_preprocessed):
                img_preprocessed.save(images_path[i], format='PNG')
            # Get predictions
            df = pd.DataFrame({'file_path': images_path})
            predictions, probas = model.predict_with_proba(df)
        # Return probas
        return probas

    # Get explanation (images must be convert into rgb, then into np array)
    explanation = explainer.explain_instance(np.array(img.convert('RGB')), classifier_fn_lime, hide_color=0, num_samples=100, batch_size=100)

    # Preprare figure & return
    new_im_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=15, hide_rest=False)
    new_im_2, mask_2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=15, hide_rest=True)
    new_im_3, mask_3 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=15, hide_rest=False)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(30,30))
    ax1.imshow(img.convert('RGB'))
    ax2.imshow(mark_boundaries(new_im_1, mask_1))
    ax3.imshow(mark_boundaries(new_im_2, mask_2))
    ax4.imshow(mark_boundaries(new_im_3, mask_3))
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')
    return fig


def get_histogram(probas: np.ndarray, list_classes: List[str]) -> Tuple[pd.DataFrame, alt.LayerChart]:
    '''Gets a probabilities histogram (to be plotted)

    Args:
        probas (np.ndarray): Probabilities
        list_classes (list<str>): The model's classes list
    Returns:
        pd.DataFrame: Dataframe with class/probability pairs
        alt.LayerChart: Histogram
    '''
    # Get dataframe
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

    markdown_content = get_model_conf_text(model_conf, model.list_classes)
    st.sidebar.markdown(markdown_content)

    # ---------------------
    # Inputs
    # ---------------------

    form = st.form(key='my-form')

    # ---------------------
    # Image
    # ---------------------

    form.write("Image to be processed")
    input_image_file_area = form.file_uploader("Image to be processed - file extension .jpg, .jpeg or .png", type=['jpg', 'jpeg', 'png'])
    form.markdown("---  \n")

    # ---------------------
    # GO Button
    # ---------------------

    # Prediction starts by clicking this button
    submit = form.form_submit_button("Predict")
    if submit:
        st.session_state.input_image_file_area = input_image_file_area


    # ---------------------
    # Button clear
    # ---------------------

    # Clear everything by clicking this button
    if st.button("Clear"):
        st.session_state.input_image_file_area = None


    # ---------------------
    # Prediction
    # ---------------------

    # Prediction and results diplay
    if st.session_state.input_image_file_area is not None:
        st.write("---  \n")
        st.markdown("## Results  \n")
        st.markdown("  \n")

        # ---------------------
        # Loading image
        # ---------------------

        img = Image.open(st.session_state.input_image_file_area)

        # ---------------------
        # Prediction
        # ---------------------

        prediction, probas, img_preprocessed, prediction_time = get_prediction(model, model_conf, img)
        st.write(f"Prediction (inference time : {int(round(prediction_time*1000, 0))}ms) :")

        st.image(img, caption="Input image")
        st.image(img_preprocessed, caption="Preprocessed Image")
        # Plot prediction if object detector
        if model_conf['model_type'] == 'object_detector':
            predicted_img = get_img_with_bbox(img, prediction)
            st.image(predicted_img, caption="Prediction")

        # ---------------------
        # Format prediction
        # ---------------------

        markdown_content = get_prediction_formatting_text(model, model_conf, prediction, probas)
        st.markdown(markdown_content)

        # ---------------------
        # Classifier only - display histogram & get explanations
        # ---------------------
        if model_conf['model_type'] == 'classifier':

            # ---------------------
            # Histogram probabilities
            # ---------------------

            df_probabilities, altair_layer = get_histogram(probas, model.list_classes)

            # Display dataframe probabilities & plot altair
            st.subheader('Probabilities histogram')
            st.write(df_probabilities)
            st.altair_chart(altair_layer)

            # ---------------------
            # Explainer
            # ---------------------

            if checkbox_explanation:
                st.write("Explenation")
                fig = explain_image(model, model_conf, img)
                st.pyplot(fig)

        st.write("---  \n")

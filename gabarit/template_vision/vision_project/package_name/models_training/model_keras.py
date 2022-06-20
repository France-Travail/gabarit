#!/usr/bin/env python3

## Generic model for Keras NN
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
# Classes :
# - ModelKeras -> Generic model for Keras NN


# Cf. fix https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import os
import time
import json
import ntpath
import shutil
import logging
import functools
import numpy as np
import pandas as pd
import seaborn as sns
import dill as pickle
from PIL import Image
from sklearn.model_selection import train_test_split
from typing import no_type_check, Union, Callable, Any

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (CSVLogger, EarlyStopping,
                                        ModelCheckpoint, TensorBoard,
                                        TerminateOnNaN, LearningRateScheduler)

from {{package_name}} import utils
from {{package_name}}.models_training import utils_deep_keras
from {{package_name}}.models_training.model_class import ModelClass

sns.set(style="darkgrid")


class ModelKeras(ModelClass):
    '''Generic model for Keras NN'''

    _default_name = 'model_keras'
    nb_iter_keras: int
    # Not implemented :
    # -> _get_model
    # -> reload_from_standalone

    # Should pby be overridden :
    # -> _get_preprocess_input

    def __init__(self, batch_size: int = 64, epochs: int = 99, validation_split: float = 0.2, patience: int = 5,
                 width: int = 224, height: int = 224, depth: int = 3, color_mode: str = 'rgb',
                 in_memory: bool = False, data_augmentation_params: dict = {},
                 nb_train_generator_images_to_save: int = 20,
                 keras_params: dict = {}, **kwargs) -> None:
        '''Initialization of the class (see ModelClass for more arguments)

        Kwargs:
            batch_size (int): Batch size
            epochs (int): Number of epochs
            validation_split (float): Percentage for the validation set split
                Only used if no input validation set when fitting
            patience (int): Early stopping patience
            width (int): NN input width (images are resized)
            height (int): NN input height (images are resized)
            depth (int): NN input depth
            color_mode (str): NN input color mode
            in_memory (bool): If all images should be loaded in memory, otherwise it uses a generator
                /!\ OOM errors can happen really quickly (depends on the dataset size)
                /!\ Data augmentation impossible if `in_memory` is set to True
            data_augmentation_params (dict): Dictionnary of parameters to be used with the data augmentation
                cf. https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
                /!\ Not used if `in_memory` is set to True
            nb_train_generator_images_to_save (int): If > 0, save some input generated images
                If helps with to understand what goes in your NN
            keras_params (dict): Parameters used by keras models.
                e.g. learning_rate, nb_lstm_units, etc...
                The purpose of this dictionary is for the user to use it as they wants in the _get_model function
                This parameter was initially added in order to do an hyperparameters search
        Raises:
            ValueError: If `in_memory` is set to True and `data_augmentation_params` is not empty
        '''
        # TODO: learning rate should be an attribute !

        # Check for errors
        if in_memory and len(data_augmentation_params) > 0:
            raise ValueError("Data augmentation is impossible for 'in_memory' mode")

        # Init.
        super().__init__(**kwargs)

        # Fix tensorflow GPU
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Param. model
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.patience = patience

        # Params. generator
        self.width = width
        self.height = height
        self.depth = depth
        self.color_mode = color_mode
        self.in_memory = in_memory
        self.data_augmentation_params = data_augmentation_params.copy()

        # Warnings if depth does not match with color_mode
        if self.color_mode == 'rgb' and self.depth != 3:
            self.logger.warning(f"`color_mode` parameter is 'rgb', but `depth` parameteris not equal to 3 ({self.depth})")
            self.logger.warning("We continue, but this can lead to errors during the training")
        if self.color_mode == 'rgba' and self.depth != 4:
            self.logger.warning(f"`color_mode` parameter is 'rgba', but `depth` parameteris not equal to 4 ({self.depth})")
            self.logger.warning("We continue, but this can lead to errors during the training")

        # TODO: add Test time augmentation ?

        # Misc.
        self.nb_train_generator_images_to_save = nb_train_generator_images_to_save

        # Model set on fit
        self.model: Any = None

        # Set preprocess input
        self.preprocess_input = self._get_preprocess_input()

        # Keras params
        self.keras_params = keras_params.copy()

        # Keras custom objects : we get the ones specified in utils_deep_keras
        self.custom_objects = utils_deep_keras.custom_objects

    def fit(self, df_train: pd.DataFrame, df_valid: Union[pd.DataFrame, None] = None, with_shuffle: bool = True, **kwargs) -> dict:
        '''Fits the model

        Args:
            df_train (pd.DataFrame): Train dataset
                Must contain file_path & file_class columns if classifier
                Must contain file_path & bboxes columns if object detector
        Kwargs:
            df_valid (pd.DataFrame): Validation dataset
                Must contain file_path & file_class columns if classifier
                Must contain file_path & bboxes columns if object detector
            with_shuffle (boolean): If the train dataset must be shuffled
                This should be used if the input dataset is not shuffled & no validation set as the split_validation takes the lines in order.
                Thus, the validation set might get classes which are not in the train set ...
        Raises:
            NotImplementedError: If the model is not `classifier` nor `object_detector`
        Returns:
            dict: Fit arguments, to be used with transfer learning fine-tuning
        '''
        if self.model_type == 'classifier':
            return self._fit_classifier(df_train, df_valid=df_valid, with_shuffle=with_shuffle, **kwargs)
        elif self.model_type == 'object_detector':
            return self._fit_object_detector(df_train, df_valid=df_valid, with_shuffle=with_shuffle, **kwargs)
        else:
            raise NotImplementedError("Only `classifier` and `object_detector` model type are supported.")

    def _fit_classifier(self, df_train: pd.DataFrame, df_valid: pd.DataFrame = None, with_shuffle: bool = True, **kwargs) -> dict:
        '''Fits the model - classifier

        Args:
            df_train (pd.DataFrame): Train dataset
                Must contain file_path & file_class columns
        Kwargs:
            df_valid (pd.DataFrame): Validation dataset
                Must contain file_path & file_class columns
            with_shuffle (boolean): If the train dataset must be shuffled
                This should be used if the input dataset is not shuffled & no validation set as the split_validation takes the lines in order.
                Thus, the validation set might get classes which are not in the train set ...
        Raises:
            ValueError: If the model is not of type `classifier`
            AssertionError: If already trained and new dataset does not match model's classes
        Returns:
            dict: Fit arguments, to be used with transfer learning fine-tuning
        '''
        if self.model_type != 'classifier':
            raise ValueError(f"`_fit_classifier` function does not support model type {self.model_type}")

        ##############################################
        # Manage retrain
        ##############################################

        # If a model has already been fitted, we make a new folder in order not to overwrite the existing one !
        # And we save the old conf
        if self.trained:
            # Get src files to save
            src_files = [os.path.join(self.model_dir, "configurations.json")]
            if self.nb_fit > 1:
                for i in range(1, self.nb_fit):
                    src_files.append(os.path.join(self.model_dir, f"configurations_fit_{i}.json"))
            # Change model dir
            self.model_dir = self._get_model_dir()
            # Get dst files
            dst_files = [os.path.join(self.model_dir, f"configurations_fit_{self.nb_fit}.json")]
            if self.nb_fit > 1:
                for i in range(1, self.nb_fit):
                    dst_files.append(os.path.join(self.model_dir, f"configurations_fit_{i}.json"))
            # Copies
            for src, dst in zip(src_files, dst_files):
                try:
                    shutil.copyfile(src, dst)
                except Exception as e:
                    self.logger.error(f"Impossible to copy {src} to {dst}")
                    self.logger.error("We still continue ...")
                    self.logger.error(repr(e))

        ##############################################
        # Prepare dataset
        # Also extract list of classes
        ##############################################

        # Extract list of classes from df_train
        list_classes = sorted(list(df_train['file_class'].unique()))
        # Also set dict_classes
        dict_classes = {i: col for i, col in enumerate(list_classes)}

        # Validate classes if already trained, else set them
        if self.trained:
            assert self.list_classes == list_classes, \
                "Error: the new dataset does not match with the already fitted model"
            assert self.dict_classes == dict_classes, \
                "Error: the new dataset does not match with the already fitted model"
        else:
            self.list_classes = list_classes
            self.dict_classes = dict_classes

        # Shuffle training dataset if wanted
        # It is advised as validation_split from keras does not shufle the data
        # Hence, for classificationt task, we might have classes in the validation data that we never met in the training data
        if with_shuffle:
            df_train = df_train.sample(frac=1.).reset_index(drop=True)

        if df_valid is None:
            self.logger.warning(f"Warning, no validation set. The training set will be splitted (validation fraction = {self.validation_split})")

        ##############################################
        # We save some preprocessed / augmented input images examples
        ##############################################

        # Save some examples
        if self.nb_train_generator_images_to_save > 0:
            self.logger.info("Retrieving a generator to save some preprocessed / augmented input images examples")
            # 1. Retrieve a generator (if in_memory, use 'valid' to avoid augmentation)
            if not self.in_memory:
                tmp_gen = self._get_generator(df_train, data_type='train', batch_size=1)
            else:
                tmp_gen = self._get_generator(df_train, data_type='valid', batch_size=1)
            # 2. Retrieve generated images one by one
            images = [tmp_gen.next()[0][0] for i in range(self.nb_train_generator_images_to_save)]
            # 3. Remove negative pixels
            min_pixel = min([np.min(_) for _ in images])
            if min_pixel < 0:
                images = [arr - min_pixel for arr in images]
            # 4. Rescale and scale uint8
            max_pixel = max([np.max(_) for _ in images])
            images = [(arr * 255 / max_pixel).astype('uint8') for arr in images]
            # 5. Cast back to image format
            images = [Image.fromarray(arr, 'RGBA' if arr.shape[-1] == 4 else 'RGB') for arr in images]
            # 6. Save
            save_dir = os.path.join(self.model_dir, f'examples_fit_{self.nb_fit + 1}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for i, im in enumerate(images):
                im_path = os.path.join(save_dir, f'example_{i}.png')
                im.save(im_path, format='PNG')

        ##############################################
        # Get generators if not in_memory, else get full data
        # Finally fit the model
        ##############################################

        if not self.in_memory:
            self.logger.info("Loading data via generators")

            # Create generators
            if df_valid is not None:
                self.logger.info("Retrieving a generator for the training set")
                train_generator = self._get_generator(df_train, data_type='train', batch_size=min(self.batch_size, len(df_train)))
                self.logger.info("Retrieving a generator for the validation set")
                valid_generator = self._get_generator(df_valid, data_type='valid', batch_size=min(self.batch_size, len(df_valid)))
                # Set dataset related args
                steps_per_epoch_arg = len(df_train) // min(self.batch_size, len(df_train))
                validation_steps_arg = len(df_valid) // min(self.batch_size, len(df_valid))
            else:
                # If no validation, we'll split the training set using validation_split attribute
                df_train_split, df_valid_split = train_test_split(df_train, test_size=self.validation_split)
                self.logger.info("Retrieving a generator for the training set")
                train_generator = self._get_generator(df_train_split, data_type='train', batch_size=min(self.batch_size, len(df_train_split)))
                self.logger.info("Retrieving a generator for the validation set")
                valid_generator = self._get_generator(df_valid_split, data_type='valid', batch_size=min(self.batch_size, len(df_valid_split)))
                # Set dataset related args
                steps_per_epoch_arg = len(df_train_split) // min(self.batch_size, len(df_train_split))
                validation_steps_arg = len(df_valid_split) // min(self.batch_size, len(df_valid_split))

            # Get fit arguments
            x_arg = train_generator
            y_arg = None
            batch_size_arg = None
            # validation_data does work with generators (TensorFlow doc is not up to date)
            validation_data_arg = valid_generator
            validation_split_arg = None

        # Load in memory - Can easily lead to OOM issues
        else:
            self.logger.info("Loading data in memory")

            # We retrieve all the data
            # Trick: we still use generators to have the correct preprocessing
            # -> data_type = valid (no shuffle, no data augmentation)
            train_generator = self._get_generator(df_train, data_type='valid', batch_size=len(df_train))
            x_train, y_train = train_generator.next()
            if df_valid is not None:
                valid_generator = self._get_generator(df_valid, data_type='valid', batch_size=min(self.batch_size, len(df_valid)))
                x_val, y_val = valid_generator.next()
                validation_data = (x_val, y_val)
            else:
                validation_data = None

            # Get fit arguments
            x_arg = x_train
            y_arg = y_train
            batch_size_arg = self.batch_size
            steps_per_epoch_arg = None
            validation_data_arg = validation_data  # Can be None if no validation set
            validation_steps_arg = None
            validation_split_arg = self.validation_split if validation_data is None else None

        # Get model (if already fitted we do not load a new one)
        if not self.trained:
            self.model = self._get_model()

        # Get callbacks (early stopping & checkpoint)
        callbacks = self._get_callbacks()

        # Fit
        # We use a try...except in order to save the model if an error arises
        # after more than a minute into training
        start_time = time.time()
        try:
            fit_arguments = {
                'x': x_arg,
                'y': y_arg,
                'batch_size': batch_size_arg,
                'steps_per_epoch': steps_per_epoch_arg,
                'validation_data': validation_data_arg,
                'validation_split': validation_split_arg,
                'validation_steps': validation_steps_arg,
            }
            fit_history = self.model.fit(  # type: ignore
                epochs=self.epochs,
                callbacks=callbacks,
                verbose=1,
                **fit_arguments,
            )
        except (RuntimeError, SystemError, SystemExit, EnvironmentError, KeyboardInterrupt, tf.errors.ResourceExhaustedError, tf.errors.InternalError,
                tf.errors.UnavailableError, tf.errors.UnimplementedError, tf.errors.UnknownError, Exception) as e:
            # Steps:
            # 1. Display tensorflow error
            # 2. Check if more than one minute elapsed & not several iterations & existence best.hdf5
            # 3. Reload best model
            # 4. We consider that a fit occured (trained = True, nb_fit += 1)
            # 5. Save & create a warning file
            # 6. Display error messages
            # 7. Raise an error

            # 1.
            self.logger.error(repr(e))

            # 2.
            best_path = os.path.join(self.model_dir, 'best.hdf5')
            time_spent = time.time() - start_time
            if time_spent >= 60 and self.nb_iter_keras == 1 and os.path.exists(best_path):
                # 3.
                self.model = load_model(best_path, custom_objects=self.custom_objects)
                # 4.
                self.trained = True
                self.nb_fit += 1
                # 5.
                self.save()
                with open(os.path.join(self.model_dir, "0_MODEL_INCOMPLETE"), 'w'):
                    pass
                with open(os.path.join(self.model_dir, "1_TRAINING_NEEDS_TO_BE_RESUMED"), 'w'):
                    pass
                # 6.
                self.logger.error("[EXPERIMENTAL] Error during model training")
                self.logger.error(f"[EXPERIMENTAL] The error happened after {round(time_spent, 2)}s of training")
                self.logger.error("[EXPERIMENTAL] A saving of the model is done but this model won't be usable as is.")
                self.logger.error(f"[EXPERIMENTAL] In order to resume the training, we have to specify this model ({ntpath.basename(self.model_dir)}) in the file 2_training.py")
                self.logger.error("[EXPERIMENTAL] Warning, the preprocessing is not saved in the configuration file")
                self.logger.error("[EXPERIMENTAL] Warning, the best model might be corrupted in some cases")
            # 7.
            raise RuntimeError("Error during model training")

        # Print accuracy & loss if level_save > 'LOW'
        if self.level_save in ['MEDIUM', 'HIGH']:
            self._plot_metrics_and_loss(fit_history)
            # Reload best model
            self.model = load_model(
                os.path.join(self.model_dir, 'best.hdf5'),
                custom_objects=self.custom_objects
            )

        # Set trained
        self.trained = True
        self.nb_fit += 1

        # Return fit arguments. This is useful for transfer learning algorithms
        return fit_arguments

    def _fit_object_detector(self, df_train: pd.DataFrame, df_valid: pd.DataFrame = None, with_shuffle: bool = True, **kwargs) -> dict:
        '''Fits the model - object detector

        Args:
            df_train (pd.DataFrame): Train dataset
                Must contain file_path & bboxes columns
        Kwargs:
            df_valid (pd.DataFrame): Validation dataset
                Must contain file_path & bboxes columns
            with_shuffle (boolean): If the train dataset must be shuffled
        Raises:
            ValueError: If the model is not of type `object_detector`
        '''
        raise NotImplementedError("'_fit_object_detector' needs to be overridden")

    @utils.trained_needed
    def predict(self, df_test: pd.DataFrame, return_proba: bool = False, batch_size: Union[int, None] = None) -> Union[np.ndarray, list]:
        '''Predictions on test set

        Args:
            df_test (pd.DataFrame): DataFrame to be predicted, with column file_path
        Kwargs:
            return_proba (bool): If the function should return the probabilities instead of the classes -- classifier only
            batch_size (int): Batch size to be used -- classifier only
        Raises:
            NotImplementedError: If the model is not `classifier` nor `object_detector`
        Returns:
            (np.ndarray | list): Array, shape = [n_samples, n_classes] or List of n_samples elements
        '''
        if self.model_type == 'classifier':
            return self._predict_classifier(df_test, return_proba=return_proba, batch_size=batch_size)
        elif self.model_type == 'object_detector':
            return self._predict_object_detector(df_test)
        else:
            raise NotImplementedError("Only 'classifier' and 'object_detector' model type are supported")

    @utils.trained_needed
    def _predict_classifier(self, df_test, return_proba: bool = False, batch_size: int = None) -> np.ndarray:
        '''Predictions on test set

        Args:
            df_test (pd.DataFrame): DataFrame to be predicted, with column file_path
        Kwargs:
            return_proba (bool): If the function should return the probabilities instead of the classes
            batch_size (int): Batch size to be used
        Raises:
            ValueError: If the model is not a classifier
        Returns:
            (np.ndarray): Array, shape = [n_samples, n_classes]
        '''
        if self.model_type != 'classifier':
            raise ValueError(f"`_predict_classifier` function does not support model type {self.model_type}")

        # Backup on training batch size if no batch size defined
        if batch_size is None:
            batch_size = self.batch_size

        # Get generator or fulldata if in_memory
        if not self.in_memory:
            self.logger.info("Retrieving a generator for test data")
            test_generator = self._get_generator(df_test, data_type='test', batch_size=min(batch_size, len(df_test)))
            # Get predict arguments
            x_arg = test_generator
            batch_size_arg = None
        else:
            self.logger.info("Retrieving a all test data in memory")
            test_generator = self._get_generator(df_test, data_type='test', batch_size=len(df_test))
            x_test, _ = test_generator.next()
            # Get predict arguments
            x_arg = x_test
            batch_size_arg = batch_size

        # Predict
        predicted_proba = self.model.predict(  # type: ignore
            x_arg,
            batch_size=batch_size_arg,
            steps=None,
            workers=8,  # TODO : Check if this is ok if there are less CPUs
            verbose=1
        )

        # We return the probabilities if wanted
        if return_proba:
            return predicted_proba

        # Finally, we get the classes predictions
        return self.get_classes_from_proba(predicted_proba)  # type: ignore

    @utils.trained_needed
    def _predict_object_detector(self, df_test: pd.DataFrame, **kwargs) -> list:
        '''Predictions on test set - works only with batch size = 1

        Args:
            df_test (pd.DataFrame): DataFrame to be predicted, with column file_path
        Raises:
            ValueError: If the model is not an object detector
        Returns:
            list: List of list of bboxes (one list per image)
        '''
        raise NotImplementedError("'_predict_object_detector' needs to be overridden")

    @utils.trained_needed
    def predict_proba(self, df_test, batch_size: int = None) -> np.ndarray:
        '''Probabilities predicted on the test set

        Args:
            df_test (pd.DataFrame): DataFrame to be predicted, with column file_path
        Kwargs:
            batch_size (int): Batch size to be used
        Raises:
            ValueError: If the model is not a classifier
        Returns:
            (np.ndarray): Array, shape = [n_samples, n_classes]
        '''
        if self.model_type != 'classifier':
            raise ValueError(f"`predict_proba` function does not support model type {self.model_type}")

        # We reuse the predict function
        return self.predict(df_test, return_proba=True, batch_size=batch_size)

    def _get_generator(self, df: pd.DataFrame, data_type: str, batch_size: int, **kwargs) -> ImageDataGenerator:
        '''Gets image generator from a list of files

        Args:
            df (pd.DataFrame): DataFrame with files to be loaded
            data_type (str): 'train', 'valid' or 'test'
            batch_size (int): Batch size to be used
        Raises:
            NotImplementedError: If the model type is not supported
        '''
        if self.model_type == 'classifier':
            return self._get_generator_classifier(df, data_type, batch_size)
        else:
            raise NotImplementedError(f"`_get_generator` needs to be overridden for model type {self.model_type}")

    def _get_generator_classifier(self, df: pd.DataFrame, data_type: str, batch_size: int, **kwarg) -> ImageDataGenerator:
        '''Gets image generator from a list of files - classifier version

        Args:
            df (pd.DataFrame): DataFrame with files to be loaded
            data_type (str): 'train', 'valid' or 'test'
            batch_size (int): Batch size to be used
        Raises:
            ValueError: If the model is not a classifier
            ValueError: If data_type is not in ['train', 'valid', 'test']
            AttributeError: If list_classes attribute is not defined
        '''
        if self.model_type != 'classifier':
            raise ValueError(f"`_get_generator_classifier` function does not support model type {self.model_type}")
        if data_type not in ['train', 'valid', 'test']:
            raise ValueError(f"{data_type} is not a valid option for argument data_type (['train', 'valid', 'test'])")
        if self.list_classes is None:
            raise AttributeError("Cannot get an image generator if list_classes is not set.")

        # Copy
        df = df.copy(deep=True)
        # Set data_gen (no augmentation if validation/test)
        if data_type == 'train':
            data_generator = ImageDataGenerator(preprocessing_function=self.preprocess_input, **self.data_augmentation_params)
        else:
            data_generator = ImageDataGenerator(preprocessing_function=self.preprocess_input)

        # Get generator
        shuffle = True if data_type == 'train' else False  # DO NOT SHUFFLE IF VALID OR TEST !
        if data_type != 'test':
            generator = data_generator.flow_from_dataframe(df, directory=None, x_col='file_path', y_col='file_class', classes=self.list_classes,
                                                           target_size=(self.width, self.height), color_mode=self.color_mode, class_mode='categorical',
                                                           batch_size=batch_size, shuffle=shuffle)
        # For the test dataset, we create a fake DataFrame with a unique class
        else:
            df['fake_class_col'] = 'all_classes'
            generator = data_generator.flow_from_dataframe(df, directory=None, x_col='file_path', y_col='fake_class_col', classes=['all_classes'],
                                                           target_size=(self.width, self.height), color_mode=self.color_mode, class_mode='categorical',
                                                           batch_size=batch_size, shuffle=False)

        return generator

    def _get_preprocess_input(self) -> Union[Callable, None]:
        '''Gets the preprocessing to be used before feeding images to the NN
        Needs to be overridden by child classes

        Returns:
            (Callable | None): Preprocessing function
        '''
        return None

    def _get_model(self) -> Any:
        '''Gets a model structure

        Returns:
            (Model): a Keras model
        '''
        raise NotImplementedError("'_get_model' needs to be overridden")

    def _get_callbacks(self, *args) -> list:
        '''Gets model callbacks

        Returns:
            list: List of callbacks
        '''
        # Get classic callbacks
        callbacks = [EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)]
        if self.level_save in ['MEDIUM', 'HIGH']:
            callbacks.append(
                ModelCheckpoint(
                    filepath=os.path.join(self.model_dir, 'best.hdf5'), monitor='val_loss', save_best_only=True, mode='auto'
                )
            )
        callbacks.append(CSVLogger(filename=os.path.join(self.model_dir, 'logger.csv'), separator='{{default_sep}}', append=False))
        callbacks.append(TerminateOnNaN())

        # Get LearningRateScheduler
        scheduler = self._get_learning_rate_scheduler()
        if scheduler is not None:
            callbacks.append(LearningRateScheduler(scheduler))

        # Manage tensorboard
        if self.level_save in ['HIGH']:
            # Get log directory
            models_path = utils.get_models_path()
            tensorboard_dir = os.path.join(models_path, 'tensorboard_logs')
            # We add a prefix so that the function load_model works correctly (it looks for a sub-folder with model name)
            log_dir = os.path.join(tensorboard_dir, f"tensorboard_{ntpath.basename(self.model_dir)}")
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # TODO: check if this class does not slow proccesses
            # -> For now: comment
            # Create custom class to monitore LR changes
            # https://stackoverflow.com/questions/49127214/keras-how-to-output-learning-rate-onto-tensorboard
            # class LRTensorBoard(TensorBoard):
            #     def __init__(self, log_dir, **kwargs) -> None:  # add other arguments to __init__ if you need
            #         super().__init__(log_dir=log_dir, **kwargs)
            #
            #     def on_epoch_end(self, epoch, logs=None):
            #         logs.update({'lr': K.eval(self.model.optimizer.lr)})
            #         super().on_epoch_end(epoch, logs)

            callbacks.append(TensorBoard(log_dir=log_dir, write_grads=False, write_images=False))
            self.logger.info(f"To start tensorboard: python -m tensorboard.main --logdir {tensorboard_dir} --samples_per_plugin images=10")
            # We use samples_per_plugin to avoid a rare issue between matplotlib and tensorboard
            # https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread

        return callbacks

    def _get_learning_rate_scheduler(self) -> Union[Callable, None]:
        '''Fonction to define a Learning Rate Scheduler
           -> if it returns None, no scheduler will be used. (def.)
           -> This function will be save directly in the model configuration file
           -> This can be overridden at runing time

        Returns:
            (Callable | None): A learning rate Scheduler
        '''
        # e.g.
        # def scheduler(epoch):
        #     lim_epoch = 75
        #     if epoch < lim_epoch:
        #         return 0.01
        #     else:
        #         return max(0.001, 0.01 * math.exp(0.01 * (lim_epoch - epoch)))
        scheduler = None
        return scheduler

    def _plot_metrics_and_loss(self, fit_history, **kwargs) -> None:
        '''Plots available metrics and losses

        Args:
            fit_history (?) : fit history
        '''
        # Manage dir
        plots_path = os.path.join(self.model_dir, 'plots')
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)

        # Get a dictionnary of possible metrics/loss plots
        metrics_dir = {
            'acc': ['Accuracy', 'accuracy'],
            'loss': ['Loss', 'loss'],
            'categorical_accuracy': ['Categorical accuracy', 'categorical_accuracy'],
            'f1': ['F1-score', 'f1_score'],
            'precision': ['Precision', 'precision'],
            'recall': ['Recall', 'recall'],
        }

        # Plot each available metric
        for metric in fit_history.history.keys():
            if metric in metrics_dir.keys():
                title = metrics_dir[metric][0]
                filename = metrics_dir[metric][1]
                plt.figure(figsize=(10, 8))
                plt.plot(fit_history.history[metric])
                plt.plot(fit_history.history[f'val_{metric}'])
                plt.title(f"Model {title}")
                plt.ylabel(title)
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Validation'], loc='upper left')
                # Save
                filename == f"{filename}.jpeg"
                plt.savefig(os.path.join(plots_path, filename))

                # Close figures
                plt.close('all')

    def _save_model_png(self, model) -> None:
        '''Tries to save the structure of the model in png format
        Graphviz necessary

        Args:
            model (?): model to plot
        '''
        # Check if graphiz is intalled
        # TODO : to be improved !
        graphiz_path = 'C:/Program Files (x86)/Graphviz2.38/bin/'
        if os.path.isdir(graphiz_path):
            os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
            img_path = os.path.join(self.model_dir, 'model.png')
            plot_model(model, to_file=img_path)

    @no_type_check  # We do not check the type, because it is complicated with managing custom_objects_str
    def save(self, json_data: Union[dict, None] = None) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        '''
        # Save configuration JSON
        if json_data is None:
            json_data = {}

        json_data['librairie'] = 'keras'
        json_data['batch_size'] = self.batch_size
        json_data['epochs'] = self.epochs
        json_data['validation_split'] = self.validation_split
        json_data['patience'] = self.patience
        json_data['width'] = self.width
        json_data['height'] = self.height
        json_data['depth'] = self.depth
        json_data['color_mode'] = self.color_mode
        json_data['in_memory'] = self.in_memory
        json_data['data_augmentation_params'] = self.data_augmentation_params
        json_data['nb_train_generator_images_to_save'] = self.nb_train_generator_images_to_save
        json_data['keras_params'] = self.keras_params
        if self.model is not None:
            json_data['keras_model'] = json.loads(self.model.to_json())
        else:
            json_data['keras_model'] = None

        # Add _get_model code if not in json_data
        if '_get_model' not in json_data.keys():
            json_data['_get_model'] = pickle.source.getsourcelines(self._get_model)[0]
        # Add _get_preprocess_input code if not in json_data
        if '_get_preprocess_input' not in json_data.keys():
            json_data['_get_preprocess_input'] = pickle.source.getsourcelines(self._get_preprocess_input)[0]
        # Save preprocess_input to a .pkl file if level_save > LOW
        pkl_path = os.path.join(self.model_dir, "preprocess_input.pkl")
        if self.level_save in ['MEDIUM', 'HIGH']:
            with open(pkl_path, 'wb') as f:
                pickle.dump(self.preprocess_input, f)
        # Add _get_learning_rate_scheduler code if not in json_data
        if '_get_learning_rate_scheduler' not in json_data.keys():
            json_data['_get_learning_rate_scheduler'] = pickle.source.getsourcelines(self._get_learning_rate_scheduler)[0]
        # Add custom_objects code if not in json_data
        if 'custom_objects' not in json_data.keys():
            custom_objects_str = self.custom_objects.copy()
            for key in custom_objects_str.keys():
                if callable(custom_objects_str[key]):
                    # Nominal case
                    if not type(custom_objects_str[key]) == functools.partial:
                        custom_objects_str[key] = pickle.source.getsourcelines(custom_objects_str[key])[0]
                    # Manage partials
                    else:
                        custom_objects_str[key] = {
                            'type': 'partial',
                            'args': custom_objects_str[key].args,
                            'function': pickle.source.getsourcelines(custom_objects_str[key].func)[0],
                        }
            json_data['custom_objects'] = custom_objects_str

        # Save strategy :
        # - best.hdf5 already saved in fit()
        # - can't pickle keras model, so we drop it, save, and reload it
        # TODO: use dill to get rid of  "can't pickle ..." errors
        keras_model = self.model
        self.model = None
        super().save(json_data=json_data)
        self.model = keras_model

    def reload_model(self, hdf5_path: str) -> Any:
        '''Loads a Keras model from a HDF5 file

        Args:
            hdf5_path (str): Path to the hdf5 file
        Returns:
            ?: Keras model
        '''
        # Fix tensorflow GPU if not already done (useful if we reload a model)
        try:
            gpu_devices = tf.config.experimental.list_physical_devices('GPU')
            for device in gpu_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except Exception:
            pass

        # We check if we already have the custom objects
        if hasattr(self, 'custom_objects') and self.custom_objects is not None:
            custom_objects = self.custom_objects
        else:
            self.logger.warning("Can't find the attribute 'custom_objects' in the model to be reloaded")
            self.logger.warning("Backup on the default custom_objects of utils_deep_keras")
            custom_objects = utils_deep_keras.custom_objects

        # Loading of the model
        keras_model = load_model(hdf5_path, custom_objects=custom_objects)

        # Set trained to true if not already true
        if not self.trained:
            self.trained = True
            self.nb_fit = 1

        # Return
        return keras_model

    def reload_from_standalone(self, **kwargs) -> None:
        '''Reloads a model from its configuration and "standalones" files
        - /!\\ Needs to be overridden /!\\ -
        '''
        raise NotImplementedError("'reload' needs to be overridden")

    def _is_gpu_activated(self) -> bool:
        '''Checks if a GPU is used

        Returns:
            bool: whether GPU is available or not
        '''
        # Check for available GPU devices
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            return True
        else:
            return False


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")

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


import os
import time
import json
import ntpath
import shutil
import logging
import functools
import numpy as np
import pandas as pd
import dill as pickle
import seaborn as sns
import matplotlib.pyplot as plt
from typing import no_type_check, Optional, Union, Callable, Any

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint,
                                        TerminateOnNaN, LearningRateScheduler)

from {{package_name}} import utils
from {{package_name}}.models_training import utils_deep_keras
from {{package_name}}.models_training.model_class import ModelClass

sns.set(style="darkgrid")


class ModelKeras(ModelClass):
    '''Generic model for Keras NN'''

    _default_name = 'model_keras'

    # Not implemented :
    # -> _get_model
    # -> reload_from_standalone

    def __init__(self, batch_size: int = 64, epochs: int = 99, validation_split: float = 0.2, patience: int = 5,
                 nb_iter_keras: int = 1, keras_params: Union[dict, None] = None, **kwargs) -> None:
        '''Initialization of the class (see ModelClass for more arguments)

        Kwargs:
            batch_size (int): Batch size
            epochs (int): Number of epochs
            validation_split (float): Percentage for the validation set split
                Only used if no input validation set when fitting
            patience (int): Early stopping patience
            nb_iter_keras (int): Number of iteration done when fitting (default: 1)
                If != 1, one fits several times the same model (but with different initializations),
                which gives a better stability
                Can't fit again if > 1.
            keras_params (dict): Parameters used by keras models.
                e.g. learning_rate, nb_lstm_units, etc...
                The purpose of this dictionary is for the user to use it as they wants in the _get_model function
                This parameter was initially added in order to do an hyperparameters search
        '''
        # TODO: learning rate should be an attribute !
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

        # Model set on fit
        self.model: Any = None

        # Keras number of iteration
        self.nb_iter_keras = nb_iter_keras

        # Keras params
        if keras_params is None:
            keras_params = {}
        self.keras_params = keras_params.copy()

        # Keras custom objects : we get the ones specified in utils_deep_keras
        self.custom_objects = utils_deep_keras.custom_objects

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, with_shuffle: bool = True, **kwargs) -> None:
        '''Fits the model

        Args:
            x_train (?): Array-like, shape = [n_samples, n_features]
            y_train (?): Array-like, shape = [n_samples, n_targets]
            x_valid (?): Array-like, shape = [n_samples, n_features]
            y_valid (?): Array-like, shape = [n_samples, n_targets]
        Kwargs:
            with_shuffle (bool): If x, y must be shuffled before fitting
                This should be used if y is not shuffled as the split_validation takes the lines in order.
                Thus, the validation set might get classes which are not in the train set ...
        Raises:
            RuntimeError: If one tries to fit again a model with nb_iter_keras > 1
            AssertionError: If different classes when comparing an already fitted model and a new dataset
        '''

        ##############################################
        # Manage retrain
        ##############################################

        if self.trained and self.nb_iter_keras > 1:
            self.logger.error("We can't fit again a Keras model if nb_iter_keras > 1")
            raise RuntimeError("We can't fit again a Keras model if nb_iter_keras > 1")
        # If a model has already been fitted, we make a new folder in order not to overwrite the existing one !
        # And we save the old conf
        elif self.trained:
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
                    self.logger.error(f"Unable to copy {src} to {dst}")
                    self.logger.error("We still go on")
                    self.logger.error(repr(e))

        ##############################################
        # Prepare x_train, x_valid, y_train & y_valid
        # Also extract list of classes if classification
        ##############################################

        # Checking input formats
        x_train, y_train = self._check_input_format(x_train, y_train, fit_function=True)
        # If the validation set is present, we check its format (but with fit_function=False)
        if y_valid is not None:
            x_valid, y_valid = self._check_input_format(x_valid, y_valid, fit_function=False)

        # If classification, we need to transform y
        if self.model_type == 'classifier':
            # if not multilabel, transform y_train as dummies (should already be the case for multi-labels)
            if not self.multi_label:
                # If len(array.shape)==2, we flatten the array if the second dimension is useless
                if isinstance(y_train, np.ndarray) and len(y_train.shape) == 2 and y_train.shape[1] == 1:
                    y_train = np.ravel(y_train)
                if isinstance(y_valid, np.ndarray) and len(y_valid.shape) == 2 and y_valid.shape[1] == 1:
                    y_valid = np.ravel(y_valid)
                # Dummies transformation
                y_train = pd.get_dummies(y_train)
                y_valid = pd.get_dummies(y_valid) if y_valid is not None else None
                # Important : get_dummies reorder the columns in alphabetical order
                # Thus, there is no problem if we fit again on a new dataframe with shuffled data
                list_classes = list(y_train.columns)
                # FIX: valid test might miss some classes, hence we need to add them back to y_valid
                if y_valid is not None and y_train.shape[1] != y_valid.shape[1]:
                    for cl in list_classes:
                        # Add missing columns
                        if cl not in y_valid.columns:
                            y_valid[cl] = 0
                    y_valid = y_valid[list_classes]  # Reorder
            # Else keep it as it is
            else:
                y_train = y_train
                y_valid = y_valid
                if hasattr(y_train, 'columns'):
                    list_classes = list(y_train.columns)
                else:
                    self.logger.warning(
                        "Can't read the name of the columns of y_train -> inverse transformation won't be possible"
                    )
                    # We still create a list of classes in order to be compatible with other functions
                    list_classes = [str(_) for _ in range(pd.DataFrame(y_train).shape[1])]

            # Set dict_classes based on list classes
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

        # Shuffle x, y if wanted
        # It is advised as validation_split from keras does not shufle the data
        # Hence, for classificationt task, we might have classes in the validation data that we never met in the training data
        if with_shuffle:
            p = np.random.permutation(len(x_train))
            x_train = np.array(x_train)[p]
            y_train = np.array(y_train)[p]
        # Else still transform to numpy array
        else:
            x_train = np.array(x_train)
            y_train = np.array(y_train)

        # Also get y_valid as numpy & get validation_data (tuple) if available
        validation_data: Optional[tuple] = None  # Def. None if y_valid is None
        if y_valid is not None:
            y_valid = np.array(y_valid)
            validation_data = (x_valid, y_valid)

        if validation_data is None:
            self.logger.warning(f"Warning, no validation set. The training set will be splitted (validation fraction = {self.validation_split})")

        ##############################################
        # Fit
        ##############################################

        # Fit for each iteration wanted
        for iter in range(self.nb_iter_keras):
            if self.nb_iter_keras > 1:
                self.logger.info(f"Training of iteration {iter}")

            # Get model (if already fitted we do not load a new one)
            if not self.trained:
                self.model = self._get_model()

            # Get callbacks (early stopping & checkpoint)
            callbacks = self._get_callbacks(iter)

            # Fit
            # We use a try...except in order to save the model if an error arises
            # after more than a minute into training
            start_time = time.time()
            try:
                fit_history = self.model.fit(  # type: ignore
                    x_train,
                    y_train,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    validation_split=self.validation_split if validation_data is None else None,
                    validation_data=validation_data,
                    callbacks=callbacks,
                    verbose=1,
                )
            except (RuntimeError, SystemError, SystemExit, EnvironmentError, KeyboardInterrupt, tf.errors.ResourceExhaustedError, tf.errors.InternalError,
                    tf.errors.UnavailableError, tf.errors.UnimplementedError, tf.errors.UnknownError, Exception) as e:
                # Steps:
                # 1. Display tensor flow error
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
                self._plot_metrics_and_loss(fit_history, iter)
                # Reload best model
                self.model = load_model(
                    os.path.join(self.model_dir, 'best.hdf5'),
                    custom_objects=self.custom_objects
                )

        # Set trained
        self.trained = True
        self.nb_fit += 1

    @utils.trained_needed
    def predict(self, x_test: pd.DataFrame, return_proba: bool = False, experimental_version: bool = False, **kwargs) -> np.ndarray:
        '''Predictions on test set

        Args:
            x_test (pd.DataFrame): DataFrame with the test data to be predicted
        Kwargs:
            return_proba (bool): If the function should return the probabilities instead of the classes
            experimental_version (bool): If an experimental (but faster) version must be used
        Raises:
            ValueError: If the model is not classifier and return_proba=True
            ValueError: If the model is neither a classifier nor a regressor
        Returns:
            (np.ndarray): Array
                # If not return_proba, shape = [n_samples,] or [n_samples, n_classes]
                # Else, shape = [n_samples, n_classes]
        '''
        # Manage errors
        if return_proba and self.model_type != 'classifier':
            raise ValueError(f"Models of the type {self.model_type} can't handle probabilities")

        # We check input format
        x_test, _ = self._check_input_format(x_test)

        # Process
        if self.nb_iter_keras <= 1:
            with_iter = False
        elif self.nb_iter_keras > 1 and self.level_save == 'LOW':
            self.logger.warning("- *************** -")
            self.logger.warning("Can't make a prediction with all iterations if level_save is LOW (no savings of the model).")
            self.logger.warning("We only make prediction with the saved model.")
            self.logger.warning("- *************** -")
            with_iter = False
        else:
            with_iter = True

        # Getting the available iterations
        if with_iter:
            list_file_hdf5 = [f for f in os.listdir(self.model_dir) if f.endswith('.hdf5')]
            nb_iter_keras = len(list_file_hdf5)
        else:
            nb_iter_keras = 1

        # Predict depends on model type
        if self.model_type == 'classifier':
            return self._predict_classifier(x_test, return_proba=return_proba,
                                            experimental_version=experimental_version,
                                            nb_iter_keras=nb_iter_keras)
        elif self.model_type == 'regressor':
            return self._predict_regressor(x_test, experimental_version=experimental_version,
                                           nb_iter_keras=nb_iter_keras)
        else:
            raise ValueError(f"The model type ({self.model_type}) must be 'classifier' or 'regressor'")

    @utils.trained_needed
    def _predict_classifier(self, x_test: pd.DataFrame, return_proba: bool = False, nb_iter_keras: int = 1,
                            experimental_version: bool = False) -> np.ndarray:
        '''Predictions on test
        Args:
            x_test (pd.DataFrame): DataFrame with the test data to be predicted
        Kwargs:
            return_proba (boolean): If the function should return the probabilities instead of the classes
            nb_iter_keras (int): Number of iterations to do (give the mean of results)
            experimental_version (bool): If an experimental (but faster) version must be used
        Raises:
            ValueError: If the model is not of classifier type
        Returns:
            (np.ndarray): Array
                # If not return_proba, shape = [n_samples,] or [n_samples, n_classes]
                # Else, shape = [n_samples, n_classes]
        '''
        if self.model_type != 'classifier':
            raise ValueError(f"Models of type {self.model_type} do not implement the method predict_classifier")

        # Predict per iteration
        predicted_proba = np.zeros((x_test.shape[0], len(self.list_classes)))
        for iter in range(nb_iter_keras):
            # We get the model corresponding to the current iteration if there are more than one available model.
            # Otherwise, we keep the model already set on the class (useful when save_level is LOW)
            if nb_iter_keras > 1:
                filename = 'best.hdf5' if iter == 0 else f'best_{iter}.hdf5'
                self.logger.info(f"Prediction with {filename}")
                self.model = load_model(
                    os.path.join(self.model_dir, f'{filename}'),
                    custom_objects=self.custom_objects
                )

            # Getting the predictions for each iteration
            if experimental_version:
                preds_iter = self.experimental_predict_proba(x_test)
            else:
                preds_iter = self.model.predict(x_test, batch_size=128, verbose=1)  # type: ignore

            # We sums the probabilities ...
            predicted_proba = predicted_proba + preds_iter

        # ... then we divide by the number of iterations in order to get a probability in [0, 1]
        predicted_proba = predicted_proba / nb_iter_keras
        # We return the probabilities if wanted
        if return_proba:
            return predicted_proba

        # Finally, we get the classes predictions
        return self.get_classes_from_proba(predicted_proba)  # type: ignore

    @utils.trained_needed
    def _predict_regressor(self, x_test, nb_iter_keras: int = 1, experimental_version: bool = False) -> np.ndarray:
        '''Predictions on test
        Args:
            x_test (pd.DataFrame): DataFrame with the test data to be predicted
        Kwargs:
            nb_iter_keras (int): Number of iterations to do (give the mean of results)
            experimental_version (bool): If an experimental (but faster) version must be used
        Raises:
            ValueError: If the model is not of regressor type
        Returns:
            (np.ndarray): Array, shape = [n_samples]
        '''
        if self.model_type != 'regressor':
            raise ValueError(f"Models of type {self.model_type} do not implement the method predict_regressor")

        # Predict per iteration
        predictions = np.zeros((x_test.shape[0], 1))  # TODO: later, manage multi-output
        for iter in range(nb_iter_keras):
            # We get the model corresponding to the current iteration if there are more than one available model.
            # Otherwise, we keep the model already set on the class (useful when save_level is LOW)
            if nb_iter_keras > 1:
                filename = 'best.hdf5' if iter == 0 else f'best_{iter}.hdf5'
                self.logger.info(f"Prediction with {filename}")
                self.model = load_model(
                    os.path.join(self.model_dir, f'{filename}'),
                    custom_objects=self.custom_objects
                )

            # Getting the predictions for each iteration
            if experimental_version:
                preds_iter = self.experimental_predict_proba(x_test)
            else:
                preds_iter = self.model.predict(x_test, batch_size=128, verbose=1)  # type: ignore

            # We sum the prediction ...
            predictions = predictions + preds_iter

        # ... then we divides by the number of iterations in order to get the mean
        predictions = predictions / nb_iter_keras

        # Finally, we get the final format
        # TODO : should certainly be changed for multi-output
        # TODO : create an equivalent of get_classes_from_proba for regression ?
        return np.array([pred[0] for pred in predictions])

    @utils.trained_needed
    def predict_proba(self, x_test: pd.DataFrame, **kwargs) -> np.ndarray:
        '''Predicts the probabilities on the test set

        Args:
            x_test (pd.DataFrame): Array-like, shape = [n_samples, n_features]
        Raises:
            ValueError: If model not classifier
        Returns:
            (np.ndarray): Array, shape = [n_samples, n_classes]
        '''
        if self.model_type != 'classifier':
            raise ValueError(f"Models of type {self.model_type} do not implement the method predict_proba")

        # We check input format
        x_test, _ = self._check_input_format(x_test)

        # We use predict again
        return self.predict(x_test, return_proba=True)

    @utils.trained_needed
    def experimental_predict_proba(self, x_test: pd.DataFrame) -> np.ndarray:
        '''Predictions on test set - simple pass forward - experimental

        Args:
            x_test (pd.DataFrame): Array-like, shape = [n_samples]
        Returns:
            (np.ndarray): Array, shape = [n_samples, n_classes]
        '''
        @tf.function
        def serve(x):
            return self.model(x, training=False)

        return serve(x_test).numpy()

    def _get_model(self) -> Model:
        '''Gets a model structure

        Returns:
            (Model): a Keras model
        '''
        raise NotImplementedError("'_get_model' needs to be overridden")

    def _get_callbacks(self, iter: int = 0) -> list:
        '''Gets model callbacks

        Kwargs:
            iter (int): Number of the current iteration, by default is 0
        Returns:
            list: List of callbacks
        '''
        # Get classic callbacks
        callbacks = [EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)]
        suff = '' if iter == 0 else f'_{iter}'
        if self.level_save in ['MEDIUM', 'HIGH']:
            callbacks.append(
                ModelCheckpoint(
                    filepath=os.path.join(self.model_dir, f'best{suff}.hdf5'), monitor='val_loss', save_best_only=True, mode='auto'
                )
            )
        callbacks.append(CSVLogger(filename=os.path.join(self.model_dir, f'logger{suff}.csv'), separator='{{default_sep}}', append=False))
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
            log_dir = os.path.join(tensorboard_dir, f"tensorboard_{ntpath.basename(self.model_dir)}_{iter}")
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

            # Append tensorboard callback
            # TODO: check compatibility tensorflow 2.3
            # WARNING : https://stackoverflow.com/questions/63619763/model-training-hangs-forever-when-using-a-tensorboard-callback-with-an-lstm-laye
            # A compatibility problem TensorBoard / TensorFlow 2.3 (cuDNN implementation of LSTM/GRU) can arise
            # In this case, the training of the model can be "blocked" and does not respond anymore
            # This problem has arisen two times on Pôle Emploi computers (windows 7 & VM Ubuntu on windows 7 host)
            # No problem on Valeuriad computers (windows 10)
            # Thus, TensorBoard is deactivated by default for now
            # While awaiting a possible fix, you are responsible for checking if TensorBoard works on your computer
            self.logger.warning(" ###################### ")
            self.logger.warning("TensorBoard deactivated : compatibility problem TensorBoard / TensorFlow 2.3 (cuDNN implementation of LSTM/GRU) can arise")
            self.logger.warning("https://stackoverflow.com/questions/63619763/model-training-hangs-forever-when-using-a-tensorboard-callback-with-an-lstm-laye")
            self.logger.warning(" In order to activate if, one has to modify the method _get_callbacks of model_keras.py")
            self.logger.warning(" ###################### ")
            # callbacks.append(TensorBoard(log_dir=log_dir, write_grads=False, write_images=False))
            # self.logger.info(f"To start tensorboard: python -m tensorboard.main --logdir {tensorboard_dir}")

        return callbacks

    def _get_learning_rate_scheduler(self) -> Union[Callable, None]:
        '''Defines a Learning Rate Scheduler
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

    def _plot_metrics_and_loss(self, fit_history, iter: int) -> None:
        '''Plots some metrics & loss

        Arguments:
            fit_history (?) : fit history
            iter (int): current iteration
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
            'mean_absolute_error': ['MAE', 'mae'],
            'mae': ['MAE', 'mae'],
            'mean_squared_error': ['MSE', 'mse'],
            'mse': ['MSE', 'mse'],
            'root_mean_squared_error': ['RMSE', 'rmse'],
            'rmse': ['RMSE', 'rmse'],
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
                filename == f"{filename}.jpeg" if iter == 0 else f"{filename}_{iter}.jpeg"
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
        json_data['nb_iter_keras'] = self.nb_iter_keras
        json_data['keras_params'] = self.keras_params
        if self.model is not None:
            json_data['keras_model'] = json.loads(self.model.to_json())
        else:
            json_data['keras_model'] = None

        # Add _get_model code if not in json_data
        if '_get_model' not in json_data.keys():
            json_data['_get_model'] = pickle.source.getsourcelines(self._get_model)[0]
        # Add _get_learning_rate_scheduler code if not in json_data
        if '_get_learning_rate_scheduler' not in json_data.keys():
            json_data['_get_learning_rate_scheduler'] = pickle.source.getsourcelines(self._get_learning_rate_scheduler)[0]
        # Add custom_objects code if not in json_data
        if 'custom_objects' not in json_data.keys():
            custom_objects_str = self.custom_objects.copy()
            for key in custom_objects_str.keys():
                if callable(custom_objects_str[key]):
                    # Nominal case
                    if not isinstance(custom_objects_str[key], functools.partial):
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
        # TODO: Use dill in order to not have "can't pickle ..." problems anymore
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
        raise NotImplementedError("'reload_from_standalone' needs to be overridden")

    def _is_gpu_activated(self) -> bool:
        '''Checks if a GPU is used

        Returns:
            bool: whether GPU is available or not
        '''
        # Checks for available GPU devices
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            return True
        else:
            return False


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")

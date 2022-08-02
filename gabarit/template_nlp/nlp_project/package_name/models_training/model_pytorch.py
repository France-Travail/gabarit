#!/usr/bin/env python3

## Generic model for Pytorch NN
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
# - ModelPyTorch -> Generic model for Pytorch NN


import os
import dill
import shutil
import logging
import numpy as np
import pandas as pd
from typing import Union, Any, List

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ProgressBar

from {{package_name}} import utils
from {{package_name}}.models_training.model_class import ModelClass

import seaborn as sns
sns.set(style="darkgrid")


class ModelPyTorch(ModelClass):
    '''Generic model for Pytorch NN'''

    _default_name = 'model_pytorch'

    # Not implemented :
    # -> predict_proba
    # -> _get_train_dataloader
    # -> _prepare_x_test
    # -> _get_model
    # -> reload

    def __init__(self, batch_size: int = 64, epochs: int = 99, validation_split: float = 0.2, patience: int = 5,
                 embedding_name: str = 'cc.fr.300.pkl', pytorch_params: Union[dict, None] = None, **kwargs) -> None:
        '''Initialization of the class (see ModelClass for more arguments)

        Kwargs:
            batch_size (int): Batch size
            epochs (int): Number of epochs
            validation_split (float): Percentage for the validation set split
                Only used if no input validation set when fitting
            patience (int): Early stopping patience
            embedding_name (str) : The name of the embedding matrix to use
            pytorch_params (dict): Parameters used by pytorch models
                e.g. learning_rate, nb_lstm_units, etc...
                The purpose of this dictionary is for the user to use it as they wants in the _get_model function
                This parameter was initially added in order to do an hyperparameters search
        '''
        # TODO: learning rate should be an attribute !
        # Init.
        super().__init__(**kwargs)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Param. model
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.patience = patience

        # Model set on fit
        self.model: Any = None

        # Param embedding (can be None if no embedding)
        self.embedding_name = embedding_name

        # Pytorch params
        if pytorch_params is None:
            pytorch_params = {}
        self.pytorch_params = pytorch_params.copy()

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, with_shuffle: bool = True, **kwargs) -> None:
        '''Fits the model

        Args:
            x_train (?): Array-like, shape = [n_samples, n_features]
            y_train (?): Array-like, shape = [n_samples, n_targets]
            x_valid (?): Array-like, shape = [n_samples, n_features]
            y_valid (?): Array-like, shape = [n_samples, n_targets]
        Kwargs:
            with_shuffle (bool): If x, y must be shuffled before fitting
                Experimental: We must verify if it works as intended depending on the formats of x and y
                This should be used if y is not shuffled as the split_validation takes the lines in order.
                Thus, the validation set might get classes which are not in the train set ...
        Raises:
            AssertionError: If different classes when comparing an already fitted model and a new dataset
        '''
        ##############################################
        # Manage retrain
        ##############################################

        # If a model has already been fitted, we make a new folder in order not to overwrite the existing one !
        # And we save the old conf
        # TODO: Do everything again !!!
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
        # Prepare x_train, x_valid, y_train & y_valid
        # Also extract list of classes
        ##############################################

        # If not multi-labels, transform y_train as dummies (should already be the case for multi-labels)
        if not self.multi_label:
            # If len(array.shape)==2, we flatten the array if the second dimension is useless
            if isinstance(y_train, np.ndarray) and len(y_train.shape) == 2 and y_train.shape[1] == 1:
                y_train = np.ravel(y_train)
            if isinstance(y_valid, np.ndarray) and len(y_valid.shape) == 2 and y_valid.shape[1] == 1:
                y_valid = np.ravel(y_valid)
            # Transformation dummies
            y_train_dummies = pd.get_dummies(y_train)
            y_valid_dummies = pd.get_dummies(y_valid) if y_valid is not None else None
            # Important : get_dummies reorder the columns in alphabetical order
            # Thus, there is no problem if we fit again on a new dataframe with shuffled data
            list_classes = list(y_train_dummies.columns)
            # FIX: valid test might miss some classes, hence we need to add them back to y_valid_dummies
            if y_valid_dummies is not None and y_train_dummies.shape[1] != y_valid_dummies.shape[1]:
                for cl in list_classes:
                    # Add missing columns
                    if cl not in y_valid_dummies.columns:
                        y_valid_dummies[cl] = 0
                y_valid_dummies = y_valid_dummies[list_classes]  # Reorder
        # Else keep it as it
        else:
            y_train_dummies = y_train
            y_valid_dummies = y_valid
            if hasattr(y_train_dummies, 'columns'):
                list_classes = list(y_train_dummies.columns)
            else:
                self.logger.warning(
                    "Can't read the name of the columns of y_train -> inverse transformation won't be possible"
                )
                # We still create a list of classes in order to be compatible with other functions
                list_classes = [str(_) for _ in range(pd.DataFrame(y_train_dummies).shape[1])]

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

        # Shuffle x, y if wanted (advised)
        # Hence we might have classes in the validation data that we never met in the training data
        if with_shuffle:
            p = np.random.permutation(len(x_train))
            x_train = np.array(x_train)[p]
            y_train_dummies = np.array(y_train_dummies)[p]
        # Else still transform to numpy array
        else:
            x_train = np.array(x_train)
            y_train_dummies = np.array(y_train_dummies)

        # Also get y_valid_dummies as numpy
        y_valid_dummies = np.array(y_valid_dummies)

        ##############################################
        # Apply validation_split if no validation given
        ##############################################

        if y_valid is None:
            self.logger.info(f"No validation set, we split the train set : ({round((1 - self.validation_split) * 100, 2)} % train, {round((self.validation_split) * 100, 2)} % validation)")
            p = np.random.permutation(int(len(x_train) * (1 - self.validation_split)))
            mask = np.ones(len(x_train), np.bool)
            mask[p] = 0
            x_valid = x_train[mask]
            y_valid_dummies = y_train_dummies[mask]
            x_train = x_train[~mask]
            y_train_dummies = y_train_dummies[~mask]

        ##############################################
        # Get data loaders
        ##############################################

        # Get train dataloader
        self.logger.info("Preparing training set")
        train_dl = self._get_train_dataloader(self.batch_size, x_train, y_train_dummies)

        # Also get valid data loader
        self.logger.info("Preparing validation set")
        valid_dl = self._get_test_dataloader(self.batch_size, x_valid, y_valid_dummies)  # We could change the batch size value

        ##############################################
        # Fit
        ##############################################

        self.logger.info("Training ;..")

        # Get model (if already fitted we do not load a new one)
        if not self.trained:
            self.model = self._get_model(len(train_dl))

        # Get callbacks (early stopping & checkpoint)
        callbacks = self._get_callbacks()

        # Get logger
        # logger = self._get_logger()
        # TODO: Understand why the training won't work when the logger is added
        logger = False

        # Fit
        # We use a try...except in order to save the model if an error arises
        # after more than a minute into training
        try:
            # We unfreeze the weights
            self.model.unfreeze()  # type: ignore
            # We go to train mode (it should be already the case, but one never knows !)
            self.model.train()  # type: ignore
            # We train
            trainer = pl.Trainer(
                default_root_dir=self.model_dir,
                max_epochs=self.epochs,
                gradient_clip_val=self.model.gradient_clip_val if hasattr(self.model, 'gradient_clip_val') else None,  # type: ignore
                weights_summary="top",  # "full" would print all layers summary
                gpus=1 if torch.cuda.is_available() else 0,
                callbacks=callbacks,
                logger=logger,
            )
            trainer.fit(self.model, train_dl, valid_dl)  # type: ignore
            # If checkpoint best_model, we reload
            if os.path.exists(os.path.join(self.model_dir, 'best_model.ckpt')):
                self.logger.info("Loading of the best model")
                self.model = self.reload_model(os.path.join(self.model_dir, 'best_model.ckpt'))
            else:
                self.logger.warning("Can't load the best model !")
            # And we freeze the weights again
            self.model.freeze()  # type: ignore
        except (RuntimeError, SystemError, SystemExit, EnvironmentError, KeyboardInterrupt, Exception) as e:
            # Steps:
            # 1. Display error
            # 2. Check if more than one minute elapsed & not several iterations & existence best.hdf5
            # 3. Reload best model
            # 4. We consider that a fit occured (trained = True, nb_fit += 1)
            # 5. Save & create a warning file
            # 6. Display error messages
            # 7. Raise an error

            # 1.
            self.logger.error(repr(e))
            raise RuntimeError("Error during model training")
            # TODO : do as in keras -> save best model and reload

            # # Print accuracy & loss if level_save > 'LOW'
            # TODO : do as in keras -> plot metrics
            # if self.level_save in ['MEDIUM', 'HIGH']:
            #     self._plot_metrics_and_loss(fit_history, iter)
            #     # Reload best model
            #     self.model = load_model(
            #         os.path.join(self.model_dir, f'best.hdf5'),
            #         custom_objects=self.custom_objects
            #     )

        # Set trained
        self.trained = True
        self.nb_fit += 1

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def predict(self, x_test, return_proba: bool = False, **kwargs) -> np.ndarray:
        '''Predictions on test set

        Args:
            x_test (?): Array-like or sparse matrix, shape = [n_samples]
        Kwargs:
            return_proba (bool): If the function should return the probabilities instead of the classes
        Returns:
            (np.ndarray): Array, shape = [n_samples, n_classes]
        '''
        # Cast en pd.Series
        x_test = pd.Series(x_test)

        # Get predictions per chunk
        predicted_proba = self.predict_proba(x_test)

        # We return the probabilities if wanted
        if return_proba:
            return predicted_proba

        # Finally, we get the classes predictions
        return self.get_classes_from_proba(predicted_proba)

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def predict_proba(self, x_test, **kwargs) -> np.ndarray:
        '''Probabilities predicted on the test set

        Args:
            x_test (?): Array-like or sparse matrix, shape = [n_samples]
        Returns:
            (np.ndarray): Array, shape = [n_samples, n_classes]
        '''
        raise NotImplementedError("'predict_proba' needs to be overridden")

    def _get_train_dataloader(self, batch_size: int, x_train, y_train_dummies=None) -> DataLoader:
        '''Prepares the input data for the model

        Args:
            batch_size (int): Train batch size
            x_train (?): Array-like, shape = [n_samples, n_features]
            y_train_dummies (?): Array-like, shape = [n_samples, n_features]
        Returns:
            (?): Data loader
        '''
        raise NotImplementedError("'_get_train_dataloader' needs to be overridden")

    def _get_test_dataloader(self, batch_size: int, x_test, y_test_dummies=None) -> DataLoader:
        '''Prepares the input data for the model

        Args:
            x_test (?): Array-like, shape = [n_samples, n_features]
            batch_size (int): Test batch size
        Kwargs:
            y_test_dummies (?): Array-like, shape = [n_samples, n_features]
        Returns:
            (?): Data loader
        '''
        raise NotImplementedError("'_get_test_dataloader' needs to be overridden")

    # TODO: redo with tokenizer pytorch
    # def _get_embedding_matrix(self, tokenizer):
    #     '''Get embedding matrix
    #
    #     Args:
    #         tokenizer (?): Tokenizer to use (useful to test with a new matrice embedding)
    #     Returns:
    #         np.ndarray: embedding matrix
    #         int: embedding size
    #     '''
    #     # Get embedding indexes
    #     embedding_indexes = utils_models.get_embedding(self.embedding_name)
    #     # Get embedding_size
    #     embedding_size = len(embedding_indexes[list(embedding_indexes.keys())[0]])
    #
    #     # Get embedding matrix
    #     # The first line of this matrix is a zero vector
    #     # The following lines are the projections of the words obtained by the tokenizer (same index)
    #
    #     # We keep only the max tokens 'num_words'
    #     # https://github.com/keras-team/keras/issues/8092
    #     if tokenizer.num_words is None:
    #         word_index = {e: i for e, i in tokenizer.word_index.items()}
    #     else:
    #         word_index = {e: i for e, i in tokenizer.word_index.items() if i <= tokenizer.num_words}
    #     # Create embedding matrix
    #     embedding_matrix = np.zeros((len(word_index) + 1, embedding_size))
    #     # Fill it
    #     for word, i in word_index.items():
    #         embedding_vector = embedding_indexes.get(word)
    #         if embedding_vector is not None:
    #             # words not found in embedding index will be all-zeros.
    #             embedding_matrix[i] = embedding_vector
    #     self.logger.info(f"Size of the embedding matrix (ie. number of matches on the input) : {len(embedding_matrix)}")
    #     return embedding_matrix, embedding_size

    def _get_model(self, train_dataloader_size: int = None) -> Any:
        '''Gets a model structure

        Kwargs:
            train_dataloader_size (int): number of batch per epochs. Useful to set a learning rate scheduler
        Returns:
            (?): a PyTorch model
        '''
        raise NotImplementedError("'_get_model' needs to be overridden")

    def _get_callbacks(self, **kwargs) -> list:
        '''Gets model callbacks

        Returns:
            list: List of callbacks
        '''
        callbacks: List[Any] = []
        # Early stopping
        if self.patience > 1:
            self.logger.info(f"Add Early Stopping (patience = {self.patience})")
            es = EarlyStopping(monitor='val_loss', patience=self.patience)
            callbacks.append(es)
        # Add checpoint callback
        if self.level_save in ['MEDIUM', 'HIGH']:
            checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                                  save_top_k=1,
                                                  dirpath=self.model_dir,
                                                  save_weights_only=True,
                                                  filename='best_model')
            callbacks.append(checkpoint_callback)
        # Add a progress bar that does not overwrite previous epoch
        pb = MyProgressBar()
        callbacks.append(pb)
        return callbacks

    def _get_logger(self) -> Any:
        '''Gets model logger

        Returns:
            ?: Logger
        '''
        # TODO: The logger crashes the model, thus we do not use it ...

        # # Def. tensorboard
        # models_path = utils.get_models_path()
        # tensorboard_dir = os.path.join(models_path, 'tensorboard_logs')
        # # We add a prefix so that the function load_model works correctly (it looks for a sub-folder with model name)
        # log_dir = os.path.join(tensorboard_dir, f"tensorboard_{ntpath.basename(self.model_dir)}")
        # if not os.path.exists(log_dir):
        #     os.makedirs(log_dir)
        # logger = pl_loggers.TensorBoardLogger(log_dir)

        # CSV logger
        csv_dir = os.path.join(self.model_dir, 'csv_logs')
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        logger = pl_loggers.CSVLogger(csv_dir)
        return logger

    # TODO plot metrics
    # def _plot_metrics_and_loss(self, fit_history) -> None:
    #     '''Function to plot accuracy & loss
    #
    #     Arguments:
    #         fit_history (?) : fit history
    #     '''
    #     # Manage dir
    #     plots_path = os.path.join(self.model_dir, 'plots')
    #     if not os.path.exists(plots_path):
    #         os.makedirs(plots_path)
    #
    #     # Get a dictionnary of possible metrics/loss plots
    #     metrics_dir = {
    #         'acc': ['Accuracy', 'accuracy'],
    #         'loss': ['Loss', 'loss'],
    #         'categorical_accuracy': ['Categorical accuracy', 'categorical_accuracy'],
    #         'f1': ['F1-score', 'f1_score'],
    #         'precision': ['Precision', 'precision'],
    #         'recall': ['Recall', 'recall'],
    #     }
    #
    #     # Plot each available metric
    #     for metric in fit_history.history.keys():
    #         if metric in metrics_dir.keys():
    #             title = metrics_dir[metric][0]
    #             filename = metrics_dir[metric][1]
    #             plt.figure(figsize=(10, 8))
    #             plt.plot(fit_history.history[metric])
    #             plt.plot(fit_history.history[f'val_{metric}'])
    #             plt.title(f"Model {title}")
    #             plt.ylabel(title)
    #             plt.xlabel('Epoch')
    #             plt.legend(['Train', 'Validation'], loc='upper left')
    #             # Save
    #             filename == f"{filename}.jpeg" if iter == 0 else f"{filename}_{iter}.jpeg"
    #             plt.savefig(os.path.join(plots_path, filename))
    #
    #             # Close figures
    #             plt.close('all')

    def save(self, json_data: Union[dict, None] = None) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        '''
        # Save configuration JSON
        if json_data is None:
            json_data = {}

        json_data['librairie'] = 'pytorch'
        json_data['batch_size'] = self.batch_size
        json_data['epochs'] = self.epochs
        json_data['validation_split'] = self.validation_split
        json_data['patience'] = self.patience
        json_data['embedding_name'] = self.embedding_name
        json_data['pytorch_params'] = self.pytorch_params
        if self.model is not None:
            json_data['pytorch_model'] = str(self.model)
        else:
            json_data['pytorch_model'] = None

        # Add _get_model code if not in json_data
        if '_get_model' not in json_data.keys():
            json_data['_get_model'] = dill.source.getsourcelines(self._get_model)[0]

        # Save strategy :
        # - best_model already saved in fit()
        # - can't pickle pytorch model, so we drop it, save, and reload it
        pytorch_model = self.model
        self.model = None
        super().save(json_data=json_data)
        self.model = pytorch_model

    def reload_model(self, model_path: str, **kwargs) -> Any:
        '''Reloads a model saved with ModelCheckpoint

        Args:
            model_path (str): model path (either checkpoint path or full torch dir, depends on subclass)
        Kwargs:
            kwargs: Dict of kwargs to override predefined params (TO BE CHECKED !!!)
        '''
        raise NotImplementedError("'reload_model' needs to be overridden")

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
        if torch.cuda.is_available():
            return True
        else:
            return False


class MyProgressBar(ProgressBar):
    '''Progress bar that does not override epochs'''
    def on_epoch_start(self, trainer, pl_module) -> None:
        print('\n\n')
        super().on_epoch_start(trainer, pl_module)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")

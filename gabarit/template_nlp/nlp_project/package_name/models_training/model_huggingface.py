#!/usr/bin/env python3

## Generic model for HuggingFace Transformers
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
# - ModelHuggingFace -> Generic model for HuggingFace Transformers


import os
import shutil
import logging
import numpy as np
import pandas as pd
import dill as pickle
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, no_type_check, Union, Tuple, Callable, Any

import torch
from datasets import Dataset
from datasets.arrow_dataset import Batch
from transformers.tokenization_utils_base import BatchEncoding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer, TextClassificationPipeline, PreTrainedTokenizer

from {{package_name}} import utils
from {{package_name}}.models_training.model_class import ModelClass

sns.set(style="darkgrid")

HF_CACHE_DIR = utils.get_transformers_path()


class ModelHuggingFace(ModelClass):
    '''Generic model for Huggingface NN'''

    _default_name = 'model_huggingface'

    # Not implemented :
    # -> reload_from_standalone

    # TODO: perhaps it would be smarter to have this class behaving as the abstract class for all the model types
    # implemented on the HF hub and to create model specific subclasses.
    # => might change it as use cases grow

    def __init__(self, batch_size: int = 8, epochs: int = 99, validation_split: float = 0.2,
                 transformer_name: str = 'Geotrend/distilbert-base-fr-cased', transformer_params: Union[dict, None] = None,
                 trainer_params: Union[dict, None] = None, **kwargs) -> None:
        '''Initialization of the class (see ModelClass for more arguments)

        Kwargs:
            batch_size (int): Batch size
            epochs (int): Number of epochs
            validation_split (float): Percentage for the validation set split
                Only used if no input validation set when fitting
            transformer_name (str) : The name of the transformer backbone to use
            transformer_params (dict): Parameters used by the Transformer model.
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

        # Param embedding (can be None if no embedding)
        self.transformer_name = transformer_name
        self.transformer_params = transformer_params

        # Trainer params
        if trainer_params is None:
            trainer_params = {
                'output_dir': self.model_dir,
                'learning_rate': 2e-5,
                'per_device_train_batch_size': self.batch_size,
                'per_device_eval_batch_size': self.batch_size,
                'num_train_epochs': self.epochs,
                'weight_decay': 0.01,
                'evaluation_strategy': 'epoch',
                'save_strategy': 'epoch',
                'save_total_limit': 1,
                'load_best_model_at_end': True
            }
        self.trainer_params = trainer_params

        # Model set on fit or on reload
        self.model: Any = None
        self.pipe: Any = None
        self.tokenizer: Any = None

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

        # If not multilabel, transform y_train as dummies (should already be the case for multi-labels)
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
        # Else keep it as it is
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

        # Shuffle x, y if wanted
        # It is advised as validation_split from keras does not shufle the data
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

        # If no valid set, split train set according to validation_split
        if y_valid is None:
            self.logger.warning(f"Warning, no validation set. The training set will be splitted (validation fraction = {self.validation_split})")
            p = np.random.permutation(len(x_train))
            cutoff = int(len(p) * self.validation_split)
            x_valid = x_train[p[0:cutoff]]
            x_train = x_train[p[cutoff:]]
            y_valid_dummies = y_train_dummies[p[0:cutoff]]
            y_train_dummies = y_train_dummies[p[cutoff:]]

        ##############################################
        # Get model & prepare datasets
        ##############################################

        # Get model (if already fitted we do not load a new one)
        if not self.trained or self.model is None:
            self.model = self._get_model(num_labels=y_train_dummies.shape[1])
            self.tokenizer = self._get_tokenizer()

        train_dataset = self._prepare_x_train(x_train, y_train_dummies)
        valid_dataset = self._prepare_x_valid(x_valid, y_valid_dummies)

        ##############################################
        # Fit
        ##############################################

        # Fit
        try:
            trainer = Trainer(
                model=self.model,
                args=TrainingArguments(**self.trainer_params),
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer)
            )
            fit_history = trainer.train()
            # Save model & tokenizer
            trainer.model.save_pretrained(save_directory=self.model_dir)
            self.tokenizer.save_pretrained(save_directory=self.model_dir)
        except (RuntimeError, SystemError, SystemExit, EnvironmentError, KeyboardInterrupt, Exception) as e:
            self.logger.error(repr(e))
            raise RuntimeError("Error during model training")

        # Print accuracy & loss if level_save > 'LOW'
        if self.level_save in ['MEDIUM', 'HIGH']:
            # Plot accuracy
            self._plot_metrics_and_loss(fit_history)
            #TODO
            # Reload best model
            #self.model = load_model(os.path.join(self.model_dir, 'best.hdf5'), custom_objects=self.custom_objects)

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
        # Predict probas
        predicted_proba = self.predict_proba(x_test)

        # We return the probabilities if wanted
        if return_proba:
            return predicted_proba

        # Finally, we get the classes predictions
        return self.get_classes_from_proba(predicted_proba)

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def predict_proba(self, x_test, **kwargs) -> np.ndarray:
        '''Predicts probabilities on the test dataset

        Args:
            x_test (?): Array-like or sparse matrix, shape = [n_samples, n_features]
        Returns:
            (np.ndarray): Array, shape = [n_samples, n_classes]
        '''
        # Does not work with np array
        if type(x_test) is np.ndarray:
            x_test = x_test.tolist()
        # Predict
        if self.model.training:
            self.model.eval()
        if self.pipe is None:
            self.pipe = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, return_all_scores=True)
        # As we are using the pipeline, we do not need to prepare x_test (done inside the pipeline)
        results = np.array(self.pipe(x_test))
        predicted_proba = np.array([[x['score'] for x in x] for x in results])
        return predicted_proba

    def _prepare_x_train(self, x_train, y_train_dummies) -> Dataset:
        '''Prepares the input data for the model - train

        Args:
            x_train (?): Array-like, shape = [n_samples, n_features]
        Returns:
            (datasets.Dataset): Prepared dataset
        '''
        # Should not happen if using fit
        if self.tokenizer is None:
            self.tokenizer = self._get_tokenizer()
        return Dataset.from_dict({'text': x_train.tolist(), 'label': y_train_dummies.astype(np.float32).tolist()}).map(self._tokenize_function, batched=True)

    def _prepare_x_valid(self, x_valid, y_valid_dummies) -> Dataset:
        '''Prepares the input data for the model - valid

        Args:
            x_valid (?): Array-like, shape = [n_samples, n_features]
        Returns:
            (datasets.Dataset): Prepared dataset
        '''
        # Same as train (we don't fit any tokenizer)
        return self._prepare_x_train(x_valid, y_valid_dummies)

    def _prepare_x_test(self, x_test) -> Dataset:
        '''Prepares the input data for the model - test

        Args:
            x_test (?): Array-like, shape = [n_samples, n_features]
        Returns:
            (datasets.Dataset): Prepared dataset
        '''
        return Dataset.from_dict({'text': x_test.tolist()}).map(self._tokenize_function, batched=True)

    def _tokenize_function(self, examples: Batch) -> BatchEncoding:
        '''Tokenizes input data

        Args:
            examples (Batch): input data (Dataset Batch)
        Returns:
            BatchEncoding: tokenized data
        '''
        return self.tokenizer(examples["text"], truncation=True)

    def _get_model(self, model_path: str = None, num_labels: int = None) -> Any:
        '''Gets a model structure

        Returns:
            (Any): a HF model
        '''
        model = AutoModelForSequenceClassification.from_pretrained(
                self.transformer_name if model_path is None else model_path,
                num_labels=len(self.list_classes) if num_labels is None else num_labels,
                problem_type="multi_label_classification",
                cache_dir=HF_CACHE_DIR)
        return model

    def _get_tokenizer(self, model_path: str = None) -> PreTrainedTokenizer:
        '''Gets a tokenizer

        Returns:
            (PreTrainedTokenizer): a HF tokenizer
        '''
        tokenizer = AutoTokenizer.from_pretrained(self.transformer_name if model_path is None else model_path, cache_dir=HF_CACHE_DIR)
        return tokenizer

    def _plot_metrics_and_loss(self, fit_history) -> None:
        '''Plots TrainOutput, for legacy and compatibility purpose

        Arguments:
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
            'train_loss': ['Train Loss', 'train_loss']
        }

        # Plot each available metric
        for metric in fit_history.metrics.keys():
            if metric in metrics_dir.keys():
                title = metrics_dir[metric][0]
                filename = metrics_dir[metric][1]
                plt.figure(figsize=(10, 8))
                plt.plot(fit_history.metrics[metric])
                #plt.plot(fit_history.metrics[f'val_{metric}'])
                plt.title(f"Model {title}")
                plt.ylabel(title)
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Validation'], loc='upper left')
                # Save
                filename == f"{filename}.jpeg"
                plt.savefig(os.path.join(plots_path, filename))

                # Close figures
                plt.close('all')

    @no_type_check  # We do not check the type, because it is complicated with managing custom_objects_str
    def save(self, json_data: Union[dict, None] = None) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        '''
        # Save configuration JSON
        if json_data is None:
            json_data = {}

        json_data['librairie'] = 'huggingface'
        json_data['batch_size'] = self.batch_size
        json_data['epochs'] = self.epochs
        json_data['validation_split'] = self.validation_split
        json_data['transformer_name'] = self.transformer_name
        json_data['transformer_params'] = self.transformer_params
        json_data['trainer_params'] = self.trainer_params

        if '_get_model' not in json_data.keys():
            json_data['_get_model'] = pickle.source.getsourcelines(self._get_model)[0]
        # Save strategy :
        # - best.hdf5 already saved in fit()
        # - can't pickle keras model, so we drop it, save, and reload it
        keras_model = self.model
        self.model = None
        super().save(json_data=json_data)
        self.model = keras_model

    def reload_model(self, model_path: str) -> Any:
        '''Loads a HF model from a directory

        Args:
            model_path (str): Path to the directory containing both the model.bin and its conf
        Returns:
            ?: HF model
        '''

        # Loading of the model
        self.model = self._get_model(model_path)
        self.tokenizer = self._get_tokenizer(model_path)
        # Set trained to true if not already true
        if not self.trained:
            self.trained = True
            self.nb_fit = 1

        # Return
        return self.model

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
        # Check for available GPU devices
        return torch.cuda.is_available()


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")

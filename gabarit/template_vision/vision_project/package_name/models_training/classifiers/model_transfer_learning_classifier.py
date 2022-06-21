#!/usr/bin/env python3

## CNN model with transfer learning - Classification
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
# - ModelTransferLearningClassifier -> CNN model with transfer learning for classification


import os
import json
import ntpath
import shutil
import logging
import pandas as pd
import dill as pickle
from functools import partial
from keras.utils import data_utils
from typing import Union, List, Callable

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16, EfficientNetB6
from tensorflow.keras.models import load_model as load_model_keras
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as enet_preprocess_input
from tensorflow.keras.layers import Dense, Input, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import (CSVLogger, EarlyStopping,
                                        ModelCheckpoint, TensorBoard,
                                        TerminateOnNaN, LearningRateScheduler)

from {{package_name}} import utils
from {{package_name}}.models_training.model_keras import ModelKeras
from {{package_name}}.models_training.classifiers.model_classifier import ModelClassifierMixin  # type: ignore


class ModelTransferLearningClassifier(ModelClassifierMixin, ModelKeras):
    '''CNN model with transfer learning for classification'''

    _default_name = 'model_transfer_learning_classifier'

    def __init__(self, with_fine_tune: bool = True, second_epochs: int = 10, second_lr: float = 1e-5, second_patience: int = 5, **kwargs) -> None:
        '''Initialization of the class (see ModelClass, ModelKeras & ModelClassifierMixin for more arguments)

        Kwargs:
            with_fine_tune (bool): If a fine-tuning step should be performed after first training
            second_epochs (int): Number of epochs for the fine-tuning step
            second_lr (float): Learning rate for the fine-tuning step
            second_patience (int): Patience for the fine-tuning step
        '''
        # Init.
        super().__init__(**kwargs)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Params
        self.with_fine_tune = with_fine_tune
        self.second_epochs = second_epochs
        self.second_lr = second_lr
        self.second_patience = second_patience

    def _fit_classifier(self, df_train: pd.DataFrame, df_valid: pd.DataFrame = None, with_shuffle: bool = True, **kwargs) -> dict:
        '''Fits the model - overrides parent function

        Args:
            df_train (pd.DataFrame): Train dataset
                Must contain file_path & file_class columns
        Kwargs:
            df_valid (pd.DataFrame): Validation dataset
                Must contain file_path & file_class columns
            with_shuffle (boolean): If the train dataset must be shuffled
                This should be used if the input dataset is not shuffled & no validation set as the split_validation takes the lines in order.
                Thus, the validation set might get classes which are not in the train set ...
        Returns:
            dict: fit arguments
        '''
        # First fit
        self.logger.info("Transfer Learning - Premier entrainement")
        fit_arguments = super()._fit_classifier(df_train, df_valid=df_valid, with_shuffle=with_shuffle, **kwargs)

        # Fine tune if wanted
        if self.with_fine_tune and self.second_epochs > 0:
            # Unfreeze all layers
            for layer in self.model.layers:  # type: ignore
                layer.trainable = True

            # /!\ Recompile, otherwise unfreeze is not taken into account ! /!\
            # Cf. https://keras.io/guides/transfer_learning/#fine-tuning
            self._compile_model(self.model, lr=self.second_lr)  # Use new LR !
            if self.logger.getEffectiveLevel() < logging.ERROR:
                self.model.summary()  # type: ignore

            # Get new callbacks
            new_callbacks = self._get_second_callbacks()

            # Second fit
            self.logger.info("Transfer Learning - Fine-tuning")
            fit_history = self.model.fit(  # type: ignore
                epochs=self.second_epochs,
                callbacks=new_callbacks,
                verbose=1,
                workers=8,  # TODO : Check if this is ok if there are less CPUs
                **fit_arguments,
            )

            # Print accuracy & loss if level_save > 'LOW'
            if self.level_save in ['MEDIUM', 'HIGH']:
                # Rename first fit plots dir
                original_plots_path = os.path.join(self.model_dir, 'plots')
                renamed_plots_path = os.path.join(self.model_dir, 'plots_initial_fit')
                shutil.move(original_plots_path, renamed_plots_path)
                # Plot new fit graphs
                self._plot_metrics_and_loss(fit_history)
                # Reload best model
                self.model = load_model_keras(
                    os.path.join(self.model_dir, 'best.hdf5'),
                    custom_objects=self.custom_objects
                )

        return fit_arguments

    def _get_model(self) -> Model:
        '''Gets a model structure

        Returns:
            (Model): a Keras model
        '''
        # The base model will be loaded by keras's internal functions
        # Keras uses the `get_file` function to load all files from a cache directory (or from the internet)
        # Per default, all keras's application try to load models files from the keras's cache directory (.keras)
        # However, these application do not have a parameter to change the default directory, but we want all data
        # inside the project's data directory (especially as we do not always have access to the internet).
        # To do so, we change keras's internal function `get_file` to use a directory inside our package as the cache dir.
        # This allows us to predownload a model from anysource.
        # IMPORTANT : we need to reset the `get_file` function at the end of this function

        # Monkey patching : https://stackoverflow.com/questions/5626193/what-is-monkey-patching
        cache_dir = os.path.join(utils.get_data_path(), 'cache_keras')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        old_get_file = data_utils.get_file
        data_utils.get_file = partial(data_utils.get_file, cache_dir=cache_dir)

        # Check if the base model exists, otherwise try to download it
        # VGG 16
        # base_model_name = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        # base_model_path = os.path.join(utils.get_data_path(), 'cache_keras', 'models', base_model_name)
        {% if vgg16_weights_backup_urls is not none %}# base_model_backup_urls = [
        {%- for item in vgg16_weights_backup_urls %}
            # '{{item}}',
        {%- endfor %}
        # ]{% else %}# base_model_backup_urls = []{% endif %}
        # EfficientNetB6
        base_model_name = 'efficientnetb6_notop.h5'
        base_model_path = os.path.join(utils.get_data_path(), 'cache_keras', 'models', base_model_name)
        {% if efficientnetb6_weights_backup_urls is not none %}base_model_backup_urls = [
        {%- for item in efficientnetb6_weights_backup_urls %}
            '{{item}}',
        {%- endfor %}
        ]{% else %}base_model_backup_urls = []{% endif %}
        # Check model presence
        if not os.path.exists(base_model_path):
            try:
                utils.download_url(base_model_backup_urls, base_model_path)
            except Exception:
                # If we can't download it, we let the function crash alone
                self.logger.warning("Can't find / download the base model for transfer learning application.")

        # Get input/output dimensions
        input_shape = (self.width, self.height, self.depth)
        num_classes = len(self.list_classes)

        # Process
        input_layer = Input(shape=input_shape)

        # Example VGG16 - to be used with tensorflow.keras.applications.vgg16.preprocess_input - cf _get_preprocess_input
        # We must use training=False to use the batch norm layers in inference mode
        # (cf. https://keras.io/guides/transfer_learning/)
        # base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        # base_model.trainable = False  # We disable the first layers
        # x = base_model(input_layer, training=False)
        # x = Flatten()(x)

        # Example EfficientNetB6 - to be used with tensorflow.keras.applications.efficientnet.preprocess_input - cf _get_preprocess_input
        # We must use training=False to use the batch norm layers in inference mode
        # (cf. https://keras.io/guides/transfer_learning/)
        base_model = EfficientNetB6(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False  # We disable the first layers
        x = base_model(input_layer, training=False)
        x = GlobalAveragePooling2D()(x)

        # Last layer
        activation = 'softmax'
        out = Dense(num_classes, activation=activation, kernel_initializer='glorot_uniform')(x)

        # Set model
        model = Model(inputs=input_layer, outputs=[out])

        # Get lr & compile
        lr = self.keras_params['learning_rate'] if 'learning_rate' in self.keras_params.keys() else 0.001
        self._compile_model(model, lr=lr)

        # Display model
        if self.logger.getEffectiveLevel() < logging.ERROR:
            model.summary()

        # Try to save model as png if level_save > 'LOW'
        if self.level_save in ['MEDIUM', 'HIGH']:
            self._save_model_png(model)

        # We reset the `get_file` function (cf. explanations)
        data_utils.get_file = old_get_file

        # Return
        return model

    def _compile_model(self, model: Model, lr: float) -> None:
        '''Compiles the model. This is usually done in _get_model, but adding a function here
        helps to simplify the fine-tuning step.

        Args:
            model (Model): Model to be compiled
            lr (float): Learning rate to be used
        '''
        # Set optimizer
        decay = self.keras_params['decay'] if 'decay' in self.keras_params.keys() else 0.0
        self.logger.info(f"Learning rate: {lr}")
        self.logger.info(f"Decay: {decay}")
        optimizer = Adam(lr=lr, decay=decay)

        # Set loss & metrics
        loss = 'categorical_crossentropy'
        metrics: List[Union[str, Callable]] = ['accuracy']

        # Compile model
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def _get_preprocess_input(self) -> Union[Callable, None]:
        '''Gets the preprocessing to be used before feeding images to the NN

        Returns:
            (Callable | None): Preprocessing function
        '''
        # Preprocessing VGG 16
        # return vgg16_preprocess_input
        # Preprocessing efficient net
        return enet_preprocess_input

    def _get_second_callbacks(self) -> list:
        '''Gets model callbacks - second fit

        Returns:
            list: List of callbacks
        '''
        # We start by renaming 'best.hdf5' & 'logger.csv'
        if os.path.exists(os.path.join(self.model_dir, 'best.hdf5')):
            os.rename(os.path.join(self.model_dir, 'best.hdf5'), os.path.join(self.model_dir, 'best_initial_fit.hdf5'))
        if os.path.exists(os.path.join(self.model_dir, 'logger.csv')):
            os.rename(os.path.join(self.model_dir, 'logger.csv'), os.path.join(self.model_dir, 'logger_initial_fit.csv'))

        # Get classic callbacks
        callbacks = [EarlyStopping(monitor='val_loss', patience=self.second_patience, restore_best_weights=True)]
        if self.level_save in ['MEDIUM', 'HIGH']:
            callbacks.append(
                ModelCheckpoint(
                    filepath=os.path.join(self.model_dir, 'best.hdf5'), monitor='val_loss', save_best_only=True, mode='auto'
                )
            )
        callbacks.append(CSVLogger(filename=os.path.join(self.model_dir, 'logger.csv'), separator='{{default_sep}}', append=False))
        callbacks.append(TerminateOnNaN())

        # Get LearningRateScheduler
        scheduler = self._get_second_learning_rate_scheduler()
        if scheduler is not None:
            callbacks.append(LearningRateScheduler(scheduler))

        # Manage tensorboard
        if self.level_save in ['HIGH']:
            # Get log directory
            models_path = utils.get_models_path()
            tensorboard_dir = os.path.join(models_path, 'tensorboard_logs_second_fit')
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

    def _get_second_learning_rate_scheduler(self) -> Union[Callable, None]:
        '''Fonction to define a Learning Rate Scheduler - second fit
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

    def save(self, json_data: Union[dict, None] = None) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        '''
        # Save configuration JSON
        if json_data is None:
            json_data = {}

        json_data['with_fine_tune'] = self.with_fine_tune
        json_data['second_epochs'] = self.second_epochs
        json_data['second_lr'] = self.second_lr
        json_data['second_patience'] = self.second_patience
        # Add _compile_model code if not in json_data
        if '_compile_model' not in json_data.keys():
            json_data['_compile_model'] = pickle.source.getsourcelines(self._compile_model)[0]
        # Add _get_second_learning_rate_scheduler code if not in json_data
        if '_get_second_learning_rate_scheduler' not in json_data.keys():
            json_data['_get_second_learning_rate_scheduler'] = pickle.source.getsourcelines(self._get_second_learning_rate_scheduler)[0]

        # Save
        super().save(json_data=json_data)

    def reload_from_standalone(self, **kwargs) -> None:
        '''Reloads a model from its configuration and "standalones" files
        - /!\\ Experimental /!\\ -

        Kwargs:
            configuration_path (str): Path to configuration file
            hdf5_path (str): Path to hdf5 file
            preprocess_input_path (str): Path to preprocess input file
        Raises:
            ValueError: If configuration_path is None
            ValueError: If hdf5_path is None
            ValueError: If preprocess_input_path is None
            FileNotFoundError: If the object configuration_path is not an existing file
            FileNotFoundError: If the object hdf5_path is not an existing file
            FileNotFoundError: If the object preprocess_input_path is not an existing file
        '''
        # Retrieve args
        configuration_path = kwargs.get('configuration_path', None)
        hdf5_path = kwargs.get('hdf5_path', None)
        preprocess_input_path = kwargs.get('preprocess_input_path', None)

        # Checks
        if configuration_path is None:
            raise ValueError("The argument configuration_path can't be None")
        if hdf5_path is None:
            raise ValueError("The argument hdf5_path can't be None")
        if preprocess_input_path is None:
            raise ValueError("The argument preprocess_input_path can't be None")
        if not os.path.exists(configuration_path):
            raise FileNotFoundError(f"The file {configuration_path} does not exist")
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"The file {hdf5_path} does not exist")
        if not os.path.exists(preprocess_input_path):
            raise FileNotFoundError(f"The file {preprocess_input_path} does not exist")

        # Load confs
        with open(configuration_path, 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        # Can't set int as keys in json, so need to cast it after reloading
        # dict_classes keys are always ints
        if 'dict_classes' in configs.keys():
            configs['dict_classes'] = {int(k): v for k, v in configs['dict_classes'].items()}
        elif 'list_classes' in configs.keys():
            configs['dict_classes'] = {i: col for i, col in enumerate(configs['list_classes'])}

        # Set class vars
        # self.model_name = # Keep the created name
        # self.model_dir = # Keep the created folder
        self.nb_fit = configs.get('nb_fit', 1)  # Consider one unique fit by default
        self.trained = configs.get('trained', True)  # Consider trained by default

        # Try to read the following attributes from configs and, if absent, keep the current one
        for attribute in ['model_type', 'list_classes', 'dict_classes', 'level_save',
                          'batch_size', 'epochs', 'validation_split', 'patience',
                          'width', 'height', 'depth', 'color_mode', 'in_memory',
                          'data_augmentation_params', 'nb_train_generator_images_to_save',
                          'with_fine_tune', 'second_epochs', 'second_lr', 'second_patience',
                          'keras_params']:
            setattr(self, attribute, configs.get(attribute, getattr(self, attribute)))

        # Reload model
        self.model = load_model_keras(hdf5_path, custom_objects=self.custom_objects)

        # Reload preprocess_input
        with open(preprocess_input_path, 'rb') as f:
            self.preprocess_input = pickle.load(f)

        # Save best hdf5 in new folder
        new_hdf5_path = os.path.join(self.model_dir, 'best.hdf5')
        shutil.copyfile(hdf5_path, new_hdf5_path)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")

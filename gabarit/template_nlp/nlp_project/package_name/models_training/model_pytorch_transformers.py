#!/usr/bin/env python3

## PyTorch transformer model
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
# - ModelPyTorchTransformers -> Model for predictions via tranformers pytorch

import os
import json
import shutil
import logging
import numpy as np
import seaborn as sns
from tqdm import tqdm
from typing import no_type_check, Union, Callable, Any, Tuple

import torch
import pytorch_lightning as pl
from torch.nn import Softmax, Sigmoid
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.sampler import RandomSampler
from transformers import AdamW, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

from {{package_name}} import utils
from {{package_name}}.models_training import utils_deep_torch
from {{package_name}}.models_training.model_pytorch import ModelPyTorch

sns.set(style="darkgrid")
tqdm.pandas()


class ModelPyTorchTransformers(ModelPyTorch):
    '''Model for predictions via tranformers pytorch'''

    _default_name = 'model_pytorch_transformers'

    def __init__(self, transformer_name: Union[str, None] = None, max_sequence_length: int = 256,
                 tokenizer_special_tokens: Union[tuple, None] = None, padding: str = "max_length",
                 truncation: bool = True, **kwargs) -> None:
        '''Initialization of the class (see ModelClass & ModelPyTorch for more arguments)

        Args:
            transformer_name (str): Name of the transformer to use
        Kwargs:
            max_sequence_length (int): Maximum number of words per sequence (ie. sentences)
            tokenizer_special_tokens (tuple): Set of "special tokens" for the tokenizer
            padding (str): Tokenizer's padding strategy
            truncation (bool): Tokenizer's padding truncation
        '''
        # Init.
        super().__init__(**kwargs)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Params
        self.transformer_name = transformer_name
        self.max_sequence_length = max_sequence_length
        if tokenizer_special_tokens is None:
            tokenizer_special_tokens = tuple()
        self.tokenizer_special_tokens = tokenizer_special_tokens
        self.padding = padding
        self.truncation = truncation

        # Retrieve tokenizer
        self.tokenizer: Any = None
        if self.transformer_name is not None:
            self.tokenizer = self._get_tokenizer()
        else:
            self.logger.warning("No transformer name specified. We can't load the tokenizer.")
            self.logger.warning("No training possible if a tokenizer is not loaded (eg. via reload_from_standalone).")

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    @no_type_check
    def predict_proba(self, x_test, limit_predict_simple: int = 5, **kwargs) -> np.ndarray:
        '''Predicts probabilities on the test dataset

        Args:
            x_test (?): Array-like or sparse matrix, shape = [n_samples]
        Kwargs:
            limit_predict_simple (bool): Maximal number of elements to use a "simple" predict (without dataloaders)
        Returns:
            (np.ndarray): Array, shape = [n_samples, n_classes]
        '''
        # Simplified predictions (faster when considering a small number of data)
        if len(x_test) <= limit_predict_simple:
            input_ids, attention_mask, _ = self._convert_inputs(x_test, None)
            # NEW FIX: convert to correct device
            self.model.convert_network_to_device()  # Convert to gpu if available
            input_ids = self.model.to_device(input_ids)
            attention_mask = self.model.to_device(attention_mask)
            # Switch to eval mode
            self.model.eval()
            # Get predictions
            logits_torch = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = np.vstack([logit.cpu() for logit in logits_torch])
            probas = self.model.get_probas_from_logits(logits)
            # Switch back to train mode
            self.model.train()
        # Predictions with DataLoader (slower when considering a small number of data)
        else:
            test_dl = self._get_test_dataloader(self.batch_size, x_test, y_test_dummies=None)  # We can change the batch size
            trainer = pl.Trainer(default_root_dir=self.model_dir, checkpoint_callback=False, logger=False,
                                 gpus=1 if torch.cuda.is_available() else 0)
            # Test on x_test
            trainer.test(model=self.model, test_dataloaders=test_dl)
            # Retrieve proba & return it
            probas = self.model.test_probas
        return probas

    def _get_tokenizer(self) -> Any:
        '''Retrieves the tokenizer'''
        # Get tokenizer path
        transformers_path = utils.get_transformers_path()
        transformer_path = os.path.join(transformers_path, self.transformer_name)

        # Check if available locally
        if os.path.exists(transformer_path):
            self.logger.info("Use the local tokenizer")
            final_transformer_localisation = transformer_path
        else:
            self.logger.warning("Can't find the local tokenizer.")
            self.logger.warning("We try to get it from the web.")
            final_transformer_localisation = self.transformer_name

        # Retrieve tokenizer
        tokenizer = AutoTokenizer.from_pretrained(final_transformer_localisation)

        # Add special tokens
        if self.tokenizer_special_tokens:
            tokenizer.add_tokens(self.tokenizer_special_tokens)

        # Return
        return tokenizer

    def _get_transformer(self) -> Any:
        '''Gets a transformer

        Raises:
            ValueError: If list_classes has not been set (ie. fit not called)
        Returns:
            (?): a transformer model
        '''
        if self.list_classes is None:
            raise ValueError("The method '_get_transformer' can't be called if 'fit' has not been called at least once")

        # Set parameters
        summary_first_dropout = self.pytorch_params['transformer_summary_first_dropout'] if 'transformer_summary_first_dropout' in self.pytorch_params.keys() else 0.1
        attention_dropout = self.pytorch_params['transformer_attention_dropout'] if 'transformer_attention_dropout' in self.pytorch_params.keys() else 0.1
        dropout = self.pytorch_params['transformer_dropout'] if 'transformer_dropout' in self.pytorch_params.keys() else 0.2
        self.logger.info(f"Transformer's summary first dropout : {summary_first_dropout}")
        self.logger.info(f"Transformer's attention dropout : {attention_dropout}")
        self.logger.info(f"Transformer's dropout : {dropout}")

        # Update pytorch_params for saving purposes
        self.pytorch_params['transformer_summary_first_dropout'] = summary_first_dropout
        self.pytorch_params['transformer_attention_dropout'] = attention_dropout
        self.pytorch_params['transformer_dropout'] = dropout

        # Get transformer path
        transformers_path = utils.get_transformers_path()
        transformer_path = os.path.join(transformers_path, self.transformer_name)

        # Check if available locally
        if os.path.exists(transformer_path):
            self.logger.info("Use the local transformer.")
            final_transformer_localisation = transformer_path
        else:
            self.logger.warning("Can't find the local transformer.")
            self.logger.warning("We try to get it from the web.")
            final_transformer_localisation = self.transformer_name

        # AutoConfig base on labels
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=final_transformer_localisation,
            summary_first_dropout=summary_first_dropout,
            attention_dropout=attention_dropout,
            dropout=dropout,
            num_labels=len(self.list_classes),
            id2label=self.dict_classes,
            label2id={v: i for i, v in self.dict_classes.items()},
        )

        # Retrieve transformer model
        transformer_model = AutoModelForSequenceClassification.from_pretrained(final_transformer_localisation, config=config)

        # Resize embedding if some special tokens added
        if self.tokenizer_special_tokens:
            transformer_model.resize_token_embeddings(len(self.tokenizer))

        # Allow full retraining
        for param in transformer_model.parameters():
            param.requires_grad = True

        # Returns tranformer model
        return transformer_model

    def _get_model(self, train_dataloader_size: Union[int, None] = None) -> Any:
        '''Gets a model structure

        Kwargs:
            train_dataloader_size (int): number of batch per epochs. Useful to set a learning rate scheduler
        Returns:
            (?): a PyTorch model
        '''
        if self.tokenizer is None:
            raise AttributeError("The tokenizer is not set. Can't get the model.")
        # Retrieve transformer
        transformer_model = self._get_transformer()

        # Set parameters
        lr = self.pytorch_params.get('learning_rate', 2.0e-05)
        decay = self.pytorch_params.get('decay', 0.0)
        adam_epsilon = self.pytorch_params.get('adam_epsilon', 1.5e-06)
        gradient_clip_val = self.pytorch_params.get('gradient_clip_val', 1.0)
        warmup_proportion = self.pytorch_params.get('warmup_proportion', 0.2)
        lr = self.pytorch_params.get('learning_rate', 2.0e-05)
        decay = self.pytorch_params.get('decay', 0.0)
        adam_epsilon = self.pytorch_params.get('adam_epsilon', 1.5e-06)
        gradient_clip_val = self.pytorch_params.get('gradient_clip_val', 1.0)
        warmup_proportion = self.pytorch_params.get('warmup_proportion', 0.2)
        run_lr_scheduler = self.pytorch_params.get('run_lr_scheduler', False)
        self.logger.info(f"Learning rate: {lr}")
        self.logger.info(f"Decay: {decay}")
        self.logger.info(f"Adam's epsilon: {adam_epsilon}")
        self.logger.info(f"Gradient clipping: {gradient_clip_val}")
        if run_lr_scheduler:
            self.logger.info("Using a learning rate scheduler")
            self.logger.info(f"Warmup proportion: {warmup_proportion}")

        # Update pytorch_params for saving purposes
        self.pytorch_params['learning_rate'] = lr
        self.pytorch_params['decay'] = decay
        self.pytorch_params['adam_epsilon'] = adam_epsilon
        self.pytorch_params['gradient_clip_val'] = gradient_clip_val
        self.pytorch_params['warmup_proportion'] = warmup_proportion
        self.pytorch_params['run_lr_scheduler'] = run_lr_scheduler

        # Loss function
        loss_func = utils_deep_torch.focal_loss if self.multi_label else utils_deep_torch.categorical_crossentropy_from_logits
        metrics_func = self.get_metrics_simple_multilabel if self.multi_label else self.get_metrics_simple_monolabel

        # Return
        model = TaskClass(multi_label=self.multi_label, transformer_model=transformer_model,
                          tokenizer=self.tokenizer, loss_func=loss_func, probas_to_classes=self.get_classes_from_proba,
                          get_metrics=metrics_func, output_path=self.model_dir,
                          epochs=self.epochs, lr=lr, decay=decay, adam_epsilon=adam_epsilon,
                          gradient_clip_val=gradient_clip_val, run_lr_scheduler=run_lr_scheduler,
                          warmup_proportion=warmup_proportion, train_dataloader_size=train_dataloader_size)
        return model

    def _convert_inputs(self, x, y=None) -> Tuple[Any, Any, Any]:
        '''Converts (x, y) data to the model's input format

        Args:
            x (?): Array-like, shape = [n_samples, n_features]
        Kwargs:
            y (?): Array-like, shape = [n_samples, n_features]
        '''
        # First, cast to numpy arrays
        x = np.array(x)
        if y is not None:
            y = np.array(y)

        # Get features from tokenizer
        features = []
        for i in tqdm(range(len(x))):
            text = x[i]
            label = y[i] if y is not None else None
            entry = self.tokenizer(text, padding=self.padding, truncation=self.truncation, max_length=self.max_sequence_length)
            features.append((entry["input_ids"], entry["attention_mask"], label))

        # Retrieve inputs, masks & labels
        all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)
        if y is not None:
            all_label = torch.tensor([f[2] for f in features], dtype=torch.float32)
        else:
            all_label = None
        return all_input_ids, all_input_mask, all_label

    def _get_train_dataloader(self, batch_size: int, x_train, y_train_dummies=None) -> DataLoader:
        '''Prepares the input data for the model

        Args:
            batch_size (int): Train batch size
            x_train (?): Array-like, shape = [n_samples, n_features]
            y_train_dummies (?): Array-like, shape = [n_samples, n_features]
        Returns:
            (DataLoader): Data loader
        '''
        all_input_ids, all_input_mask, all_label = self._convert_inputs(x_train, y_train_dummies)
        if all_label is not None:
            train_dataset = TensorDataset(all_input_ids, all_input_mask, all_label)
        else:
            raise ValueError("No label associated with the training set...")
        train_dl = DataLoader(train_dataset, num_workers=0, batch_size=batch_size, sampler=RandomSampler(train_dataset))
        return train_dl

    def _get_test_dataloader(self, batch_size: int, x_test, y_test_dummies=None) -> DataLoader:
        '''Prepares the input data for the model

        Args:
            x_test (?): Array-like, shape = [n_samples, n_features]
            batch_size (int): Test batch size
        Kwargs:
            y_test_dummies (?): Array-like, shape = [n_samples, n_features]
        Returns:
            (DataLoader): Data loader
        '''
        all_input_ids, all_input_mask, all_label = self._convert_inputs(x_test, y_test_dummies)
        if all_label is not None:
            test_dataset = TensorDataset(all_input_ids, all_input_mask, all_label)
        else:
            test_dataset = TensorDataset(all_input_ids, all_input_mask)
        test_dl = DataLoader(test_dataset, num_workers=0, batch_size=batch_size)
        return test_dl

    def save(self, json_data: Union[dict, None] = None) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        '''
        # Save configuration JSON
        if json_data is None:
            json_data = {}

        # Add specific data
        json_data['transformer_name'] = self.transformer_name
        json_data['max_sequence_length'] = self.max_sequence_length
        json_data['tokenizer_special_tokens'] = self.tokenizer_special_tokens
        json_data['padding'] = self.padding
        json_data['truncation'] = self.truncation

        # Save
        super().save(json_data=json_data)

    def reload_model(self, model_path: str, **kwargs) -> Any:
        '''Reloads a model saved with ModelCheckpoint

        Args:
            model_path (str): Checkpoint's full path
        Kwargs:
            kwargs: Dict of kwargs to override predefined params (TO BE CHECKED !!!)
        Returns:
            ?: Pytorch lightning model
        '''
        model = TaskClass.load_from_checkpoint(model_path, **kwargs)
        # Idea to test in order not to have mandatory hyperparameters savings
        # 1. Only save the weights of the model
        # 2. Save the tokenizer
        # 3. Reload :
        #   - Reload the weights (torch.load ...)
        #   - Reload the tokenizer
        #   - Initialize the class with these arguments

        # Set trained to true if not already true
        if not self.trained:
            self.trained = True
            self.nb_fit = 1

        return model

    def reload_from_standalone(self, **kwargs) -> None:
        '''Reloads a model from its configuration and "standalones" files
        - /!\\ Experimental /!\\ -

        Kwargs:
            configuration_path (str): Path to configuration file
            checkpoint_path (str): Checkpoint's full path
        Raises:
            ValueError: If configuration_path is None
            ValueError: If checkpoint_path is None
            FileNotFoundError: If the object configuration_path is not an existing file
            FileNotFoundError: If the object checkpoint_path is not an existing file
        '''
        # Retrieve args
        configuration_path = kwargs.get('configuration_path', None)
        checkpoint_path = kwargs.get('checkpoint_path', None)

        if configuration_path is None:
            raise ValueError("The argument configuration_path can't be None")
        if checkpoint_path is None:
            raise ValueError("The argument checkpoint_path can't be None")
        if not os.path.exists(configuration_path):
            raise FileNotFoundError(f"The file {configuration_path} does not exist")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"The file {checkpoint_path} does not exist")

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
        for attribute in ['x_col', 'y_col',
                          'list_classes', 'dict_classes', 'multi_label', 'level_save',
                          'batch_size', 'epochs', 'validation_split', 'patience',
                          'embedding_name', 'max_sequence_length', 'tokenizer_special_tokens',
                          'padding', 'truncation', 'pytorch_params', 'transformer_name']:
            setattr(self, attribute, configs.get(attribute, getattr(self, attribute)))

        # Reload model
        self.model = self.reload_model(checkpoint_path)
        # Do not forget to freeze
        self.model.freeze()  # type: ignore

        # Save best model in new folder
        new_checkpoint_path = os.path.join(self.model_dir, 'best_model.ckpt')
        shutil.copyfile(checkpoint_path, new_checkpoint_path)

        # Reload tokenizer
        self.tokenizer = self.model.tokenizer  # type: ignore


# Set task (i.e. our model)
class TaskClass(pl.LightningModule):
    @no_type_check  # Mypy does not detect changes in hparams
    def __init__(self, multi_label: bool, transformer_model, tokenizer, loss_func: Callable, probas_to_classes: Callable,
                 get_metrics: Callable, output_path: str, epochs: int = 5, lr: float = 2.0e-05, decay: float = 0.0,
                 adam_epsilon: float = 1.5e-06, gradient_clip_val: Union[float, int] = 1.0,
                 run_lr_scheduler: bool = False, warmup_proportion: float = 0.2,
                 train_dataloader_size: Union[int, None] = None) -> None:
        '''Initialization task class

        Args:
            multi_label: If working on a multilabel problem
            transformer_model: Transformer model loaded
            tokenizer: Tokenizer associated with the transformer model
            loss_func (callable): Loss function to be used
            probas_to_classes (callable): Function to get classes from probas
            get_metrics (callable): Function to get metrics
            output_path (str): Model directory
        Kwargs:
            epochs (int): Number of epochs to use
            lr (int): Initial learning rate to use
            decay (float): Decay to use
            adam_epsilon (float): Adam's epsilon to use
            gradient_clip_val (int | float): Gradient clipping value to use
            run_lr_scheduler (bool): Whether a learning rate scheduler should be used
            warmup_proportion (float): Warmup proportion to be used (for learning rate scheduler)
            train_dataloader_size (int): Number of batch per epochs. Useful to set a learning rate scheduler
        '''
        super(TaskClass, self).__init__()
        self.save_hyperparameters("multi_label", "transformer_model", "tokenizer", "loss_func",
                                  "probas_to_classes", "get_metrics", "output_path",
                                  "epochs", "lr", "decay", "adam_epsilon", "gradient_clip_val",
                                  "run_lr_scheduler", "warmup_proportion", "train_dataloader_size")
        self.multi_label: bool = self.hparams.multi_label
        self.model: Any = self.hparams.transformer_model
        self.tokenizer: Any = self.hparams.tokenizer
        self.loss_func: Callable = self.hparams.loss_func
        self.probas_to_classes: Callable = self.hparams.probas_to_classes
        self.get_metrics: Callable = self.hparams.get_metrics
        self.output_path: str = self.hparams.output_path
        self.epochs: int = self.hparams.epochs
        self.lr: float = self.hparams.lr
        self.decay: float = self.hparams.decay
        self.adam_epsilon: float = self.hparams.adam_epsilon
        self.gradient_clip_val: Union[int, float] = self.hparams.gradient_clip_val
        self.run_lr_scheduler: bool = self.hparams.run_lr_scheduler
        self.warmup_proportion: float = self.hparams.warmup_proportion
        self.train_dataloader_size: int = self.hparams.train_dataloader_size

    def to_device(self, obj) -> Any:  # Useful ?
        return obj.cuda() if torch.cuda.is_available() else obj

    def convert_network_to_device(self) -> None:
        if torch.cuda.is_available():
            self.to('cuda')
        else:
            self.to('cpu')

    def configure_optimizers(self) -> dict:
        '''Setup optimizers'''
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay_rate": self.decay,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay_rate": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, eps=self.adam_epsilon)
        return_dict = {'optimizer': optimizer}

        # Add LR scheduler if needed
        if self.run_lr_scheduler and self.train_dataloader_size is not None:
            total_steps = int(self.train_dataloader_size * self.epochs)
            warmup_steps = self.warmup_proportion
            if warmup_steps <= 1:
                warmup_steps = int(total_steps * warmup_steps)
            else:
                warmup_steps = int(warmup_steps)  # Typing
            # TODO : to be improved -> allow other LR schedulers
            lr_scheduler = LambdaLR(optimizer, lr_lambda=utils_deep_torch.LRScheduleWithWarmup(warmup_steps=warmup_steps, total_steps=total_steps))
            return_dict['lr_scheduler'] = {}
            return_dict['lr_scheduler']['scheduler'] = lr_scheduler
            return_dict['lr_scheduler']['interval'] = 'step'  # Update at each step
            return_dict['lr_scheduler']['frequency'] = 1
        # Return dict
        return return_dict

    def forward(self, input_ids, attention_mask=None):
        '''Forward step'''
        x = self.model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, head_mask=None)
        return x[0]  # Return logits

    def training_step(self, batch, batch_idx):
        '''Training step'''
        input_ids, attention_mask, label = batch
        logits = self.forward(input_ids, attention_mask)
        loss = self.loss_func(logits, label)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        '''Validation step'''
        input_ids, attention_mask, label = batch
        logits = self.forward(input_ids, attention_mask)
        loss = self.loss_func(logits, label)
        outputs = {
            "val_loss": loss,
            "logit": logits.cpu().numpy(),  # Useful for metrics
            "label": label.cpu().numpy(),  # Useful for metrics
        }
        return outputs

    def validation_epoch_end(self, outputs) -> None:
        '''On epochs ends, gets validation metrics'''
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()  # val_loss = loss mean among batches
        y_true = np.vstack([x["label"] for x in outputs])
        logits = np.vstack([x["logit"] for x in outputs])
        probas = self.get_probas_from_logits(logits)
        y_pred = np.array(list(map(self.probas_to_classes, probas)))
        # If not multi-labels, retrieves classes from OHE vectors
        if not self.multi_label:
            y_true = np.array(list(map(self.probas_to_classes, y_true)))
        # Get metrics
        metrics_df = self.get_metrics(y_true, y_pred)
        all_metrics = metrics_df[metrics_df.Label == 'All']
        # Logs metrics
        self.log("val_loss", avg_loss, prog_bar=True, on_step=False, logger=True)
        for col in all_metrics.columns:
            if col != 'Label' and all_metrics[col].values[0] is not None:
                self.log(f"val_{col}", all_metrics[col].values[0], prog_bar=True, on_step=False, logger=True)

    def test_step(self, batch, batch_idx):
        '''Test step'''
        input_ids, attention_mask = batch[:2]
        logits = self.forward(input_ids, attention_mask)
        outputs = {"logit": logits.cpu().numpy()}
        return outputs

    def test_epoch_end(self, outputs) -> None:
        '''On epochs ends, gets test metrics'''
        logits = np.vstack([x["logit"] for x in outputs])
        probas = self.get_probas_from_logits(logits)
        setattr(self, "test_probas", probas)
        # Nothing to log ?

    def get_probas_from_logits(self, logits) -> np.ndarray:
        '''Function to retrieve probas from logits (in numpy format)'''
        if self.multi_label:
            probas = Sigmoid()(torch.from_numpy(logits)).numpy()
        else:
            probas = Softmax(dim=-1)(torch.from_numpy(logits)).numpy()
        return probas


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")

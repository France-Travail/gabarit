#!/usr/bin/env python3

## Model embedding + LSTM
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
import math
import logging
import numpy as np
import seaborn as sns
from tqdm import tqdm
from functools import partial
from typing import List, Union, Any, Tuple, no_type_check

import torch
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.multiprocessing import Pool, cpu_count
from transformers import AutoConfig, AutoModelWithLMHead, AutoTokenizer, AdamW, get_linear_schedule_with_warmup

from {{package_name}} import utils
from {{package_name}}.models_training.model_pytorch import ModelPyTorch

sns.set(style="darkgrid")
tqdm.pandas()

TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelPyTorchLanguageModel(ModelPyTorch):
    '''Model for predictions via tranformers pytorch'''

    _default_name = 'model_pytorch_language_model'

    def __init__(self, transformer_name: str, max_sequence_length: int = 256,
                 tokenizer_special_tokens: list = ["xxbos", "xxeos"],
                 padding: str = "max_length", truncation: bool = True, **kwargs) -> None:
        '''Initialization of the class (see ModelClass & ModelPyTorch for more arguments)

        Args:
            transformer_name (str): Name of the transformer to use
        Kwargs:
            max_sequence_length (int): Maximum number of words per sequence (ie. sentences)
            tokenizer_special_tokens (tuple): Set of "special tokens" for the tokenizer
        '''

        # Init.
        super().__init__(**kwargs)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Params
        self.transformer_name = transformer_name
        self.max_sequence_length = max_sequence_length
        self.tokenizer_special_tokens = tokenizer_special_tokens
        self.padding = padding
        self.truncation = truncation
        # Retrieve tokenizer
        self.tokenizer = self._get_tokenizer()

    # TODO: to be improved / removed ? / commented
    def forward(self) -> None:
        sentence = "Le chat mange une pomme."
        token_ids = torch.tensor([self.tokenizer.encode(sentence)])
        self.model.convert_network_to_device()  # Convert to gpu if available
        token_ids = self.model.to_device(token_ids)
        self.model.eval()
        last_layer = self.model(token_ids)[0]
        self.model.train()
        print(last_layer.shape)
        print(last_layer)

    def _get_tokenizer(self) -> Any:
        '''Retrieves the tokenizer

        Returns:
            None
        '''
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
        self.logger.info(f"Transformer's summary first dropout : {summary_first_dropout}")

        # Update pytorch_params for saving purposes
        self.pytorch_params['transformer_summary_first_dropout'] = summary_first_dropout

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
        )

        # Get model base on config
        transformer_model = AutoModelWithLMHead.from_pretrained(final_transformer_localisation, config=config)

        # Resize embedding if some special tokens added
        if self.tokenizer_special_tokens:
            transformer_model.resize_token_embeddings(len(self.tokenizer))
        if hasattr(transformer_model, "pred_layer"):  # Workaround for a flaubert bug
            transformer_model.pred_layer.n_words = len(self.tokenizer)

        # Allow full retraining
        for param in transformer_model.parameters():
            param.requires_grad = True

        # Returns tranformer model
        return transformer_model

    def _get_model(self, train_dataloader_size: Union[int, None] = None) -> Any:
        '''Gets a model structure

        Kwargs:
            train_dataloader_size (int): number of batch per epochs. Useful to set a learning rate scheduler
        Raises:
            AttributeError: If tokenizer has not been set
        Returns:
            (?): a PyTorch model
        '''
        if self.tokenizer is None:
            raise AttributeError("The tokenizer is not set. Can't get the model.")

        # Tokenizer required to set up transformer model
        tokenizer = self.tokenizer
        transformer_model = self._get_transformer()

        # Set parameters
        output_path = self.model_dir
        epochs = self.epochs
        lr = self.pytorch_params.get('learning_rate', 2.0e-05)
        decay = self.pytorch_params.get('decay', 0.0)
        adam_epsilon = self.pytorch_params.get('adam_epsilon', 1.5e-06)
        gradient_clip_val = self.pytorch_params.get('gradient_clip_val', 1.0)
        warmup_proportion = self.pytorch_params.get('warmup_proportion', 0.2)
        run_gpus = True if str(TORCH_DEVICE) == 'cuda' else False
        self.logger.info(f"Learning rate: {lr}")
        self.logger.info(f"Decay: {decay}")
        self.logger.info(f"Adam's epsilon: {adam_epsilon}")
        self.logger.info(f"Gradient clipping: {gradient_clip_val}")
        # TODO : Put LR scheduler optional ?
        if train_dataloader_size is not None:
            self.logger.info("Using a learning rate scheduler ...")
            self.logger.info(f"Warmup proportion: {warmup_proportion}")

        # Update pytorch_params for saving purposes
        self.pytorch_params['learning_rate'] = lr
        self.pytorch_params['decay'] = decay
        self.pytorch_params['adam_epsilon'] = adam_epsilon
        self.pytorch_params['gradient_clip_val'] = gradient_clip_val
        self.pytorch_params['warmup_proportion'] = warmup_proportion

        # Set task (i.e. our model)
        # Return
        task = TaskClass(transformer_model=transformer_model, tokenizer=tokenizer, lr=lr, topK_val=3,
                         adam_epsilon=adam_epsilon, output_path=output_path, decay=decay,
                         warmup_proportion=warmup_proportion, epochs=epochs, run_gpus=run_gpus,
                         monitor=["val_loss", "loss"], gradient_clip_val=gradient_clip_val,
                         train_dataloader_size=train_dataloader_size)
        return task

    # y_train_dummies useless for a langage model; here for compatibility purposes with ModelPytorch
    def _get_train_dataloader(self, batch_size: int, x_train, y_train_dummies=None) -> DataLoader:
        '''Prepares the input data for the model

        Args:
            batch_size (int): Train batch size
            x_train (?): Array-like, shape = [n_samples, n_features]
            y_train_dummies (?): Array-like, shape = [n_samples, n_features]
        Returns:
            (?): Data loader
        '''
        ds = TextDataset(self.tokenizer, texts=x_train, block_size=256)
        collate = partial(pad_sequence, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        train_dl = DataLoader(ds, num_workers=4, sampler=None, batch_size=batch_size, collate_fn=collate)
        return train_dl

    # y_test_dummies useless for a langage model; here for compatibility purposes with ModelPytorch
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
        ds = TextDataset(self.tokenizer, texts=x_test, block_size=256)
        collate = partial(pad_sequence, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        test_dl = DataLoader(ds, num_workers=4, sampler=None, batch_size=batch_size, collate_fn=collate)
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

    # TODO ?
    # def reload_from_standalone


class TextDataset(Dataset):

    BOS, EOS = "xxbos", "xxeos"

    def __init__(self, tokenizer, texts: List[str], block_size: int) -> None:

        self.examples = []

        # We use all cpu cores and split the examples into cpu_cores chunks
        # (or 10k per chunk, cf. https://stackoverflow.com/a/43817408)
        cpu_cores = cpu_count()
        cpu_cores = 1
        chunk_size = max(100, min(10000, math.floor(len(texts) / cpu_cores)))
        n_executors = min(cpu_cores, math.ceil(len(texts) / chunk_size))
        print(chunk_size)
        print(n_executors)
        with Pool(n_executors) as pool:
            pool_imap = pool.imap(
                partial(self._to_ids, tokenizer=tokenizer),
                iter(texts),
                chunk_size,
            )
            tokenized_text = list(tqdm(pool_imap, total=len(texts), desc="Tokenization"))

        tokenized_text = [y for x in tokenized_text for y in x]

        for i in range(0, len(tokenized_text) - block_size + 1, block_size):
            self.examples.append(
                tokenizer.build_inputs_with_special_tokens(
                    tokenized_text[i : i + block_size]  # noqa: E203
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, item) -> Any:
        return torch.tensor(self.examples[item])

    def _to_ids(self, text, tokenizer) -> Any:
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"xxbos {text} xxeos"))


class TaskClass(pl.LightningModule):
    @no_type_check  # Mypy does not detect changes in hparams
    def __init__(self, transformer_model, tokenizer, lr: float, topK_val: int, adam_epsilon: float,
                 output_path: str, decay: float, warmup_proportion: float, epochs: int, run_gpus: bool, monitor,
                 gradient_clip_val, train_dataloader_size: Union[int, None]) -> None:
        super(TaskClass, self).__init__()
        self.save_hyperparameters("transformer_model", "tokenizer", "lr", "topK_val", "adam_epsilon",
                                  "output_path", "decay", "warmup_proportion", "epochs", "run_gpus",
                                  "monitor", "gradient_clip_val", "train_dataloader_size")
        self.model: Any = self.hparams.transformer_model
        self.tokenizer: Any = self.hparams.tokenizer
        self.lr: float = self.hparams.lr
        self.topK_val: int = self.hparams.topK_val
        self.adam_epsilon: float = self.hparams.adam_epsilon
        self.output_path: str = self.hparams.output_path
        self.decay: float = self.hparams.decay
        self.warmup_proportion: float = self.hparams.warmup_proportion
        self.epochs: int = self.hparams.epochs
        self.run_gpus: bool = self.hparams.run_gpus
        self.monitor: Any = self.hparams.monitor
        self.gradient_clip_val: Any = self.hparams.gradient_clip_val
        self.post_init()
        self.tokenizer = self.hparams.tokenizer  # TODO: useful ?
        self.train_dataloader_size: Union[int, None] = self.hparams.train_dataloader_size

    def post_init(self) -> None:
        self.model = self.to_device(self.model)

    def to_device(self, obj) -> Any:
        return obj.cuda() if self.run_gpus else obj

    def configure_optimizers(self) -> dict:
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
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            eps=self.adam_epsilon,
        )

        # Hack as lightning does not support step LR schedulers every iteration
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/624
        if self.train_dataloader_size is not None:
            t_total = int(self.train_dataloader_size) * self.epochs
            warmup_steps = self.warmup_proportion
            if warmup_steps <= 1:
                warmup_steps = int(t_total * warmup_steps)
            self.lr_sched = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            self.lr_sched = None
        # return optimizer

        return_dict = {'optimizer': optimizer}
        if self.lr_sched is not None:
            # lr_scheduler = LambdaLR(optimizer, lr_lambda=utils_deep_torch.LRScheduleWithWarmup(warmup_steps=warmup_steps, total_steps=total_steps))
            return_dict['lr_scheduler'] = {}
            return_dict['lr_scheduler']['scheduler'] = self.lr_sched
            return_dict['lr_scheduler']['interval'] = 'step'  # Update at each step
            return_dict['lr_scheduler']['frequency'] = 1
        return return_dict

    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None, on_tpu = None, using_native_amp=None, using_lbfgs=None):
    #     optimizer.step()
    #     self.lr_sched.step()
    #     optimizer.zero_grad()

    def mask_tokens(self, inputs: torch.Tensor, tokenizer) -> Tuple[Any, Any]:
        """Prepare masked tokens inputs/labels for masked language modeling
        : 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training
        # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = self.to_device(torch.full(labels.shape, 0.15))
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]

        probability_matrix.masked_fill_(self.to_device(torch.tensor(special_tokens_mask, dtype=torch.bool)), value=0.0)
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(self.to_device(torch.full(labels.shape, 0.8))).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(self.to_device(torch.full(labels.shape, 0.5))).bool() & masked_indices & ~indices_replaced
        )
        random_words = self.to_device(torch.randint(len(tokenizer), labels.shape, dtype=torch.long))
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def forward(self, inputs, labels=None):
        inputs = self.to_device(inputs)
        if labels is not None:
            labels = self.to_device(labels)
        if self.model.config.model_type == "flaubert":
            # god knows why !
            return self.model(inputs, labels=labels)
        else:
            return self.model(inputs, masked_lm_labels=labels)

    def training_step(self, batch, batch_idx):
        inputs, labels = self.mask_tokens(batch, self.tokenizer)
        outputs = self.forward(inputs, labels)
        loss, logits = outputs[:2]
        self.log("loss", loss, prog_bar=True, on_step=True, logger=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = self.mask_tokens(batch, self.tokenizer)
        outputs = self.forward(inputs, labels)
        loss, logits = outputs[:2]

        # Accuracy TopK
        labels = self.to_device(labels)
        masked = inputs.view(-1) == self.tokenizer.mask_token_id
        masked_labels = labels.view(-1)[masked]
        masked_logits = logits.view(-1, logits.shape[2])[masked]
        top_pred = torch.topk(masked_logits, self.topK_val, dim=1).indices
        in_top_k = (masked_labels == top_pred.T).any(axis=0).tolist()
        self.log("val_loss", float(loss), prog_bar=True, on_step=True, logger=True)
        return {"val_loss": float(loss), f"in_top_{self.topK_val}": in_top_k}

    @no_type_check  # TODO: super class returns None, shouldn't it be the same here ?
    def validation_epoch_end(self, outputs) -> dict:
        name = f"in_top_{self.topK_val}"
        val_loss = sum([out["val_loss"] for out in outputs]) / len(outputs)
        in_top_k: list = sum([out[name] for out in outputs], [])
        log = {
            "val_loss": float(val_loss),
            f"acc_top{self.topK_val}": self.div(sum(in_top_k), len(in_top_k)),
        }
        # self.log(f"val_loss: {float(val_loss)}, acc_top{self.topK_val}: {self.div(sum(in_top_k), len(in_top_k))}")
        self.log("val_loss", float(val_loss), prog_bar=True, on_step=False, logger=True)
        self.log(f"acc_topK_{self.topK_val}", self.div(sum(in_top_k), len(in_top_k)))
        return {"log": log, "val_loss": float(val_loss)}

    def div(self, a, b) -> Any:
        return a / b if b != 0 else np.nan

    def test_step(self, batch, batch_nb):
        raise NotImplementedError

    def test_epoch_end(self, outputs) -> Any:
        raise NotImplementedError

    def convert_network_to_device(self) -> None:
        if torch.cuda.is_available():
            self.to('cuda')
        else:
            self.to('cpu')


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")

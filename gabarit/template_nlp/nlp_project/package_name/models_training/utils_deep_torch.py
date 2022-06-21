#!/usr/bin/env python3

## Utils - tools-functions for deep_learning pytorch models

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


import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

# Get logger
logger = logging.getLogger(__name__)


def logloss_with_na(pred, target, **kwargs) -> Any:
    '''Evaluates log loss, without NAs

    Args:
        pred : The predictions done by a model
        target : The target
    Returns:
        (?) : The log loss
    '''
    pred, target = pred.contiguous().view(-1, 1), target.contiguous().view(-1, 1)
    notna = (target != -1).detach()
    pred, target = pred[notna], target[notna]
    return F.binary_cross_entropy_with_logits(pred, target, **kwargs)


def focal_loss(pred, target, gamma: float = 2.0, **kwargs) -> Any:
    '''Evaluates focal loss

    Args:
        pred : The predictions done by a model
        target : The target
    Kwargs:
        gamma (float): Gamma value
    Returns:
        (?) : The focal loss
    '''
    logpt = -logloss_with_na(pred, target, reduction="none")
    pt = torch.exp(logpt)
    loss = -((1 - pt) ** gamma) * logpt
    return loss.mean()


def categorical_crossentropy_from_logits(pred, target, **kwargs) -> Any:
    '''Applies categorical_crossentropy from logits.

    Args:
        pred : The predictions done by a model
        target : The target
    Returns:
        (?) : Categorical cross entropy
    '''
    # Transform target to indexes
    target_ind = torch.argmax(target, dim=1)
    # Apply CrossEntropyLoss
    loss = CrossEntropyLoss()(pred, target_ind)
    return loss


class LRScheduleWithWarmup(object):
    '''Implements linear schedule with warmup'''

    def __init__(self, warmup_steps: int = 5, total_steps: int = 30) -> None:
        '''Initialization of the class.

        Args:
            warmup_steps (int) : the number of warmup steps before lr decreases
            total_steps (int) : the number of steps before which the lr is equal to zero
        '''
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, epoch) -> float:
        # "epoch"s are instead steps if called at steps interval
        if epoch < self.warmup_steps:
            return float(epoch) / float(max(1, self.warmup_steps))
        else:
            return max(0.0, float(self.total_steps - epoch) / float(max(1, self.total_steps - self.warmup_steps)))


if __name__ == '__main__':
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")

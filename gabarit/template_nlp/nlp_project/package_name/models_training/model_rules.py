#!/usr/bin/env python3

## Generic rules model for text classification
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
# - ModelRules -> Generic rules model


# Get logger
import re
import logging
import numpy as np
import pandas as pd
from functools import partial
from typing import Union, Tuple, List, Any

from {{package_name}} import utils
from {{package_name}}.models_training.model_class import ModelClass


logger = logging.getLogger(__name__)


class ModelRules(ModelClass):
    '''Generic rules model for text classification
    The rules (contained in the table 'rules') each have three attributes:
    name (facultative, given on the fly if not explicit), pattern (a regex)
    and cls, the returned class if the pattern is found.
    '''

    _default_name = 'model_rules'

    def __init__(self, default: Union[str, None] = None, **kwargs) -> None:
        '''Initialization of the class (see ModelClass for more arguments)

        Kwargs:
            default (str) : The default prediction if no match
        '''
        self.default = default
        self.rules: List[dict] = []
        super().__init__(**kwargs)

    def create_rule(self, name: Union[str, None] = None, pattern: str = "*", cls: str = "classe_1") -> None:
        '''Creates a rule and appends it to the set of rules.

        Kwargs:
            name (str) : The name of the rule
            pattern (str) : The regex used to match
            cls (str) : The class to give if the pattern matches

        '''
        self.rules.append({'name': f'rule_{len(self.rules)}' if name is None else name,
                           'pattern': pattern,
                           'cls': cls})

    def validate_rule(self, rule: dict) -> Tuple[bool, Union[dict, None]]:
        '''Validates if a rule is ok or not : attributes present and with the correct type. Creates a name on the fly if needed.

        Args:
            rule (dict) : The rule to validate
        '''
        k = rule.keys()
        ret = True
        if 'name' in k:
            if not isinstance(rule['name'], str):
                logger.warning('For a rule, \'name\' must be of type str, automatic renaming')
                rule['name'] = 'rule_' + str(len(self.rules))
        else:
            rule['name'] = 'rule_' + str(len(self.rules))

        if 'pattern' not in k or not isinstance(rule['pattern'], str):
            logger.warning('For a rule, \'pattern\' must be of type str and present')
            ret = False

        if 'cls' not in k or not isinstance(type(rule['cls']), str):
            logger.warning('For a rule, \'cls\' must be of type str and present')
            ret = False
        if ret is False:
            rule = None

        return ret, rule

    def add_rules(self, rules: Union[List[dict], dict]) -> None:
        ''' Adds a set of rules (list) or a pre-formated rule (dict) to the rules already present

        Args:
            rules (list<dict> or dict) : the rules to add
        '''
        if isinstance(rules, dict):
            rules = [rules]

        for rule in rules:
            ok, rule = self.validate_rule(rule)
            if ok:
                self.rules.append(rule)

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, **kwargs) -> None:
        '''"Trains" the model by compiling the regex'''
        # If not multi-labels, transform y_train as dummies (should already be the case for multi-labels)
        if not self.multi_label:
            # If len(array.shape)==2, we flatten the array if the second dimension is useless
            if isinstance(y_train, np.ndarray) and len(y_train.shape) == 2 and y_train.shape[1] == 1:
                y_train = np.ravel(y_train)
            if isinstance(y_valid, np.ndarray) and len(y_valid.shape) == 2 and y_valid.shape[1] == 1:
                y_valid = np.ravel(y_valid)
            # Transformation dummies
            y_train_dummies = pd.get_dummies(y_train)
            self.list_classes = list(y_train_dummies.columns)
        # Else keep it as it
        else:
            y_train_dummies = y_train
            if hasattr(y_train_dummies, 'columns'):
                self.list_classes = list(y_train_dummies.columns)
            else:
                logger.warning("Can't read the name of the columns of y_train -> inverse transformation won't be possible")
                # We still create a list of classes in order to be compatible with other functions
                self.list_classes = [str(_) for _ in range(pd.DataFrame(y_train_dummies).shape[1])]

        # Set dict_classes based on list classes
        self.dict_classes = {i: col for i, col in enumerate(self.list_classes)}
        self.index_classes = {col: i for i, col in enumerate(self.list_classes)}
        for rule in self.rules:
            rule['regexp'] = re.compile(rule['pattern'])

        # Set trained
        self.trained = True
        self.nb_fit += 1

    def apply_rules(self, text: str, cls=None) -> Union[int, str, None]:
        '''Applies the set of rules on a text. Gives the label of the class if cls is not None otherwise
        gives 1 (used in the multi-labels case)

        Args:
            text (str) : The text on which to apply the rules
            cls (str) : If not None, the function gives the label of the class
        Return:
            The self.default, the label of the class, 0 or 1
        '''
        match = None
        for rule in self.rules:
            if rule["regexp"].search(text):
                if cls is not None:
                    match = rule["cls"]
                    break
                else:
                    match = 1 if rule["cls"] is None else 0

        if match is None:
            match = self.default
        return match

    @utils.data_agnostic_str_to_list
    def predict(self, x_test: Union[List[str], pd.Series]) -> Any:
        ''''Predicts' on the test dataset by applying the set of rules, either all at the same time either by class
        in the multi-labels/multi-classes (TODO : the distinction does not exist for now)

        Args:
            x_test (list<str> or pd.series) : The list of text to apply the rules to
        Return:
            (?) : The predictions
        '''
        x_test = pd.Series(x_test)
        if self.multi_label:
            preds = pd.DataFrame()
            for i in self.list_classes:
                preds[i] = partial(self.apply_rules, cls=i)
        else:
            preds = x_test.apply(self.apply_rules)

        return preds


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")

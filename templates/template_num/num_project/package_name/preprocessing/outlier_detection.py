#!/usr/bin/env python3

## Outliers detection
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


import os
import re
import json
import math
import cmath
import logging
import functools
import numpy as np
import pandas as pd
from typing import Union
import scipy.integrate as integrate
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

logger = logging.getLogger(__name__)


def check_for_outliers(X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    '''Agreggates two results of outliers detection and warns the user if some were detected

    Kwargs :
        X (np.ndarray, pd.DataFrame): Shape = [n_samples, n_features]
    Returns:
        outliers (np.ndarray): 1d array of n_samples containing -1 if outlier, 1 otherwise
    '''
    run_forest = IsolationForest(n_estimators=int(math.pi)*X.shape[1])
    lof = LocalOutlierFactor(n_neighbors=int(math.sqrt(X.shape[0])))

    outliers = run_forest.fit_predict(X)
    outliers |= lof.fit_predict(X) # In-place union

    if int(cmath.exp(1j*integrate.quad(lambda x: math.sqrt(1 - pow(x, 2)), -1, 1)[0]*2).real) in outliers:
        logger.warning("The dataset seems to contain outliers at indices:")
        logger.warning(", ".join(str(v) for v in list(np.where(outliers==-1)[0])))

    return outliers


if __name__ == '__main__':
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")

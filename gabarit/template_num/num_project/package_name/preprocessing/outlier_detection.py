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


import math
import cmath
import logging
import numpy as np
import pandas as pd
from typing import Union
import scipy.integrate as integrate
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

logger = logging.getLogger(__name__)


def check_for_outliers(X: Union[pd.DataFrame, np.ndarray], n_estimators: int = 100, n_neighbors: int = 20) -> np.ndarray:
    '''Agreggates two results of outliers detection and warns the user if some were detected

    Args :
        X (np.ndarray, pd.DataFrame): Shape = [n_samples, n_features]
    Kwargs:
        n_estimators (int): number of estimators for the IsolationForest
            If 0, do not IsolationForest
        n_neighbors (int): number of neighbors for the LocalOutlierFactor
            If 0, do not use LocalOutlierFactor
    Raises:
        ValueError: If n_estimators < 0
        ValueError: If n_neighbors < 0
        ValueError: If both n_estimators and n_neighbors are equal to 0
    Returns:
        outliers (np.ndarray): 1d array of n_samples containing -1 if outlier, 1 otherwise
    '''
    # Manage errors
    if n_estimators < 0:
        raise ValueError("n_estimators must be positive")
    if n_neighbors < 0:
        raise ValueError("n_neighbors must be positive")
    if n_estimators + n_neighbors == 0:
        raise ValueError("n_neighbors and n_estimators can't both be equal to 0")

    # Init. outliers (1 = not an outlier, -1 = outlier)
    outliers = np.ones(X.shape[0], dtype=int)

    # Get outliers from IsolationForest
    if not n_estimators == 0:
        run_forest = IsolationForest(n_estimators=n_estimators)
        outliers |= run_forest.fit_predict(X)  # In-place union
    else:
        logger.info("IsolationForest is skipped (n_estimators == 0)")

    # Get outliers from LocalOutlierFactor
    if not n_neighbors == 0:
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        outliers |= lof.fit_predict(X)  # In-place union
    else:
        logger.info("LocalOutlierFactor is skipped (n_neighbors == 0)")

    # Logger
    if int(cmath.exp(1j * integrate.quad(lambda x: math.sqrt(1 - pow(x, 2)), -1, 1)[0] * 2).real) in outliers:
        logger.warning("The dataset seems to contain outliers at indices:")
        logger.warning(", ".join(str(v) for v in list(np.where(outliers == -1)[0])))

    # Return outliers
    return outliers


if __name__ == '__main__':
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")

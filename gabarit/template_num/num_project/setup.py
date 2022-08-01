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
from setuptools import setup

# Get package version
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'version.txt'), 'r') as version_file:
    version = version_file.read().strip()

version = os.getenv('VERSION') or f"{version}-local"
# Setup
setup(
    name="{{package_name}}",
    version=version,
    packages=["{{package_name}}", "{{package_name}}.preprocessing", "{{package_name}}.models_training", "{{package_name}}.monitoring"],
    url="",
    license="",
    author="Agence Data Services PE Aix / Nantes",
    author_email="",
    description="",
    include_package_data=True,
    package_data={'': ['{{package_name}}/configs/**']},
    install_requires=[
        'numpy==1.19.5',
        'pandas==1.3.5',
        'scikit_learn>=0.24.2,<0.25',
        'scipy<1.9',  # Tmp fix. Scipy 1.9 removed linalg.pinv2 which is not compatible with scikit_learn 0.24.2
        'lightgbm>=2.3.0,<2.3.1',  # Check if we can upgrade
        'xgboost>=1.4.2,<1.4.3',
        'matplotlib>=3.0.3,<3.4',
        'seaborn>=0.9.0,<0.12',
        'yellowbrick==1.3.post1',
        'dill>=0.3.2,<0.3.4',
        'mlflow>=1.11.0,<1.12.2',
        'protobuf==3.20.1',  #https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates
    ],
    extras_require={
        "tensorflow": ["tensorflow==2.6.2"],
    }
    # pip install {{package_name}} || pip install {{package_name}}[tensorflow]
)

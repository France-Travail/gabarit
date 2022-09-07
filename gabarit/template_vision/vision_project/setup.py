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
        'pandas>=1.3,<1.4; python_version < "3.8"',
        'pandas>=1.3,<1.5; python_version >= "3.8"',
        'numpy>=1.19,<1.22; python_version < "3.8"',
        'numpy>=1.19,<1.24; python_version >= "3.8"',
        'scikit_learn>=1.0.0,<1.2',
        'matplotlib>=3.0.3,<3.6',
        'seaborn>=0.9.0,<0.13',
        'opencv-python-headless==4.5.5.62',
        'dill>=0.3.2,<0.3.6',
        'protobuf==3.20.1',  # https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates
        'mlflow>=1.11,<1.29',
        'tensorflow==2.6.2',
        'pycocotools==2.0.4',
        'tqdm>=4.40,<4.65',
    ],
    extras_require={
        "detectron": ["torch==1.8.1+cpu", "detectron2==0.6+cpu", "torchvision==0.9.1+cpu"],  # If GPU with cuda 11.1 : replace +cpu by +cu111
    }
    # pip install {{package_name}} || pip install {{package_name}}[detectron]
)

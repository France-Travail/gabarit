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

# Get package directory
package_directory = os.path.dirname(os.path.abspath(__file__))

# Get package version (env variable or verion file + -local)
version_path = os.path.join(package_directory, 'version.txt')
with open(version_path, 'r') as version_file:
    version = version_file.read().strip()
version = os.getenv('VERSION') or f"{version}-local"

# Get package description
readme_path = os.path.join(package_directory, 'README.md')
with open(readme_path, 'r') as readme_file:
    long_description = readme_file.read()

# Setup
setup(
    name='gabarit',
    version=version,
    packages=['gabarit', 'gabarit.template_nlp', 'gabarit.template_num', 'gabarit.template_vision'],
    license='AGPL-3.0',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Agence Data Services PE Nantes',
    description="Kickstart your AI project as code",
    url="https://github.com/OSS-Pole-Emploi/AI_frameworks",
    platforms=['windows', 'linux'],
    include_package_data=True,
    install_requires=[
        'Jinja2==3.0.3',
        'mypy==0.910',
    ],
    entry_points={
        'console_scripts': [
            'generate_nlp_project = gabarit.template_nlp.generate_nlp_project:main',
            'generate_num_project = gabarit.template_num.generate_num_project:main',
            'generate_vision_project = gabarit.template_vision.generate_vision_project:main'
        ],
    }
)

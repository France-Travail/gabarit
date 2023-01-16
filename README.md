[![pypi badge](https://img.shields.io/pypi/v/gabarit.svg)](https://pypi.python.org/pypi/gabarit)
![NLP tests](https://github.com/OSS-Pole-Emploi/gabarit/actions/workflows/nlp_build_tests.yaml/badge.svg)
![NUM tests](https://github.com/OSS-Pole-Emploi/gabarit/actions/workflows/num_build_tests.yaml/badge.svg)
![VISION tests](https://github.com/OSS-Pole-Emploi/gabarit/actions/workflows/vision_build_tests.yaml/badge.svg)
![API tests](https://github.com/OSS-Pole-Emploi/gabarit/actions/workflows/api_build_tests.yaml/badge.svg)
![NLP wheel](https://github.com/OSS-Pole-Emploi/gabarit/actions/workflows/nlp_wheel.yaml/badge.svg)
![NUM wheel](https://github.com/OSS-Pole-Emploi/gabarit/actions/workflows/num_wheel.yaml/badge.svg)
![VISION wheel](https://github.com/OSS-Pole-Emploi/gabarit/actions/workflows/vision_wheel.yaml/badge.svg)
![Documentation](https://github.com/OSS-Pole-Emploi/gabarit/actions/workflows/docs.yaml/badge.svg)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Generic badge](https://img.shields.io/badge/python-3.7|3.8-blue.svg)](https://shields.io/)


# Gabarit - Templates Data Science

Gabarit provides you with a set of python templates (a.k.a. frameworks) for your Data Science projects. It allows you to generate a code base that includes many features to speed up the production and testing of your AI models. You just have to focus on the core of Data Science.

![Animated gif](https://qgahwbtxikhdqmorproz.supabase.co/storage/v1/object/public/public/gabarit_right_optim.gif)

## Table of Content <!-- omit in toc -->
1. [Licence](#licence)
2. [Frameworks](#frameworks)
3. [Prerequisites](#prerequisites)
4. [Usage](#usage)
	1. [Installation](#installation)
	2. [Kickstart a new project](#kickstart-a-new-project)
	3. [Setup the new project](#setup-the-new-project)
5. [Security warning](#security-warning)
6. [Ethics](#ethics)
7. [Contacts](#contacts)

**📚 Documentation :** [https://oss-pole-emploi.github.io/gabarit](https://oss-pole-emploi.github.io/gabarit) <!-- omit in toc -->

---

## Licence

This project is distributed under the GNU AFFERO GENERAL PUBLIC LICENSE V3.0. Please check the LICENSE file.

---

## Frameworks

As a team, we strive to help Data Scientists across the board (and ourselves!) build awesome IA projects by speeding up the development process. This repository contains several frameworks allowing any data scientist, IA enthousiast (or developper of any kind, really) to kickstart an IA project from scratch.  

We hate it when a project is left in the infamous POC shadow valley where nice ideas and clever models are forgotten, thus we tried to pack as much production-ready features as we could in these frameworks.  

As Hadley Wickhman would say: "you can't do data science in a GUI". We are strong believers that during a data science or IA project, you need to be able to fine tune every nooks and crannies to make the best out of your data.  

Therefore, these frameworks act as project templates that you can use to generate a code base from nothing (except for a project name). Doing so would allow your fresh and exciting new project to begin with loads of features on which you wouldn't want to focus this early :
- Built-in models: from the ever useful TF/IDF + SVM to the more recent transformers
- Model-agnostic save/load/reload : perfect to embed your model behind a web service
- Generic training/predict scripts to work with your data as soon as possible
- DVC & MLFlow integration (you have to configure it to point to your own infrastructures)
- Streamlit demo tool
- ... and so much more !

Three IA Frameworks are available:

- **NLP** to tackle classification use cases on textual data

	-	Relies on the Words'n fun module for the preprocessing requirements

	-   Supports :

		- Mono Class / Mono Label classification

		- Multi Classes / Mono Label classification

		- Mono Class / Multi Labels classification


- **Numeric** to tackle classification and regression use cases on numerical data

	- Supports :

		- Regression
		- Multi Classes / Mono Label classification
		- Mono Class / Multi Labels classification


- **Computer Vision** to tackle classification use cases on images

	- Supports

		- Mono Class / Mono Label classification
		- Multi Classes / Mono Label classification
		- Area of interest detection

In addition we provide an [**API Framework**](/gabarit/template_api) that can be used to expose a gabarit model or one of your own.

## Prerequisites

To use these frameworks, you should already have python >= 3.7 installed. Note that this project started in python 3.7 but is now tested with python 3.8.
Obviously any prior knowledge of the holy trinity of python ML modules (pandas, sklearn, numpy) alongside Deep Learning frameworks (torch & tensorflow/keras) would be incredibly useful.


## Usage


### Installation

We packaged this project such that it can be directly installed from PyPI : `pip install gabarit` .  
However, it is not really necessary as this just intalls Jinja2==3.0.3 and adds some entry points. Basically you can manually install Jinja2 `pip install Jinja2==3.0.3` and you'll be able to generate new projects by calling the `generate_XXX_project.py` files individually.  

In the following we consider that you installed the project through pip, which enables entry points. Each entry point refers to a corresponding package generation file.


### Kickstart a new project

Each individual framework has a `generate_XXX_project` entry point that creates a new project code base.
They take several parameters as input :

- **`-n`** or **`--name`** : Name of the package/project (lowercase, no whitespace)
- **`-p`** or **`--path`** : Path (Absolute or relative) where to create the main directory of the project
- **`-c`** or **`--config`** : Path (Absolute or relative) to a .ini configuration file.  
	An default configuration file is given alongside each project. (`default_config.ini`).
	It usually contains stuff like default encoding, default separator for .csv files, pip proxy settings, etc.
- **`--upload`** or **`--upload_intructions`** : Path (Absolute or relative) to a file that contains a list of instructions to upload a trained model to your favorite storage solution.
- **`--dvc`** or **`--dvc_config`** : Path (Absolute or relative) to a DVC configuration file. If not provided, DVC won't be used.


Example : `generate_nlp_project -n my_awesome_package -p my_new_project_path -c my_configuration.ini --upload my_instructions.md --dvc dvc_config`


### Setup the new project

- (Optionnal) We strongly advise to create a python virtual env

	- `pip install virtualenv`

	- `python -m venv my_awesome_venv`

	- `cd my_awesome_venv/Scripts/ && activate` (windows) or `source my_awesome_venv/bin/activate` (linux)

- Requirements : `pip install --no-cache-dir -r requirements.txt`

- Setup the project (in develop mode) : `python setup.py develop`


If the `make` tool is available, you can use the features provided in `Makefile`:

- `create-virtualenv`
and
- `init-local-env`



## Security warning
Gabarit relies on a number of open source packages and therefore may carry on their potential security vulnerabilities. Our philosophy is to be as transparent as possible, which is why we are actively monitoring the dependabot analysis. In order to limit these vulnerabilities, we are in the regular process of upgrading these packages as soon as we can.
Notice that some packages (namely torch and tensorflow) might lag a few versions behind the actual up to date version due to compatibility issues with CUDA and our own infrastructure.

However, we remind you to be vigilant about the security vulnerabilities of the code and models that you will produce with these frameworks. It is your responsibility to ensure that the final product matches the security standards of your organization.

## Ethics
Pôle emploi intends to include the development and use of artificial intelligence algorithms and solutions in a sustainable and ethical approach. As such, Pôle emploi has adopted an ethical charter, resulting from collaborative and consultative work. The objective is to guarantee a framework of trust, respectful of the values of Pôle emploi, and to minimize the risks associated with the deployment of these technologies.

The pdf file is located in [pole-emploi.org](https://www.pole-emploi.org/accueil/communiques/pole-emploi-se-dote-dune-charte-pour-une-utilisation-ethique-de-lintelligence-artificielle.html?type=article) :

[PDF - Ethics charter - Pôle emploi](https://www.pole-emploi.org/files/live/sites/peorg/files/images/Communiqu%c3%a9%20de%20presse/Charte%20de%20p%c3%b4le%20emploi%20pour%20une%20Intelligence%20Artificielle%20%c3%a9....pdf)

## Contacts

If you have any question/enquiry feel free to drop us an email : contactadsaiframeworks.00619@pole-emploi.fr

- Alexandre GAREL - Data Scientist
- Nicolas GREFFARD - Data Scientist
- Gautier SOLARD - Data Scientist
- Nicolas TOUZOT - Product Owner

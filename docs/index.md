# Gabarit - Templates Data Science

Gabarit provides you with a set of python templates (a.k.a. frameworks) for your Data Science projects. It allows you to generate a code base that includes many features to speed up the production and testing of your AI models. You just have to focus on the core of Data Science.

## Philosophy

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

## Frameworks

Gabarit contains the following frameworks :

### [**NLP**](/frameworks/NLP) 
*to tackle classification use cases on textual data*

  -	Relies on the Words'n fun module for the preprocessing requirements
  - Supports :
      - Mono Class / Mono Label classification
      - Multi Classes / Mono Label classification
      - Mono Class / Multi Labels classification

### [**Numeric**](/frameworks/NUM) 
*to tackle classification and regression use cases on numerical data*

  - Supports :
    - Regression
    - Multi Classes / Mono Label classification
    - Mono Class / Multi Labels classification

### [**Computer Vision**](/frameworks/VISION) 
*to tackle classification use cases on images*

  - Supports
    - Mono Class / Mono Label classification
    - Multi Classes / Mono Label classification
    - Area of interest detection

### [**API**](/frameworks/API) 
*for exposing your model to the world*

  - Supports
    - Gabarit model created with one of the previous package
    - Any model of your own
  - Provides
    - A [FastAPI](https://fastapi.tiangolo.com/) to expose your model

These frameworks have been developped to manage different subjects but share a common structure and a common philosophy. Once a project made using a framework is in production, any other project can be sent into production following the same process.
Along with these frameworks, an API template has been developped and should soon be open sourced as well. With it, you can expose framework made models in no time !

## Getting started

### Installation
Gabarit supports python >= 3.7. To install it, run the command : 

```bash
pip install gabarit
```

This will install `gabarit` package and all frameworks.

### Kickstart a new project
To create a new project from a template, use gabarit entry points : 

- `generate_nlp_project`
- `generate_num_project`
- `generate_vision_project`
- `generate_api_project`

Example : `generate_nlp_project -n my_awesome_package -p my_new_project_path -c my_configuration.ini --upload my_instructions.md --dvc dvc_config`

They take several parameters as input :

- **`-n `** or **`--name `** : Name of the package/project (lowercase, no whitespace)
- **`-p `** or **`--path `** : Path (Absolute or relative) where to create the main directory of the project
- **`-c `** or **`--config `** : Path (Absolute or relative) to a .ini configuration file.  
	An default configuration file is given alongside each project. (`default_config.ini`).
	It usually contains stuff like default encoding, default separator for .csv files, pip proxy settings, etc.
- **`--upload `** or **`--upload_intructions `** : Path (Absolute or relative) to a file that contains a list of instructions to upload a trained model to your favorite storage solution.
- **`--dvc `** or **`--dvc_config `** : Path (Absolute or relative) to a DVC configuration file. If not provided, DVC won't be used.

### Setup your new project

- (Optionnal) We strongly advise to create a python virtual env

	- `pip install virtualenv`
	- `python -m venv my_awesome_venv`
	- `cd my_awesome_venv/Scripts/ && activate` (windows) or `source my_awesome_venv/bin/activate` (linux)

- Requirements : `pip install --no-cache-dir -r requirements.txt`

- Setup the project (in develop mode) : `python setup.py develop`


If the `make` tool is available, you can use the features provided in `Makefile`:

- `create-virtualenv`
- `init-local-env`

## Security warning
Gabarit relies on a number of open source packages and therefore may carry on their potential security vulnerabilities. Our philosophy is to be as transparent as possible, which is why we are actively monitoring the dependabot analysis. In order to limit these vulnerabilities, we are in the regular process of upgrading these packages as soon as we can.
Notice that some packages (namely torch and tensorflow) might lag a few versions behind the actual up to date version due to compatibility issues with CUDA and our own infrastructure.

However, we remind you to be vigilant about the security vulnerabilities of the code and model that you will produce with these frameworks. It is your responsibility to ensure that the final product matches the security standards of your organization.

## Ethics
Pôle emploi intends to include the development and use of artificial intelligence algorithms and solutions in a sustainable and ethical approach. As such, Pôle emploi has adopted an ethical charter, resulting from collaborative and consultative work. The objective is to guarantee a framework of trust, respectful of the values of Pôle emploi, and to minimize the risks associated with the deployment of these technologies.

The pdf file is located in [pole-emploi.org](https://www.pole-emploi.org/accueil/communiques/pole-emploi-se-dote-dune-charte-pour-une-utilisation-ethique-de-lintelligence-artificielle.html?type=article) :

[PDF - Ethics charter - Pôle emploi](https://www.pole-emploi.org/files/live/sites/peorg/files/images/Communiqu%c3%a9%20de%20presse/Charte%20de%20p%c3%b4le%20emploi%20pour%20une%20Intelligence%20Artificielle%20%c3%a9....pdf)

## Contacts

If you have any question/enquiry feel free to drop us a mail : contactadsaiframeworks.00619@pole-emploi.fr

- Alexandre GAREL - Data Scientist
- Nicolas GREFFARD - Data Scientist
- Gautier SOLARD - Data Scientist
- Nicolas TOUZOT - Product Owner
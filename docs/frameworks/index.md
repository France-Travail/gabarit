## General principles

### Templates structure

- Data must be kept in a directory named `project_name-data/` located at the root folder of the project (i.e. where `setup.py` is).

- Any data mapping or lookup can be kept under `project_name-data/sources/`. Warning : we're talking small files (< 1 Mo). Larger files should be managed through DVC (or git lfs for that matter).

- Embedding files or equivalent should also be kept under `project_name-data/`.

- Transformers models (e.g. Flaubert) should be kept under `project_name-transformers/` at the root directory of the project.

- Trained models that you built and trained are automatically saved under `project_name-models/`.

- Sklearn preprocessing pipelines (mainly from the numerical framework) are automatically stored within `project_name-pipelines/`.

- The Computer Vision template has some more subdirectories in the `project_name-data/` folder:

    - `cache_keras`: subfolder that replaces the default keras' cache folder. Used with transfer learning classifiers.

	- `transfer_learning_weights`: subfolder that holds networks weights to be used with custom Faster RCNN implementation.

	- `detectron2_conf_files`: subfolder that holds all necessary configuration files to be used with the detectron2 models.


- The `tests/` directory contains numerous unit tests allowing to automatically validate the intended behaviour of the different features. It is of utter importance to keep them up to date depending on your own developments to ensure that everything is working fine. Feel free to check already existing test files if you need some directions. Note that to launch a specific test case you just have to run : `python test_file.py`; for instance: `python tests/test_model_tfidf_dense.py`.

- Numbered files contained in `project_name-scripts/` (e.g. `2_training.py`) hint the main steps of the project. They are indicative but we strongly advise to use them as it can speed up the development steps. It orchestrates the main features of this project: utils functions, preprocessing pipelines and model classes.

- The `preprocess.py` file contains the different preprocessing pipeline available by default by the package/project. More specifically, it contains a dictionnary of the pipelines. It will be used to create working datasets (for instance training set, valid test and test set).

-	Beware that the first row of each generated csv file after running a preprocessing will contain the name of the preprocessing pipeline applied such that it can be reused in the future. Hence, this row (e.g. `#preprocess_P1`) is a metadata and **it has to be skipped** while parsing the csv file. Our templates provide a function (`utils.read_csv`) that does it automatically (it also returns the metadata).

- The modelling part is built as follow :

    - ModelClass : main class that manages how data / models are saved and how performance metrics are computed

    - ModelPipeline : inherits from ModelClass, manages sklearn pipeline models

    - ModelKeras : inherits from ModelClass, manages Keras/Tensorflow models

    - ModelPyTorch : inherits from ModelClass, manages PyTorch models

    - ModelXXX : built-in implementation of standard models used in the industry, inherits from one of the above classes when appropriate

### Main steps of a given project

The intended flow of a project driven by one of these framework is the following:

- 0 – Utility files

    - Split train/valid/test, sampling, embedding download, etc...


- 1 – Preprocessing

- 2 – Model training

    - You can tune the parameters within the script or update the model class depending on your needs

- 3 – Predictions on a dataset

- 4 – Play with a streamlit demonstrator to showcase your models

### Data formats

Input data are supposed to be `.csv` files and the separator and encoding are to be provided during the generation of the project. It is obviously possible to use another datatype but a transformation step to `.csv` will be required to use the scripts provided by default.

Concerning the prediction target, please refer to `2_training.py`. Usually we expect One Hot Encoded format for multi-labels use cases. For single-label use cases, a single column (string for classification, float for regression) is expected.

## Features
Projects generated through the frameworks provide several main features:

### Model saving and reloading

When a new model is instanciated, a directory is created within `project_name-models/`. It is named after the model type and its date of creation. Each model class exposes a `save` function that allow to save everything necessary to load it back:

- Configuration file
- Serialized object (.pkl)
- "standalone" model
- If Deep Learning : the network weights
- etc.

Thus any model can be loaded through the `utils_models.load_model` function. The "standalone" mode ensures that the model can be loaded even after its code has been modified. Indeed, the .pkl file could be out of sync with the model class (it it was modified after the model had been saved). In this specific case, you can use `0_reload_model.py`.


### Third party AI modules

To this day, 3 main AI modules are used:

- Scikit Learn

- TensorFlow (Keras)

- PyTorch (PyTorch Lightning)

Do no hesitate to extend this list as is the case for LighGBM for instance.


### DVC

A new project can automatically be set up to run in sync with [DVC](https://dvc.org) if you supply the necessary configuration during project generation. We strongly advise to use DVC or similar (git lfs could do the trick) to keep both your code and your datasets synchronized to be able to re-train a model in the same conditions sometime down the line. Please refrain to upload large datasets (>1mo) directly on your version control system. Once setup, dvc configuration is available within `.dvc/`


### MLFlow
A new project can automatically be set up to work alongside a [MLFlow](https://mlflow.org) instance. If you supply a MLFlow host url during project generation, training metrics will be automatically be send to your MLFlow server. Refer to `2_training.py` and `monitoring/model_logger.py` for further informations about this mechanism.


### Streamlit demonstrator
A generic demonstrator is automatically created when you generate a new project with the frameworks. It relies on [Streamlit](https://streamlit.io) to expose a handy front-end to showcase your work. The demonstrator script can be easily modified to fit your specific needs.
![Streamlit demo](/assets/images/streamlit.PNG){ loading=lazy }


### Exploratory Data Analysis (EDA)
Some frameworks provide a generic exploratory data analysis notebook to quickly grasp your datasets (`project_name-exploration/EDA/`). Feel free to have a go with it before starting heavy modelling work; EDA is an extraordinary opportunity to get to know your data which will greatly help you further down the line.

### Misc.

Some additionnal features :

- Basic hyper-parameter search is provided within `2_training.py`
- You can use Tensorflow checkpoints to restart the training of a model without having to start from scratch
- A custom made Learning Rate Scheduler for Tensorflow is also provided
- Etc... feel free to explore the generated classes to learn more about what you can do !


## Industrialization


### Principles

Industrialization of a project generated from one of the framework roughly follows the same pattern.
Once you have trained a model which is a release candidate :

- Push the actual serialized model to your artifact repository (for instance artifactory or nexus)

    * Instructions about how to technically push the model are usually specified within the model directory

- Push the python module (the project you generated with a framework) to your artifact repository (it could be pypi or any system able to host a python repository)

    * First you have to build a wheel of the project `.whl` : `python setup.py sdist bdist_wheel`

    * Then you have to push it to your repository, for instance by using [twine](https://pypi.org/project/twine/) : `twine upload --username {USER} --password {PWD} --repository-url https://{repository_url} dist/*.whl`

    * Note that we strongly advise to embed these steps within a Continuous Integration Pipeline and ensuring that all your unit tests are OK (you can use nose to run your test suite : `pip install nose nose-cov && nosetests tests/`)

    * Beware, function `utils_models.predict` has to be adapted to your project needs (e.g. if some specific computations are required before or after the actual inference).

        + This is the function that has to be called by the web service that will serve your model. Using `utils_models.predict` instead of the actual predict method of the model class ensure that your service can stay model agnostic: if one day you decide to change your design, to use another model; the service won't be impacted.

    * Warning: some libraries (such as torch, detectron2, etc.) may not be hosted on PyPI. You'll need to add an extra `--find-links` option to your pip installation.

    	+ If you don't have access to the internet, you'll need to setup a proxy which will host all the needed libraries. You can then use `--trusted-host` and `--index-url` options.

- You can use our API Framework to expose your model, see [API section](/frameworks/API)


### Update your model

If you want to update the model exposed by the API, you just have to push a new version of the serialized model to your repository and update your service (typically only the model version). If the actual code base of the model (for instance in the predict method) was updated, you would also have to publish a new version of the python module.  

### Unit tests

Numerous unit tests are provided by the framework. Don't forget to adapt them when you modify the code. If you wish to add features, it is obviously advised to add new unit tests.

## Misc.

- To this day, each framework is tested and integrated on our own continuous integration pipeline.
- If a GPU is available, some models will automatically try to use it during training and inference

### Update a project with the latest Gabarit version

It can be tricky to update a project to a newer version of Gabarit as you probably made changes into the code and don't want them to be removed.  
As our philosophy is to give you code and let you adapt it for your specific usage, we can't control everything.  

However, we still provide an operating procedure that must keep your changes while updating the project to the latest Gabarit version :

1. Create a new branch from your latest commit C0

2. Find the Gabarit's version last used to generate your project

3. Generate a project ON TOP of your code using this version

      * Commit the changes (commit C1)

4. Create a patch : `git diff HEAD XXXXXX > local_diff.patch` where XXXXXX is the SHA-1 of the latest commit C0

      * This patch holds every changes you made since you last generated the project, except for new files
      * Note that we don't really care for new files as they are not removed with Gabarit new generation

5. Generate a project ON TOP of your code, but this time with the latest Gabarit version. Commit the changes (commit C2).
   
      * The `.gitignore` file might change, be careful NOT TO COMMIT files that are "unignored".

6. Apply the patch : `git am -3 < local_diff.patch`

      1. **RENAMED / MOVED / DELETED FILE** : this won't work for renamed / moved / deleted files.

           * You'll have to manage them manually
           * You need to remove files that are no longer in the new Gabarit version BEFORE applying the patch.
           * The patch will then probably crash. You will have to fix it manually.

      2. You will probably have conflict, resolve them
      3. Add files and commit changes (commit C3)
      4. You might need to run `git am --skip` as we only had a single patch to apply

7. Squash the last commits (you should have 3 commits)

      * `git reset --soft HEAD~3`
      * `git commit -m "my_message"`

8.  CHECK IF EVERYTHING SEEMS OK

9.  Merge your branch & push :) 

Be aware that some of your defined functions might need to be updated as the newer Gabarit version might have some breaking changes.

![Visual update processus illustration](/assets/images/update.png)
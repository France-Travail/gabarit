# NUM Framework

## Project structure

Here is the structure of a project generated with `generate_num_project` command : 

```bash
.
├─ template_num                       # your application package
│    ├─ models_training               # global config and utilities
│    │    ├─ classifiers
│    │    │    ├─ models_sklearn      # package containing some predefined scikit-learn classifiers
│    │    │    └─ models_tensorflow   # package containing some predefined tensorflow classifiers
│    │    ├─ regressors      
│    │    │    ├─ models_sklearn      # package containing some predefined scikit-learn regressors
│    │    │    └─ models_tensorflow   # package containing some predefined tensorflow regressors
│    │    ├─ ...
│    │    ├─ model_class.py           # module containing base Model class
│    │    └─ utils_models.py          # module containing utility functions
│    │
│    ├─ monitoring                    # package containing monitoring utilities (mlflow, model explicability)
│    │
│    ├─ preprocessing                 # package containing preprocessing logic
│    │
│    ├─ __init__.py
│    └─ utils.py
│
├─ template_num-data                  # Folder where to store your data
├─ template_num-exploration           # Folder where to store your exploratory notebooks
├─ template_num-models                # Folder containing trained models
├─ template_num-pipelines             # Folder containing fitted pipelines are stored
├─ template_num-scripts               # Folder containing script for preprocessing, training, etc.
├─ template_num-tutorials             # Folder containing a tutorial notebook
.
.
.
├─ makefile
├─ setup.py
└─ README.md
```

## Numeric framewrok specificities

- Preprocessing has to be computed in a two step fashion to avoid bias:

  - Fit your transformations on the training data (`1_preprocess_data.py`)

  - Transform your validation/test sets (`2_apply_existing_pipeline.py`)

- Preprocessing pipelines are stored in the `project_name-pipelines` folder

  - They are then stored as a .pkl object in the model folders (so that these can be used during inference)

!!! warning

    If you used a custom preprocessing function `funcA` with `FunctionTransformer`, be aware that the pickled pipeline 
    may not return wanted results if you later modify `funcA` definition. 
    
    Please check [gabarit/issues/63](https://github.com/France-Travail/gabarit/issues/63)
# NLP Framework

## Project structure

Here is the structure of a project generated with `generate_nlp_project` command : 

```bash
.
├─ template_nlp                 # your application package
│    ├─ models_training         # global config and utilities
│    │    ├─ models_sklearn     # package containing some predefined scikit-learn models
│    │    ├─ models_tensorflow  # package containing some predefined tensorflow models
│    │    ├─ model_class.py     # module containing Model base class
│    │    ├─ ...
│    │    └─ utils_models.py    # module containing utility functions
│    │
│    ├─ monitoring              # package containing monitoring utilities (mlflow, model explicability)
│    │
│    ├─ preprocessing           # package containing preprocessing logic
│    │
│    ├─ __init__.py
│    └─ utils.py
│ 
├─ template_nlp-data            # Folder where to store your data
├─ template_nlp-exploration     # Folder where to store your exploratory notebooks
├─ template_nlp-models          # Folder containing trained models
├─ template_nlp-scripts         # Folder containing script for preprocessing, training, etc.
├─ template_nlp-tutorials       # Folder containing a tutorial notebook
.
.
.
├─ makefile
├─ setup.py
└─ README.md
```

!!! warning

    If you used a custom preprocessing function `funcA` with `FunctionTransformer`, be aware that the pickled pipeline 
    may not return wanted results if you later modify `funcA` definition. 
    
    Please check [gabarit/issues/63](https://github.com/France-Travail/gabarit/issues/63)
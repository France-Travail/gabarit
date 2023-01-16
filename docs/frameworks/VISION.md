# VISION Framework
## Project structure

Here is the structure of a project generated with `generate_vision_project` command : 

```bash
.
├─ template_vision              # your application package
│    ├─ models_training         # global config and utilities
│    │    └─ classifiers        # package containing some predefined classifiers
│    │    ├─ object_detectors   # package containing some predefined object detectors
│    │    ├─ ...
│    │    ├─ model_class.py     # module containing base Model class
│    │    └─ utils_models.py    # module containing utility functions
│    │
│    ├─ monitoring              # package containing monitoring utilities (mlflow, model explicability)
│    │
│    ├─ preprocessing           # package containing preprocessing logic
│    │
│    ├─ __init__.py
│    └─ utils.py
│
├─ template_vision-data         # Folder where to store your data
├─ template_vision-exploration  # Folder where to store your exploratory notebooks
├─ template_vision-models       # Folder containing trained models
├─ template_vision-scripts      # Folder containing script for preprocessing, training, etc.
├─ template_vision-tutorials    # Folder containing a tutorial notebook
.
.
.
├─ makefile
├─ setup.py
└─ README.md
```

## Computer vision framewrok specificities

- The expected input data format is different than in the other frameworks.

  - For image classification, 3 differents formats can be used :

    1. A root folder with a subfolder per class (containing all the images associated with this class)
    2. A unique folder containing every image where each image name is prefixed with its class
    3. A folder containing all the images and a .csv metadata file containing the image/class matching

  - For object detection, you must provide a .csv metadata file containing the bounding boxes for each image
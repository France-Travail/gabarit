# API Framework

## Project structure

Here is the structure of a project generated with `generate_api_project` command : 

```bash
.
├─ template_api                 # your application package
│   ├─ core                     # global config and utilities
│   │    ├─ __init__.py
│   │    ├─ config.py
│   │    ├─ event_handlers.py   # load your model to your app at startup
│   │    └─ logtools.py
│   │
│   ├─ model                    # model classes
│   │    ├─ __init__.py
│   │    ├─ model_base.py
│   │    └─ model_gabarit.py
│   │
│   ├─ routers                  # applications routes
│   │    ├─ schemas
│   │    ├─ __init__.py
│   │    ├─ functional.py
│   │    ├─ technical.py
│   │    └─ utils.py
│   │
│   ├─ __init__.py
│   └─ application.py
│
├─ tests
│   └─ ...
.
.
.
├─ .env                         # environement variables for settings
├─ makefile
├─ Dockerfile
├─ pyproject.toml               # your package dependencies and infos
├─ setup.py
├─ launch.sh                    # start your application
└─ README.md
```


## Quickstart

Gabarit has generated a `template_api` python package that contains all your
[FastAPI](https://fastapi.tiangolo.com/) application logic.

It contains three main sub-packages (cf. [project structure](#project-structure)) :

- `core` package for configuration and loading your model into your application

- `model` package for defining how to download your model, to load it and make predictions

- `routers` package for defining your API routes and how they work

Have a look at your `.env` file to see the default settings :
```bash
APP_NAME="template_api"
API_ENTRYPOINT="/template_api/rs/v1"
MODEL_PATH="template_api-models/model.pkl"
```

### Create a virtualenv and install your package

With make :
```bash
make run create-virtualenv
source .venv/bin/activate

make init-local-env
```

Without make :
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### Start your application

To start your FastAPI application activate your virtual environment and then use the
script `launch.sh` or the `run` command of the makefile :

```bash
chmod +x launch.sh
make run
```

This will start a FastAPI thanks to [uvicorn](https://www.uvicorn.org/) that listen
on port 5000. Visit [http://localhost:5000/docs](http://localhost:5000/docs) to see the automatic interactive API
documentation (provided by FastAPI and [Swagger UI](https://github.com/swagger-api/swagger-ui))

## How it works

### Model class

Your application use a `Model` object to make predictions. You will find a base `Model` class
in [`template_api.model.model_base`](/reference/template_api/model/model_base) :

```python
class Model:
    def __init__(self):
        self._model: Any = None
        self._model_conf: dict = None
        self._model_explainer = None
        self._loaded: bool = False

    def is_model_loaded(self) -> bool:
        """return the state of the model"""
        return self._loaded

    def loading(self, **kwargs):
        """load the model"""
        self._model, self._model_conf = self._load_model(**kwargs)
        self._loaded = True

    def predict(self, *args, **kwargs) -> Any:
        """Make a prediction thanks to the model"""
        return self._model.predict(*args, **kwargs)
    
    def explain_as_json(self, *args, **kwargs) -> Union[dict, list]:
        """Compute explanations about a prediction and return a JSON serializable object"""
        return self._model_explainer.explain_instance_as_json(*args, **kwargs)

    def explain_as_html(self, *args, **kwargs) -> str:
        """Compute explanations about a prediction and return an HTML report"""
        return self._model_explainer.explain_instance_as_html(*args, **kwargs)

    def _load_model(self, **kwargs) -> Tuple[Any, dict]:
        """Load a model from a file

        Returns:
            Tuple[Any, dict]: A tuple containing the model and a dict of metadata about it.
        """
        ...

    @staticmethod
    def download_model(**kwargs) -> bool:
        """You should implement a download method to automatically download your model"""
        ...
```

As you can see, a `Model` object has four main attributes :

- `_model` containing your gabarit, scikit-learn or whatever model object

- `_model_conf` which is a python dict with metadata about your model

- `_model_explainer` containing your model explainer

- `_loaded` which is set to `True` after `_load_model` has been called

The `Model` class also define a `download_model` method that will be used to download
your model. By default it does nothing and returns `True`.

You will also find a `ModelGabarit` class in `template_api.model.model_gabarit` that
is suited to a model constructed thanks to a Gabarit template.

It is a great example of how to adapt the base `Model` class to your use case.

### Load your model at startup

Your model is loaded into your application at startup thanks to
[`template_api.core.event_handlers`](/reference/template_api/core/event_handlers) :

```python
from typing import Callable

from fastapi import FastAPI
from ..model.model_base import Model

def _startup_model(app: FastAPI) -> None:
    """Create and Load model"""
    model = Model()
    model.loading()
    app.state.model = model

def start_app_handler(app: FastAPI) -> Callable:
    """Startup handler: invoke init actions"""

    def startup() -> None:
        logger.info("Startup Handler: Load model.")
        _startup_model(app)

    return startup
```

To change the model used by your application, change the model imported here.

### Functional and technical routers

Routers are split into two categories by default : technical and functional ones.

- Technical routers are used for technical purpose such as verifying liveness or getting
  infos about your application
- Functional ones are used to implement your business logic such as model predictions
  or model explicability

Since gabarit could not know what data your model is expecting, the default `/predict` route
from [`template_api.routers.functional`](/reference/template_api/routers/functional) use a starlette Request object
instead of pydantic.

For a cleaner way to handle requests and reponses you should use pydantic as
[stated in FastAPI documentation](https://fastapi.tiangolo.com/tutorial/body/#create-your-data-model)

You can use routes from template_api.routers.technical as examples of how to create
requests and responses schemas thanks to pydantic or have a look at the
[FastAPI documentation](https://fastapi.tiangolo.com/tutorial/response-model/).

### Dockerfile

A minimal `Dockerfile` is provided by the template. You should have a look a it, especially
if you have to download your model in your containers.
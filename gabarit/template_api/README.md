# API template

The API template is a very fast way to create a [FastAPI](https://fastapi.tiangolo.com/) 
powered application to expose your model, especially if you used Gabarit to generate your 
project.

It helps you create a high performance REST API ready for production. 

## Table of content <!-- omit from toc --> 
- [Quickstart](#quickstart)


## Quickstart
To generate a new API project, use the command line interface `generate_api_project` :

```bash
> generate_api_project --help
usage: generate_api_project [-h] -n NAME -p PATH [--gabarit_package GABARIT_PACKAGE] [--gabarit_package_version GABARIT_PACKAGE_VERSION]
                            [-c CUSTOM [CUSTOM ...]]

optional arguments:
  -h, --help            
                        show this help message and exit

  -n NAME, --name NAME  
                        Project name

  -p PATH, --path PATH  
                        Path (relative or absolute) to project directory

  --gabarit_package GABARIT_PACKAGE
                        Gabarit package you want to use

  --gabarit_package_version GABARIT_PACKAGE_VERSION
                        Gabarit package version you want to use

  -c CUSTOM [CUSTOM ...], --custom CUSTOM [CUSTOM ...]
                        Add custom templates such as a custom .env file or a 
                        custom Dockerfile. 
                        Example : 
                        --custom /my/.env.custom=.env /all/from/dir=
```

## Parameters
### `--name`
Name of your project

### `--path`
Path to your project

### `--gabarit_package`
Gabarit package that you want to use. This package will be added to your `pyproject.toml`
as a dependency of your project. It thus should be uploaded to pypi or a private repository
from where it can be installed.

```toml
[project]
dependencies = [
    "gabarit_package",
    # ...
]
```

### `--gabarit_package_version`
Gabarit package version to use. The version of `gabarit_package` will be fixed in your
`pyproject.toml`

```toml
[project]
dependencies = [
    "gabarit_package==gabarit_package_version",
    # ...
]
```

### `--custom`
This argument is used to add or overwrite template files from Gabarit API template.

You can for example provide a custom `.env` file to specify you custom settings : 

```bash
generate_api_project -n myproject -p path/to/myproject -c /path/to/.env.custom=.env
```

The custom provided template files should follow the pattern : 
`/path/to/custom/file=path/to/destination`

You can also provide a directory to recursively add all files in as custom templates.
For exemple, suppose you have a folder containing your custom templates : 

```bash
/path/to/my/custom/templates
│
├─ package_name
│   └─ core
│        └─ config.py
│
├─ custom_script.sh
└─ .env
```

Then by providing it to the CLI : 

```bash
generate_api_project -n myproject -p path/to/myproject -c /path/to/my/custom/templates=
```

You will overwrite `package_name/core/config.py`, `.env` and add `custom_script.sh` to
your generated project.

The custom templates you provide treated as [JINJA templates](https://jinja.palletsprojects.com/)
just as the gabarit templates so you can use all JINJA logic and all the variables used by
gabarit : 
- package_name
- gabarit_package
- gabarit_package_version

<!-- 
The "omit from toc" comments are here for the Markdown All in One VSCode extension :
it permits to remove a title from the auto table of content

See https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one
--> 
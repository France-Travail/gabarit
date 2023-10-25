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
usage: generate_api_project [-h] -n NAME -p PATH [--gabarit_package GABARIT_PACKAGE] [--gabarit_import_name GABARIT_IMPORT_NAME]
                            [-c CUSTOM [CUSTOM ...]]

optional arguments:
  -h, --help            show this help message and exit
  -n NAME, --name NAME  Project name
  -p PATH, --path PATH  Path (relative or absolute) to project directory
  --gabarit_package GABARIT_PACKAGE
                        Gabarit package you plan to use in your API (let empty otherwise). Example : my-gabarit-
                        project[explicability]==0.1.2
  --gabarit_import_name GABARIT_IMPORT_NAME
                        The import name of your Gabarit package might be different from the package name as with scikit-learn /
                        sklearn. Use this argument to specify a different import name
  -c CUSTOM [CUSTOM ...], --custom CUSTOM [CUSTOM ...]
                        Add custom templates such as a custom .env file or a custom Dockerfile. Example : --custom
                        /path/to/my/.env.custom=.env include/all/from/dir=
```

## Parameters
### `--name`
Name of your project

### `--path`
Path to your project

### `--gabarit_package`
Gabarit package you plan to use in your API. This package will be added to your 
[`pyproject.toml`](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html)
as a dependency of your project.

Specify the version of the package you need in your project. Otherwise an error will be raised
except you use the flag `--gabarit_no_spec`

See the [pip documentation about requirement specifiers](https://pip.pypa.io/en/stable/reference/requirement-specifiers/#examples)
for few requirement specification examples.

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
SHELL = /bin/sh
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

####################################################
# Initialization of the local dev environment
####################################################

NAME_VENV=venv_{{package_name}}

ifeq ($(shell which python), )
python_exec=/usr/bin/python3.8
else
python_exec=python
endif

.PHONY: help


help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
 
.DEFAULT_GOAL := help

create-virtualenv: ## Creation of a virtual environment
	@echo "Using $(python_exec)"
	pip install virtualenv;\
	$(python_exec) -m venv $(NAME_VENV)


init-local-env: ## Initialization of the local dev environment
ifndef VIRTUAL_ENV
	$(error Python virtual environment not activated !)
endif
	mkdir -p ./{{package_name}}-models
	mkdir -p ./{{package_name}}-pipelines
	mkdir -p ./{{package_name}}-data/sources
	python -m pip install{% if pip_trusted_host is not none %} --trusted-host {{pip_trusted_host}}{% endif %}{% if pip_index_url is not none %} --index-url {{pip_index_url}}{% endif %} --upgrade pip
	pip install{% if pip_trusted_host is not none %} --trusted-host {{pip_trusted_host}}{% endif %}{% if pip_index_url is not none %} --index-url {{pip_index_url}}{% endif %} -r requirements.txt;\
	python setup.py develop{% if dvc_config_ok is true %}
	@ bash -c "if command -v dvc &> /dev/null; then echo \"DVC install\" && dvc install; else echo \"----- [DVC] Can't find dvc command. Skipping  the install\"; fi;"
	cp ./{{package_name}}-ressources/git_hooks/post-checkout .git/hooks/post-checkout
	cp ./{{package_name}}-ressources/git_hooks/pre-commit .git/hooks/pre-commit
	cp ./{{package_name}}-ressources/git_hooks/pre-push .git/hooks/pre-push{% endif %}

####################################################
# Tests
####################################################

test: ## Launch python tests
ifndef VIRTUAL_ENV
	$(error Python virtual environment not activated !)
endif
	nosetests -c nose_setup_coverage.cfg tests --exe # https://stackoverflow.com/questions/1457104/nose-unable-to-find-tests-in-ubuntu

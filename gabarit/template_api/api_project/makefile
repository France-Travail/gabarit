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

.PHONY: help

help: ## This help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help

####################################################
# Local virtualenv
####################################################

NAME_VENV=.venv

create-virtualenv:
	python -m venv .venv

init-local-env:
ifndef VIRTUAL_ENV
	$(error Please activate your virutalenv : source .venv/bin/activate)
endif
	pip install -e .[dev]

####################################################
# Tests
####################################################

test: ## Launch python tests
ifndef VIRTUAL_ENV
	$(error Please activate your virutalenv : source .venv/bin/activate)
endif
	pytest --cov-report term-missing

####################################################
# Code quality
####################################################

quality: black isort

black: ## Formatter
ifndef VIRTUAL_ENV
	$(error Please activate your virutalenv : source .venv/bin/activate)
endif
	@echo 'Black formatting'
	black {{package_name}}

isort: ## Import formatter
ifndef VIRTUAL_ENV
	$(error Please activate your virutalenv : source .venv/bin/activate)
endif
	@echo 'Isort formatting'
	isort {{package_name}}

####################################################
# Run application localy
####################################################

run:
ifeq ($(OS),Windows_NT)
	$(error launch.sh is meant to be used on linux)
endif
ifndef VIRTUAL_ENV
	$(error Please activate your virutalenv : source .venv/bin/activate)
endif
	./launch.sh

####################################################
# Build docker image
####################################################

build-d:
	docker build -t {{package_name}}_api -f Dockerfile.svc .

run-d:
	docker container run --rm -p 5000:5000 {{package_name}}_api

run-d-it:
	docker container run --rm -p 5000:5000 -i -t {{package_name}}_api

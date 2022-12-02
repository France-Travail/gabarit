#!/bin/bash
#
export option="$1"

download-model
uvicorn --host 0.0.0.0 --port 5000 {{package_name}}.application:app ${option}

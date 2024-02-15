#!/usr/bin/env sh

export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libmimalloc.so.2"
export LD_BIND_NOW=1

: "${HOST:=0.0.0.0}"
: "${PORT:=5000}"
: "${WORKERS:=1}"

download-model

uvicorn {{package_name}}.application:app \
  --log-config log_conf.json \
  --host $HOST \
  --port $PORT \
  --workers $WORKERS

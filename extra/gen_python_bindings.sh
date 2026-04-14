#!/bin/bash

set -Eeuo pipefail
set -x

cp -f extra/ds-service.proto src/slurm_workflows

cd src

python -m grpc_tools.protoc \
    --proto_path=. \
    --python_out=. \
    --pyi_out=. \
    --grpc_python_out=. \
    slurm_workflows/ds-service.proto



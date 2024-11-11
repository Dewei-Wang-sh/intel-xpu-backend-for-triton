#!/bin/bash

set -vxe

script_dir=$(dirname "$0")
source "$script_dir/run_util.sh"

export PATH="$HOME/miniforge3/bin:$PATH"

python -m venv ./.venv; source ./.venv/bin/activate
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/triton/lib:$LD_LIBRARY_PATH:$VIRTUAL_ENV/lib
export CPATH=$CPATH:$VIRTUAL_ENV/include:$VIRTUAL_ENV/include/sycl

conda run --no-capture-output -n triton bash "$script_name"

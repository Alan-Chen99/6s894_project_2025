#!/usr/bin/bash

set -eux

cd "$(dirname "$0")"

# currently uv wont notice the change unless setup.py is changed
# its likely not configured correctly but we just touch the file for now
touch csrc/setup.py

uv sync --verbose

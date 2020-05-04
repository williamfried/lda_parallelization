#!/bin/bash

# A quick few lines to get a spark VM up and running.
sudo python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# This won't persist outside of the session that this
# script runs within.
# TODO: Need to get a way around this.
export PYSPARK_PYTHON=python3

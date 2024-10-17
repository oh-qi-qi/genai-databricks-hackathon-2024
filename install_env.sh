#!/bin/bash

# Install the required environment using conda
conda env create -f /Volumes/llm_workspace/default/regubim-ai-volume/databricks_base_environment.yml --force

# Activate the environment
source activate databricks_base_environment

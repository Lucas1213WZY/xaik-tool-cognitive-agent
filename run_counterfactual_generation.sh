#!/bin/bash
# Wrapper script to run generate_counterfactual_trials.py with conda environment

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate rlnb_ibl_env

# Run the Python script with all arguments passed through
python3 generate_counterfactual_trials.py "$@"

#!/bin/bash
# Wrapper script to run counterfactual generation with conda Python

cd /Users/wangzhuoyulucas/Documents/GitHub/xaik-tool-cognitive-agent

# Use conda environment's Python directly
CONDA_PYTHON="/Users/wangzhuoyulucas/anaconda3/envs/rlnb_ibl_env/bin/python"

# Run the generation script
"$CONDA_PYTHON" generate_counterfactual_trials.py \
  --input-csv code_for_papers/old/coxam/rl_fit_trials.csv \
  --param-csv src/user_simulation/param_config/CoXAM_counterfactual_simulation_cog_param.csv \
  --dt-weights code_for_papers/old/coxam/model_counterfactual/best_model.zip \
  --output-csv code_for_papers/old/coxam/rl_fit_trials_ct_output_final.csv

echo "✓ Generation complete!"

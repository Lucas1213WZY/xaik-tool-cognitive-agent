import os
import pandas as pd
import random
from skopt.space import Real, Integer, Categorical
import warnings
warnings.simplefilter("ignore", UserWarning)


from data_loader import AIDatasetLoader, ParticipantDatasetLoader
from human_models import ExplanationAwareHumanModel, SensitiveFeatures, SalientFeatures, ImportanceCategorization, AttributionSum
from ui import UI
from hyperparameter_tuner import HyperparameterTuner
from em_tuner import PopulationEMTuner

"""
Main entry point to run the entire pipeline: 
1) Load data
2) Create data loaders
3) Filter for the desired app/model/algorithm
4) Filter participant data for the desired XAIType
5) Select participant
6) Run experiment
"""
# ---------- USER-SPECIFIC CONFIG -----------
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")
data_dir = os.path.join(current_dir, 'data', 'datasets')
user_study_dir = os.path.join(current_dir, 'data', 'user study results')

# # ---------- LOAD AI DATA -----------

# Global cache for AI loaders
loader_cache = {}

def get_or_create_ai_loader(app_id, exp_method, model_name, xai_type):
    """
    Retrieve a cached AIDatasetLoader or create one based on participant-specific parameters.

    Args:
        app_id (str): The dataset identifier (e.g. "wine_quality").
        exp_method (str): Explanation generation method (e.g. "lime").
        model_name (str): AI model name (e.g. "mlp").
        xai_type (str): Explanation type (e.g. "importance", "none").
        base_data_dir (str): Directory containing values.csv, metadata.csv, etc.

    Returns:
        AIDatasetLoader: Filtered dataset loader based on the configuration.
    """
    cache_key = (app_id, exp_method, model_name, xai_type)

    if cache_key in loader_cache:
        return loader_cache[cache_key]

    # Load CSVs
    file_values = os.path.join(data_dir, 'values.csv')
    file_metadata = os.path.join(data_dir, 'metadata.csv')
    file_explanation = os.path.join(data_dir, f'{xai_type.lower()}.csv')

    df_values = pd.read_csv(file_values)
    df_metadata = pd.read_csv(file_metadata)
    df_explanation = pd.read_csv(file_explanation)

    explanation_columns = ['a0_i', 'a1_i', 'a2_i', 'a3_i', 'a4_i'] if xai_type != "none" else None
    loader = AIDatasetLoader(df_values, df_metadata, df_explanation, explanation_columns)

    # Apply filters based on expMethod and modelName
    def ai_condition(df):
        condition = pd.Series([True] * len(df), index=df.index)
        if 'appId' in df.columns:
            condition &= (df['appId'] == app_id)
        if 'expMethod' in df.columns:
            condition &= (df['expMethod'] == exp_method)
        if 'modelName' in df.columns:
            condition &= (df['modelName'] == model_name)
        return condition

    loader = loader.filter_loader(ai_condition)
    loader_cache[cache_key] = loader
    return loader


# ---------- LOAD PARTICIPANT DATA -----------
participant_file = os.path.join(user_study_dir, f'3-datasets-jan-09-2026-trials.csv')
participant_df = pd.read_csv(
    participant_file,
    na_values=[v for v in pd._libs.parsers.STR_NA_VALUES if v != 'None'],
    keep_default_na=False
)

# Create a ParticipantDatasetLoader
participant_loader = ParticipantDatasetLoader(participant_df)

# ================================================= #

ui = UI()

# Step 2: List all current participants
all_participants = participant_loader.list_all_participants()

# Randomly select a number of participants
selected_participants = random.sample(all_participants, min(80, len(all_participants)))


# shortlisted participants
# selected_participants = participant_loader.filter_loader(lambda df: (df['appId'].lower() == 'adult') & (df['XAIType'].lower() == 'attribution')).list_all_participants()

# Select participants based on a specific XAIType
# selected_participants = random.sample(
#     selected_participants,
#     40    
# )

# Step 4: Filter the DataFrame
filtered_df = participant_loader.df[participant_loader.df['Participant ID'].isin(selected_participants)]

# Step 5: Reinitialize participant loader with filtered participants
participant_loader = ParticipantDatasetLoader(filtered_df)


models = {
    "Sensitive-features categorization": SensitiveFeatures,
    "Salient-features categorization": SalientFeatures,
    "Importance categorization": ImportanceCategorization,
    "Attribution Sum": AttributionSum,
    "DT": ExplanationAwareHumanModel,
    "KNN": ExplanationAwareHumanModel,
    "MLP": ExplanationAwareHumanModel,
}




param_spaces = {
    "Sensitive-features categorization": [
        Real(1, 100.0, name="sensitivity"),
        Integer(1, 5, name="k"),
        Real(-4.0, -0.5, name="retrieval_threshold"),
    ],
    "Salient-features categorization": [
        Real(1, 100.0, name="sensitivity"),
        Integer(1, 4, name="k"),
        Real(-4.0, -0.5, name="retrieval_threshold"),
    ],
    "Importance categorization": [
        Real(1, 100.0, name="sensitivity"),
        Integer(1, 5, name="k"),
        Real(-4.0, -0.5, name="retrieval_threshold"),
    ],
    "Attribution Sum": [
        Real(0.1, 8.0, name="scaling_factor"),
        Integer(2, 5, name="k"),
        Real(-4.0, -0.5, name="retrieval_threshold"),
    ],
    # "DT": [
    #     Categorical(["DecisionTree"], name="model_type"),
    #     Integer(1, 5, name="max_depth"),
    #     Real(0, 5.0, name="smoothing_factor")
    # ],
    # "KNN": [
    #     Categorical(["KNN"], name="model_type"),
    #     Integer(1, 8, name="n_neighbors"),
    #     Real(0, 5.0, name="smoothing_factor")
    # ],

    # "MLP": [
    #     Categorical(["MLP"], name="model_type"),
    #     Integer(1, 50, name="hidden_dim"),
    #     Real(0, 5.0, name="smoothing_factor")
    # ],
}


# available_conditions = ["w/o XAI", "w/ XAI"]

# tuner_individual = HyperparameterTuner(models, param_spaces, get_or_create_ai_loader, participant_loader, ui, mode="individual", available_conditions=available_conditions)
# for session_num in [1, 2]:
#     tuner_individual.tune(n_iter=30, save=True, session_num=session_num)



models = {
    "Attribution Sum": AttributionSum,
    "Sensitive-features categorization": SensitiveFeatures,
    "Salient-features categorization": SalientFeatures,
    "Importance categorization": ImportanceCategorization,
}

tuner = PopulationEMTuner(
    models=models,
    param_spaces=param_spaces,
    loader_getter=get_or_create_ai_loader,              # your existing function
    participant_loader=participant_loader,    # your existing loader
    ui=UI(),
    available_conditions=("w/ XAI", "w/o XAI"),
    optimization_metric="nll_model_participant",
    max_participants=2000,                    # optional cap
)

import time
# start_time = time.time() 

# group_df, assign_df = tuner.fit(
#     sessions=[1, 2],          # your session numbers
#     n_rounds=5,               # usually 2–3 is enough
#     jitter_K=7,               # speed knob (0 fastest, 6 good, 10 better)
#     jitter_scale=2.0,         # size of local search
#     soft=True,                # stabilizes strategy ties
#     beta=3.0,                 # softmax sharpness
#     cov_diagonal=True,        # fastest + stable
# )

# group_path, assign_path = tuner.save_results("./results/pop_em")
# print(group_path, assign_path)

# end_time = time.time()
# print(f"EM Tuning completed in {end_time - start_time:.2f} seconds.")

participant_loader = ParticipantDatasetLoader(participant_df)
selected_participants = random.sample(participant_loader.list_all_participants(), len(participant_loader.list_all_participants()))
filtered_df = participant_loader.df[participant_loader.df['Participant ID'].isin(selected_participants)]
participant_loader = ParticipantDatasetLoader(filtered_df)

tuner.participant_loader = participant_loader
tuner.load_group_gaussians("./results/pop_em/importance_attribution_gaussians.csv")

start_time = time.time()

assign_df = tuner.assign_all_participants(
    sessions=[1, 2],
    search="deterministic",
    max_evals_per_strategy=10,   # <= 10
    step_scale=1.0,
    shrink=0.5,
    min_step_frac=0.05,
    soft=False,
)

assign_df.to_csv("./results/pop_em_subset/refit_all_assignments_detlocal.csv", index=False)

print(f"Refitting all participants completed in {time.time() - start_time:.2f} seconds.")
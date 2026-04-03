import os
import pandas as pd
import sys
import random
import numpy as np
from pathlib import Path

# parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
parent_dir = os.getcwd()
sys.path.insert(0, parent_dir)

from human_models import ExplanationAwareHumanModel, AttributionSum, SalientFeatures, SensitiveFeatures, ImportanceCategorization
from data_loader import AIDatasetLoader
from experiment_runner import StrategyComparisonRunner
from ui import UI
from collections import Counter

# Set up paths
data_dir = os.path.join(parent_dir, 'data', 'datasets')
file_values = os.path.join(data_dir, 'values.csv')
file_metadata = os.path.join(data_dir, 'metadata.csv')
file_explanation = os.path.join(data_dir, 'importance.csv')  # or 'attribution.csv'

# Load data
df_values = pd.read_csv(file_values)
df_metadata = pd.read_csv(file_metadata)
df_explanation = pd.read_csv(file_explanation)


# Forest Cover
forest_blocks = [
    {
        "train": [24, 25, 154, 168, 183, 195, 215, 266, 292, 295],
        "testWithXAI": [21, 61, 102, 110, 130, 137, 151, 152, 179, 217, 223, 234, 239, 247, 270, 273, 278, 290],
        "testWithoutXAI": [8, 17, 22, 32, 53, 73, 81, 86, 95, 118, 122, 145, 172, 219, 220, 256, 260, 291]
    },
    {
        "train": [50, 78, 155, 163, 203, 206, 222, 225, 257, 298],
        "testWithXAI": [0, 44, 48, 65, 101, 135, 136, 139, 167, 175, 201, 207, 233, 236, 245, 246, 287, 288],
        "testWithoutXAI": [2, 9, 20, 41, 42, 70, 82, 89, 91, 94, 107, 109, 149, 177, 194, 205, 226, 274]
    }
]

# Wine Quality
wine_blocks = [
    {
        "train": [8, 25, 32, 43, 51, 66, 73, 81, 82, 121],
        "testWithXAI": [0, 4, 6, 22, 27, 36, 41, 42, 46, 62, 64, 65, 80, 86, 98, 101, 111, 117],
        "testWithoutXAI": [3, 13, 31, 44, 45, 52, 59, 67, 74, 75, 76, 78, 90, 91, 97, 105, 106, 110]
    },
    {
        "train": [7, 10, 24, 33, 40, 54, 87, 89, 114, 120],
        "testWithXAI": [20, 29, 34, 38, 39, 47, 56, 60, 71, 84, 88, 96, 100, 102, 103, 107, 112, 118],
        "testWithoutXAI": [1, 5, 14, 19, 21, 23, 26, 35, 55, 61, 68, 69, 77, 83, 108, 113, 116, 119]
    }
]

# Adult
adult_blocks = [
    {
        "train": [14, 17, 119, 141, 168, 169, 213, 215, 260, 289],
        "testWithXAI": [2, 33, 35, 75, 76, 84, 95, 117, 125, 135, 158, 172, 190, 194, 210, 235, 246, 261],
        "testWithoutXAI": [5, 6, 18, 58, 81, 132, 145, 156, 161, 165, 171, 179, 221, 275, 276, 277, 294, 296]
    },
    {
        "train": [45, 47, 139, 187, 198, 212, 255, 263, 293, 295],
        "testWithXAI": [3, 19, 23, 26, 29, 66, 78, 97, 101, 114, 121, 150, 184, 207, 232, 267, 281, 282],
        "testWithoutXAI": [4, 20, 49, 50, 89, 94, 103, 111, 120, 224, 240, 242, 243, 268, 269, 290, 291, 298]
    }
]

# Mushrooms
mushroom_blocks = [
    {
        "train": [4, 74, 78, 108, 126, 195, 237, 330, 383, 393],
        "testWithXAI": [20, 22, 40, 54, 64, 81, 101, 106, 112, 125, 138, 245, 247, 297, 312, 319, 323, 352],
        "testWithoutXAI": [59, 61, 107, 141, 154, 173, 192, 198, 208, 219, 256, 274, 299, 316, 325, 329, 365, 390]
    },
    {
        "train": [10, 50, 96, 116, 134, 170, 224, 250, 359, 399],
        "testWithXAI": [52, 58, 91, 159, 181, 184, 202, 207, 226, 244, 260, 270, 271, 317, 321, 345, 350, 356],
        "testWithoutXAI": [37, 49, 66, 95, 120, 135, 143, 150, 155, 206, 213, 218, 228, 234, 279, 282, 366, 392]
    }
]

# -------------------------------
# 2.  HELPER: BUILD SESSION LIST
# -------------------------------
def make_session(blocks):
    """
    Returns both:
    - session: the list of instance entries
    - session_meta: a dict mapping instance_id -> (is_training, with_explanation)
    """
    session = []
    session_meta = {}
    
    for block in blocks:
        for key in ("train", "testWithXAI", "testWithoutXAI"):
            ids = random.sample(block[key], len(block[key]))
            for i in ids:
                if key == "train":
                    session.append({"instance_id": i, "is_training": True,  "with_explanation": True})
                    session_meta[i] = (True, True)
                elif key == "testWithXAI":
                    session.append({"instance_id": i, "is_training": False, "with_explanation": True})
                    session_meta[i] = (False, True)
                else:
                    session.append({"instance_id": i, "is_training": False, "with_explanation": False})
                    session_meta[i] = (False, False)

    return session, session_meta


param_config = dict(
    k=3,
    decay_param=0.5,
    sensitivity=20.0,
    retrieval_threshold=-2.5,
    scaling_factor=1.0
)


CSV_COLUMNS = [
    "appId", "strategy", "k", "decay_param", "sensitivity", "retrieval_threshold",
    "time_scale", "base_decision_time", "encoding_time_per_feature",
    "save_new_exemplar", "temperature", "scaling_factor",
    "instance_id", "trialType", "Tested w/ XAI",
    "predicted", "prob_truth", "prob_0", "prob_1", "Correct"
]

def save_rows_to_csv(rows, path, write_header):
    if not rows:
        return

    df = pd.DataFrame(rows)

    # Ensure all expected columns are present
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[CSV_COLUMNS]  # enforce exact column order

    df.to_csv(path, mode="a", header=write_header, index=False)


def make_mask(df, **wanted):
    """
    Build a row mask that is True when every present column matches
    the requested value.  Columns that are absent are ignored.
    """
    mask = pd.Series(True, index=df.index)
    for col, val in wanted.items():
        if col in df.columns:          # only test if the column exists
            mask &= (df[col] == val)
    return mask


# -------------------------------
# 1.  CONFIGURE THE SWEEP
# -------------------------------
DATASETS = {
    "Forest Cover":   {"appId": "forest_cover",  "model":"xgboost",  "expMethod": "shap", "blocks": forest_blocks},
    "Wine Quality":   {"appId": "wine_quality",  "model":"mlp", "expMethod": "lime",  "blocks": wine_blocks},
    "Adult Income":          {"appId": "adult",         "model":"xgboost", "expMethod": "lime",  "blocks": adult_blocks},
    "Mushrooms":      {"appId": "mushrooms",     "model":"mlp", "expMethod": "shap",  "blocks": mushroom_blocks},
}

N_PARTICIPANTS = 2                     # per (dataset, xai_type, strategy)
RNG = np.random.default_rng(seed=1234)  # reproducible

# sampling helpers -------------------------------------------------------------
def sample_sensitivity():
    return float(np.clip(RNG.normal(16, 14),  1, 40))

def sample_k():
    return int(np.clip(round(RNG.normal(3, 1.2)), 1, 4))

def sample_scaling():
    return float(max(0.1, RNG.normal(2.9, 1.8)))   #  ≥0.1

def sample_thresh():
    return float(np.clip(RNG.normal(-2.3, 0.45), -2.9, -2.0))

STRATEGIES = {
    "Attribution sum":  AttributionSum,
    "Sensitive-features categorization":     SensitiveFeatures,
    "Salient-features categorization":  SalientFeatures,
    "Importance categorization":           ImportanceCategorization,
}



# which strategies are allowed for each XAI type -------------------------------
STRATS_BY_XAI = {
    "none":        ["Sensitive-features categorization", ],
    "importance":  ["Sensitive-features categorization", "Salient-features categorization",
                    "Importance categorization", "Attribution sum"],
    "attribution": ["Sensitive-features categorization", "Attribution sum"],
}

# ------------------------------------------------------------------------------ 
CSV_COLUMNS += ["XAIType"]        # keep same order but add a column
rows, out_path = [], Path("results/CoAX simulated data.csv")
save_every = 20_000                # flush every 20 k rows
first_write = True


for ds_name, ds_cfg in DATASETS.items():
    print(f"Dataset ➜ {ds_name}")
    # make a fresh session *once* per dataset (same for every participant here)
    session, session_meta = make_session(ds_cfg["blocks"])

    for xai_type in ("None", "Importance", "Attribution"):
        print(f"  XAI-type ➜ {xai_type}")
        
        if xai_type == "None" or xai_type == "Importance":
            print()


        # pick the correct explanation file --------------------
        file_expl = os.path.join(data_dir, f"{'attribution' if xai_type=='Attribution' else 'importance'}.csv")
        df_expl   = pd.read_csv(file_expl)

        ai_loader = (
            AIDatasetLoader(
                feature_values_df       = df_values,
                metadata_df             = df_metadata,
                explanation_values_df   = df_expl,
                explanation_columns     = ['a0_i','a1_i','a2_i','a3_i','a4_i']
            )
            .filter_loader(
                lambda df,
                    app = ds_cfg["appId"],
                    mdl = ds_cfg["model"],
                    exp = ds_cfg["expMethod"]:
                make_mask(df, appId=app, modelName=mdl, expMethod=exp)
            )
        )
        
        # loop over eligible strategies -------------------------
        for strat_name in STRATS_BY_XAI[xai_type]:
            StratCls = STRATEGIES[strat_name]
            print(f"    Strategy ➜ {strat_name}")

            for participant in range(N_PARTICIPANTS):
                # ---------- 1) sample hyper-parameters ----------
                config = {
                    "k"                 : sample_k(),
                    "sensitivity"       : sample_sensitivity(),
                    "retrieval_threshold": sample_thresh(),
                    "decay_param"       : 0.5,              # fixed defaults
                }
                if strat_name == "Attribution sum":
                    config["scaling_factor"]  = sample_scaling()
                    config["explanation_type"] = xai_type   # pass importance/attribution
                # (PartialMatching / Topk / ExplanationOnly ignore scaling_factor)

                # ---------- 2) build & run ----------------------
                strategy = StratCls(**config)
                runner   = StrategyComparisonRunner(strategy, ai_loader, UI())
                logs     = runner.generalized_run_experiment(session)
                # print(logs)

                # ---------- 3) collect results -----------------
                for lg in logs:
                    # if lg["ai_prediction"] is None:
                    #     print(lg)
                    if lg["step"] != "infer":
                        continue
                    # if  not lg["response"]:
                    #     print("Huh")
                    #     continue
                    # if not probs.get(0, None) and not probs.get(1, None):
                    #     print(lg)
                    probs   = lg["response"]
                    truth   = lg["ai_prediction"]
                    is_tr, with_xai = session_meta[lg["instance_id"]]

                    # print(probs)
                    # sys.exit()

                    keys = np.array(list(probs.keys()))
                    weights = np.array(list(probs.values()))
                    choice = np.random.choice(keys, p=weights/weights.sum())

                    rows.append({
                        "appId"          : ds_name,
                        "XAIType"         : xai_type,
                        "Strategy"         : strat_name,
                        **config,                          # write all hyper-params
                        "Instance Index"      : lg["instance_id"],
                        "trialType"      : "Train" if is_tr else "Test",
                        "Tested w/ XAI" : "w/ XAI" if with_xai else "w/o XAI",
                        "predicted"        : choice,
                        "prob_truth"       : probs.get(truth, 0.0),
                        "prob_0"           : probs.get(0, "this"),
                        "prob_1"           : probs.get(1, "this"),
                        "Correct"          : int(choice == truth)
                    })

                # flush periodically so RAM stays sane ----------
                if len(rows) >= save_every:
                    save_rows_to_csv(rows, out_path, write_header=first_write)
                    first_write = False
                    rows.clear()



# final write ---------------------------------------------------------------
save_rows_to_csv(rows, out_path, write_header=first_write)
print(f"✅ finished – data stored in {out_path.resolve()}")
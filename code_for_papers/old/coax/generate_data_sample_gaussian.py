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
def make_session(blocks, train_with_explanation=True):
    """
    Returns both:
    - session: the list of instance entries
    - session_meta: a dict mapping instance_id -> (is_training, with_explanation)

    train_with_explanation controls whether training trials are flagged as with_explanation.
    """
    session = []
    session_meta = {}

    for block in blocks:
        for key in ("train", "testWithXAI", "testWithoutXAI"):
            ids = random.sample(block[key], len(block[key]))
            for i in ids:
                if key == "train":
                    session.append({"instance_id": i, "is_training": True, "with_explanation": bool(train_with_explanation)})
                    session_meta[i] = (True, bool(train_with_explanation))
                elif key == "testWithXAI":
                    session.append({"instance_id": i, "is_training": False, "with_explanation": True})
                    session_meta[i] = (False, True)
                else:
                    session.append({"instance_id": i, "is_training": False, "with_explanation": False})
                    session_meta[i] = (False, False)

    return session, session_meta


def make_mask(df, **wanted):
    """
    Build a row mask that is True when every present column matches
    the requested value.  Columns that are absent are ignored.
    """
    mask = pd.Series(True, index=df.index)
    for col, val in wanted.items():
        if col in df.columns:
            mask &= (df[col] == val)
    return mask


# -------------------------------
# 3.  PARAMETER SAMPLING (ROW-BASED)
# -------------------------------
df = pd.read_csv("./results/02-01-2026/three datasets strategies.csv")

def sample_config_row(df_params, strategy_name, tested_with_xai_value):
    """
    Sample ONE row for a participant config.

    - First try to sample from (Strategy == strategy_name) & (Tested w/ XAI == tested_with_xai_value)
    - If empty, relax to (Strategy == strategy_name) only.
    - Resample until required fields for that strategy are non-null.
    """
    while True:
        sub = df_params[(df_params["Strategy"] == strategy_name) & (df_params["Tested w/ XAI"] == tested_with_xai_value)]
        if sub.empty:
            sub = df_params[(df_params["Strategy"] == strategy_name)]
        if sub.empty:
            raise ValueError(f"No rows found in df for Strategy='{strategy_name}' (even after relaxation).")

        row = sub.sample(1).iloc[0]

        # Required fields
        required = ["k", "retrieval_threshold"]
        if strategy_name == "Attribution Sum":
            required += ["scaling_factor"]
        else:
            required += ["sensitivity"]

        ok = True
        for c in required:
            if c not in row.index or pd.isna(row[c]):
                ok = False
                break

        if ok:
            return row


CSV_COLUMNS = [
    "Participant Id", "appId", "Strategy", "k", "decay_param", "sensitivity", "retrieval_threshold", "scaling_factor",
    "Instance Index", "trialType", "Tested w/ XAI",
    "predicted", "Prob correct", "Prob 0", "Prob 1", "Correct",
    "XAIType"
]

def save_rows_to_csv(rows, path, write_header):
    if not rows:
        return

    df_out = pd.DataFrame(rows)

    # Ensure all expected columns are present
    for col in CSV_COLUMNS:
        if col not in df_out.columns:
            df_out[col] = np.nan
    df_out = df_out[CSV_COLUMNS]  # enforce exact column order

    df_out.to_csv(path, mode="a", header=write_header, index=False)


# -------------------------------
# 4.  CONFIGURE THE SWEEP
# -------------------------------
DATASETS = {
    "Forest Cover":   {"appId": "forest_cover",  "model":"xgboost",  "expMethod": "shap", "blocks": forest_blocks},
    "Wine Quality":   {"appId": "wine_quality",  "model":"mlp",      "expMethod": "lime", "blocks": wine_blocks},
    "Adult Income":    {"appId": "adult",         "model":"xgboost",  "expMethod": "lime", "blocks": adult_blocks},
    # "Mushrooms":      {"appId": "mushrooms",     "model":"mlp",      "expMethod": "shap", "blocks": mushroom_blocks},
}

N_PARTICIPANTS = 20

STRATEGIES = {
    "Attribution Sum":                 AttributionSum,
    "Sensitive-features categorization": SensitiveFeatures,
    "Salient-features categorization":   SalientFeatures,
    "Importance categorization":         ImportanceCategorization,
}

STRATS_BY_XAI = {
    "None": ["Sensitive-features categorization"],
    "Importance": ["Sensitive-features categorization",
                   "Salient-features categorization",
                   "Importance categorization",
                   "Attribution Sum"],
    "Attribution": ["Sensitive-features categorization",
                    "Attribution Sum"],
}

rows, out_path = [], Path("results/CoAX simulated data gaussian.csv")
save_every = 20_000
open(out_path, "w").close()
first_write = True


def run_one_param_regime(
    *,
    ds_cfg,
    ds_name,
    xai_type,
    strat_name,
    StratCls,
    ai_loader,
    participant_idx,
    param_regime,              # "w/ XAI" or "w/o XAI"
    train_with_explanation,    # bool
):
    """
    Run one participant simulation under a specific parameter regime (w/ XAI vs w/o XAI),
    then keep only rows whose trial "Tested w/ XAI" matches param_regime.
    """

    # Skip conditions requested:
    # (1) Importance categorization w/o XAI
    if strat_name == "Importance categorization" and param_regime == "w/o XAI":
        return -1

    # (2) Sensitive-features categorization w/ XAI if XAI type is Attribution
    if xai_type == "Attribution" and strat_name == "Sensitive-features categorization" and param_regime == "w/ XAI":
        return -1


    # 1) sample one config ROW for this participant+regime
    # cfg_row = sample_config_row(df, strat_name, param_regime)

    # config = {
    #     "k": int(cfg_row["k"]),
    #     "retrieval_threshold": float(cfg_row["retrieval_threshold"]),
    #     "decay_param": 0.5,
    # }
    # sample one parameter set from the learned Gaussian
    params = tuner.sample_params(
        strategy=strat_name,
        xai_type=xai_type.lower(),
        condition=param_regime,
        n=1,
    )[0]

    config = {
        "k": int(params["k"]),
        "retrieval_threshold": float(params["retrieval_threshold"]),
        "decay_param": params.get("decay_param", 0.5),
    }

    if strat_name == "Attribution Sum":
        config["scaling_factor"] = float(params["scaling_factor"])
        config["explanation_type"] = xai_type.lower()
    else:
        config["sensitivity"] = float(params["sensitivity"])


    # if strat_name == "Attribution Sum":
    #     config["scaling_factor"] = float(cfg_row["scaling_factor"])
    #     config["explanation_type"] = xai_type.lower()
    # else:
    #     config["sensitivity"] = float(cfg_row["sensitivity"])

    # 2) build session (possibly with train explanation relaxed)
    session, session_meta = make_session(ds_cfg["blocks"], train_with_explanation=train_with_explanation)

    # 3) run
    strategy = StratCls(**config)
    runner = StrategyComparisonRunner(strategy, ai_loader, UI())
    logs = runner.generalized_run_experiment(session)

    # 4) collect only matching trials
    kept = 0
    for lg in logs:
        if lg.get("Step") != "infer":
            continue

        probs = lg["response"]
        truth = lg["ai_prediction"]

        is_tr, with_xai = session_meta[lg["instance_id"]]
        trial_tested = "w/ XAI" if with_xai else "w/o XAI"

        # STRICT: keep only rows matching the parameter regime
        if trial_tested != param_regime:
            continue

        try:
            keys = np.array(list(probs.keys()))
            weights = np.array(list(probs.values()))
            choice = np.random.choice(keys, p=weights / weights.sum())
        except Exception as e:
            print("Error in sampling choice from probs:", probs)
            raise e

        rows.append({
            "appId": ds_cfg["appId"],
            "XAIType": xai_type,
            "Strategy": strat_name,
            **config,
            "Instance Index": lg["instance_id"],
            "trialType": "Train" if is_tr else "Test",
            "Tested w/ XAI": trial_tested,
            "predicted": choice,
            "Prob correct": probs.get(truth, 0.0),
            "Prob 0": probs.get(0, np.nan),
            "Prob 1": probs.get(1, np.nan),
            "Correct": int(choice == truth),
            "Participant Id": participant_idx
        })
        kept += 1

    return kept


from em_tuner import PopulationEMTuner
from skopt.space import Real, Integer, Categorical
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

tuner = PopulationEMTuner(
    models=None,
    param_spaces=param_spaces,
    loader_getter=None,              # your existing function
    participant_loader=None,    # your existing loader
    ui=UI(),
    available_conditions=("w/ XAI", "w/o XAI"),
    optimization_metric="nll_model_participant",
    max_participants=2000, 
)

tuner.load_group_gaussians("./results/pop_em/importance_attribution_gaussians.csv")

participant_counter = 0
for ds_name, ds_cfg in DATASETS.items():
    print(f"Dataset ➜ {ds_name}")

    for xai_type in ("None", "Importance", "Attribution"):
        print(f"  XAI-type ➜ {xai_type}")

        # pick correct explanation file
        file_expl = os.path.join(data_dir, f"{'attribution' if xai_type=='Attribution' else 'importance'}.csv")
        df_expl = pd.read_csv(file_expl)

        ai_loader = (
            AIDatasetLoader(
                feature_values_df=df_values,
                metadata_df=df_metadata,
                explanation_values_df=df_expl,
                explanation_columns=['a0_i', 'a1_i', 'a2_i', 'a3_i', 'a4_i']
            )
            .filter_loader(
                lambda df,
                       app=ds_cfg["appId"],
                       mdl=ds_cfg["model"],
                       exp=ds_cfg["expMethod"]:
                make_mask(df, appId=app, modelName=mdl, expMethod=exp)
            )
        )

        if STRATS_BY_XAI.get(xai_type, []) == []:
            print("    (no eligible strategies for this XAI type; skipping)")
            continue

        for strat_name in STRATS_BY_XAI[xai_type]:
            StratCls = STRATEGIES[strat_name]
            print(f"    Strategy ➜ {strat_name}")

            for participant in range(N_PARTICIPANTS):
                participant_counter += 1

                # Run two independent simulations: one for params fit to w/ XAI, one for params fit to w/o XAI.
                for param_regime in ("w/ XAI", "w/o XAI"):

                    # First attempt: keep your original training setup (train_with_explanation=True)
                    kept = run_one_param_regime(
                        ds_cfg=ds_cfg,
                        ds_name=ds_name,
                        xai_type=xai_type,
                        strat_name=strat_name,
                        StratCls=StratCls,
                        ai_loader=ai_loader,
                        participant_idx=participant_counter,
                        param_regime=param_regime,
                        train_with_explanation=True,
                    )


                    if kept == -1:
                        # skipped condition
                        continue

                    # If nothing matched, relax ONLY the training-with-XAI condition by making training match the regime
                    if kept == 0:
                        kept = run_one_param_regime(
                            ds_cfg=ds_cfg,
                            ds_name=ds_name,
                            xai_type=xai_type,
                            strat_name=strat_name,
                            StratCls=StratCls,
                            ai_loader=ai_loader,
                            participant_idx=participant_counter,
                            param_regime=param_regime,
                            train_with_explanation=(param_regime == "w/ XAI"),
                        )

                # flush periodically
                if len(rows) >= save_every:
                    save_rows_to_csv(rows, out_path, write_header=first_write)
                    first_write = False
                    rows.clear()


# final write
save_rows_to_csv(rows, out_path, write_header=first_write)
print(f"✅ finished – data stored in {out_path.resolve()}")



df_assign = pd.read_csv("./results/pop_em_subset/importance_attribution_refit.csv")

def build_strategy_sampler(df, dataset, xai_type, tested_with_xai):
    sub = df[
        (df["appId"] == dataset) &
        (df["XAIType"] == xai_type) &
        (df["Tested w/ XAI"] == tested_with_xai)
    ]

    if sub.empty:
        return None

    counts = sub["Strategy"].value_counts()
    strategies = counts.index.tolist()
    probs = (counts / counts.sum()).values

    return strategies, probs

def sample_strategies(strategies, probs, n):
    return np.random.choice(strategies, size=n, p=probs)


rows2 = []
out_path2 = Path("results/CoAX simulated data_strategy_sampled.csv")
open(out_path2, "w").close()
first_write2 = True

participant_counter = 0

for ds_name, ds_cfg in DATASETS.items():
    print(f"[SAMPLED] Dataset ➜ {ds_name}")

    for xai_type in ("None", "Importance", "Attribution"):
        print(f"  XAI-type ➜ {xai_type}")

        # explanation file
        file_expl = os.path.join(
            data_dir,
            f"{'attribution' if xai_type=='Attribution' else 'importance'}.csv"
        )
        df_expl = pd.read_csv(file_expl)

        ai_loader = (
            AIDatasetLoader(
                feature_values_df=df_values,
                metadata_df=df_metadata,
                explanation_values_df=df_expl,
                explanation_columns=['a0_i', 'a1_i', 'a2_i', 'a3_i', 'a4_i']
            )
            .filter_loader(
                lambda df,
                       app=ds_cfg["appId"],
                       mdl=ds_cfg["model"],
                       exp=ds_cfg["expMethod"]:
                make_mask(df, appId=app, modelName=mdl, expMethod=exp)
            )
        )

        for param_regime in ("w/ XAI", "w/o XAI"):

            sampler = build_strategy_sampler(
                df_assign,
                dataset=ds_name,
                xai_type=xai_type,
                tested_with_xai=param_regime
            )

            if sampler is None:
                print(f"    (no strategy distribution for {param_regime}; skipping)")
                continue

            strategies, probs = sampler

            for participant in range(N_PARTICIPANTS):
                participant_counter += 1

                strat_name = np.random.choice(strategies, p=probs)
                StratCls = STRATEGIES[strat_name]

                kept = run_one_param_regime(
                    ds_cfg=ds_cfg,
                    ds_name=ds_name,
                    xai_type=xai_type,
                    strat_name=strat_name,
                    StratCls=StratCls,
                    ai_loader=ai_loader,
                    participant_idx=participant_counter,
                    param_regime=param_regime,
                    train_with_explanation=True,
                )

                if kept == -1:
                    continue

                if kept == 0:
                    kept = run_one_param_regime(
                        ds_cfg=ds_cfg,
                        ds_name=ds_name,
                        xai_type=xai_type,
                        strat_name=strat_name,
                        StratCls=StratCls,
                        ai_loader=ai_loader,
                        participant_idx=participant_counter,
                        param_regime=param_regime,
                        train_with_explanation=(param_regime == "w/ XAI"),
                    )

                # flush periodically
                if len(rows2) >= save_every:
                    save_rows_to_csv(rows2, out_path2, write_header=first_write2)
                    first_write2 = False
                    rows2.clear()

save_rows_to_csv(rows2, out_path2, write_header=first_write2)
print(f"✅ strategy-sampled data stored in {out_path2.resolve()}")


# ============================================================
# 5.  FREQUENCY-MATCHED SIMULATION (BY Strategy distribution)
# ============================================================
# This block re-runs the same simulation logic, but allocates the number
# of simulated participants per strategy to match the empirical frequency
# of Strategy in df for each (XAIType, Tested w/ XAI) combination.
#
# Interpretation:
# - For each dataset + xai_type + param_regime:
#     total simulated participants = N_PARTICIPANTS
#     participants per strategy ~ proportional to df frequency for that combo
#
# Note: This writes to a new CSV (so you don't overwrite the earlier run).

def allocate_by_frequency(df_params, *, appId, xai_type, param_regime, eligible_strategies, total_n):
    """
    Returns dict: {strategy_name: n_participants} where n_participants sums to total_n.
    Uses largest-remainder rounding so totals match exactly.
    """
    sub = df_params[
        (df_params["appId"] == appId) &
        (df_params["XAIType"] == xai_type) &
        (df_params["Tested w/ XAI"] == param_regime)
    ]

    # count only eligible strategies
    counts = sub["Strategy"].value_counts().reindex(eligible_strategies).fillna(0).astype(int)

    # if nothing found, fall back to uniform allocation across eligible strategies
    if counts.sum() == 0:
        base = total_n // len(eligible_strategies)
        rem = total_n - base * len(eligible_strategies)
        alloc = {s: base for s in eligible_strategies}
        for s in eligible_strategies[:rem]:
            alloc[s] += 1
        return alloc

    props = counts / counts.sum()

    # initial floor allocation
    raw = props * total_n
    floored = np.floor(raw).astype(int)
    alloc = {s: int(floored.loc[s]) for s in eligible_strategies}

    # distribute remainder by largest fractional parts
    remainder = total_n - sum(alloc.values())
    if remainder > 0:
        frac = (raw - floored).sort_values(ascending=False)
        for s in frac.index[:remainder]:
            alloc[s] += 1

    return alloc


# # Output file for the frequency-matched run
# rows_freq, out_path_freq = [], Path("results/CoAX simulated data (freq-matched).csv")
# save_every_freq = 20_000
# open(out_path_freq, "w").close()
# first_write_freq = True

# participant_counter_freq = 0

# for ds_name, ds_cfg in DATASETS.items():
#     print(f"[FREQ] Dataset ➜ {ds_name}")

#     for xai_type in ("None", "Importance", "Attribution"):
#         print(f"  [FREQ] XAI-type ➜ {xai_type}")

#         # pick correct explanation file
#         file_expl = os.path.join(data_dir, f"{'attribution' if xai_type=='Attribution' else 'importance'}.csv")
#         df_expl = pd.read_csv(file_expl)

#         ai_loader = (
#             AIDatasetLoader(
#                 feature_values_df=df_values,
#                 metadata_df=df_metadata,
#                 explanation_values_df=df_expl,
#                 explanation_columns=['a0_i', 'a1_i', 'a2_i', 'a3_i', 'a4_i']
#             )
#             .filter_loader(
#                 lambda df,
#                        app=ds_cfg["appId"],
#                        mdl=ds_cfg["model"],
#                        exp=ds_cfg["expMethod"]:
#                 make_mask(df, appId=app, modelName=mdl, expMethod=exp)
#             )
#         )

#         eligible = STRATS_BY_XAI.get(xai_type, [])
#         if not eligible:
#             print("    [FREQ] (no eligible strategies for this XAI type; skipping)")
#             continue

#         # For each param regime, allocate participants across strategies by empirical freq
#         for param_regime in ("w/ XAI", "w/o XAI"):
#             alloc = allocate_by_frequency(
#                 df,
#                 appId=ds_cfg["appId"],
#                 xai_type=xai_type,
#                 param_regime=param_regime,
#                 eligible_strategies=eligible,
#                 total_n=N_PARTICIPANTS,
#             )

#             print(f"    [FREQ] Regime ➜ {param_regime} | Allocation: {alloc}")

#             for strat_name, n_part in alloc.items():
#                 StratCls = STRATEGIES[strat_name]
#                 print(f"      [FREQ] Strategy ➜ {strat_name} | n={n_part}")

#                 for _ in range(n_part):
#                     participant_counter_freq += 1

#                     kept = run_one_param_regime(
#                         ds_cfg=ds_cfg,
#                         ds_name=ds_name,
#                         xai_type=xai_type,
#                         strat_name=strat_name,
#                         StratCls=StratCls,
#                         ai_loader=ai_loader,
#                         participant_idx=participant_counter_freq,
#                         param_regime=param_regime,
#                         train_with_explanation=True,
#                     )

#                     if kept == -1:
#                         continue

#                     # If nothing matched, relax ONLY the training-with-XAI condition by making training match the regime
#                     if kept == 0:
#                         kept = run_one_param_regime(
#                             ds_cfg=ds_cfg,
#                             ds_name=ds_name,
#                             xai_type=xai_type,
#                             strat_name=strat_name,
#                             StratCls=StratCls,
#                             ai_loader=ai_loader,
#                             participant_idx=participant_counter_freq,
#                             param_regime=param_regime,
#                             train_with_explanation=(param_regime == "w/ XAI"),
#                         )

#                     # flush periodically
#                     if len(rows) >= save_every_freq:
#                         save_rows_to_csv(rows, out_path_freq, write_header=first_write_freq)
#                         first_write_freq = False
#                         rows.clear()

# # final write
# save_rows_to_csv(rows, out_path_freq, write_header=first_write_freq)
# print(f"✅ [FREQ] finished – data stored in {out_path_freq.resolve()}")




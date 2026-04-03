"""
CoAX simulation runner (frequency-matched by strategy) with training-size sweep.

Key features:
- Strategy sampling is frequency/percentage-matched using empirical distribution from df_params
  for each (appId, XAIType, Tested w/ XAI) combination.
- Training set size is swept over TRAIN_SIZES (e.g., 1..15 ~step 3).
- Training instances are NOT constrained to the original block["train"] lists.
  Instead, we sample n_train training instances from the available instance pool
  for the dataset condition (appId/model/expMethod), excluding the block's test ids if possible.
- Saves N_train as a new column in the output CSV.

Assumptions:
- Your `human_models`, `data_loader`, `experiment_runner`, `ui` modules are importable.
- Your df_params file contains columns: Strategy, Tested w/ XAI, appId, XAIType, k, retrieval_threshold,
  sensitivity and/or scaling_factor.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path

# -------------------------------
# 0) Imports from your codebase
# -------------------------------
parent_dir = os.getcwd()
sys.path.insert(0, parent_dir)

from human_models import AttributionSum, SalientFeatures, SensitiveFeatures, ImportanceCategorization
from data_loader import AIDatasetLoader
from experiment_runner import StrategyComparisonRunner
from ui import UI

# -------------------------------
# 1) Paths + data
# -------------------------------
data_dir = os.path.join(parent_dir, "data", "datasets", "standard set")
file_values = os.path.join(data_dir, "values.csv")
file_metadata = os.path.join(data_dir, "metadata.csv")

df_values = pd.read_csv(file_values)
df_metadata = pd.read_csv(file_metadata)

# Parameter-results CSV (used to sample hyperparams + compute strategy frequencies)
df_params = pd.read_csv("./results/pop_em_subset/refit_all_assignments_detlocal-Jan-10.csv")

# -------------------------------
# 2) Blocks (test sets remain fixed)
# -------------------------------
forest_blocks = [
    {
        "train": [24, 25, 154, 168, 183, 195, 215, 266, 292, 295],
        "testWithXAI": [21, 61, 102, 110, 130, 137, 151, 152, 179, 217, 223, 234, 239, 247, 270, 273, 278, 290],
        "testWithoutXAI": [8, 17, 22, 32, 53, 73, 81, 86, 95, 118, 122, 145, 172, 219, 220, 256, 260, 291],
    },
    {
        "train": [50, 78, 155, 163, 203, 206, 222, 225, 257, 298],
        "testWithXAI": [0, 44, 48, 65, 101, 135, 136, 139, 167, 175, 201, 207, 233, 236, 245, 246, 287, 288],
        "testWithoutXAI": [2, 9, 20, 41, 42, 70, 82, 89, 91, 94, 107, 109, 149, 177, 194, 205, 226, 274],
    },
]

wine_blocks = [
    {
        "train": [8, 25, 32, 43, 51, 66, 73, 81, 82, 121],
        "testWithXAI": [0, 4, 6, 22, 27, 36, 41, 42, 46, 62, 64, 65, 80, 86, 98, 101, 111, 117],
        "testWithoutXAI": [3, 13, 31, 44, 45, 52, 59, 67, 74, 75, 76, 78, 90, 91, 97, 105, 106, 110],
    },
    {
        "train": [7, 10, 24, 33, 40, 54, 87, 89, 114, 120],
        "testWithXAI": [20, 29, 34, 38, 39, 47, 56, 60, 71, 84, 88, 96, 100, 102, 103, 107, 112, 118],
        "testWithoutXAI": [1, 5, 14, 19, 21, 23, 26, 35, 55, 61, 68, 69, 77, 83, 108, 113, 116, 119],
    },
]

adult_blocks = [
    {
        "train": [14, 17, 119, 141, 168, 169, 213, 215, 260, 289],
        "testWithXAI": [2, 33, 35, 75, 76, 84, 95, 117, 125, 135, 158, 172, 190, 194, 210, 235, 246, 261],
        "testWithoutXAI": [5, 6, 18, 58, 81, 132, 145, 156, 161, 165, 171, 179, 221, 275, 276, 277, 294, 296],
    },
    {
        "train": [45, 47, 139, 187, 198, 212, 255, 263, 293, 295],
        "testWithXAI": [3, 19, 23, 26, 29, 66, 78, 97, 101, 114, 121, 150, 184, 207, 232, 267, 281, 282],
        "testWithoutXAI": [4, 20, 49, 50, 89, 94, 103, 111, 120, 224, 240, 242, 243, 268, 269, 290, 291, 298],
    },
]

mushroom_blocks = [
    {
        "train": [4, 74, 78, 108, 126, 195, 237, 330, 383, 393],
        "testWithXAI": [20, 22, 40, 54, 64, 81, 101, 106, 112, 125, 138, 245, 247, 297, 312, 319, 323, 352],
        "testWithoutXAI": [59, 61, 107, 141, 154, 173, 192, 198, 208, 219, 256, 274, 299, 316, 325, 329, 365, 390],
    },
    {
        "train": [10, 50, 96, 116, 134, 170, 224, 250, 359, 399],
        "testWithXAI": [52, 58, 91, 159, 181, 184, 202, 207, 226, 244, 260, 270, 271, 317, 321, 345, 350, 356],
        "testWithoutXAI": [37, 49, 66, 95, 120, 135, 143, 150, 155, 206, 213, 218, 228, 234, 279, 282, 366, 392],
    },
]

# -------------------------------
# 3) Config
# -------------------------------
DATASETS = {
    "Forest Cover": {"appId": "forest_cover", "model": "xgboost", "expMethod": "shap", "blocks": forest_blocks},
    "Wine Quality": {"appId": "wine_quality", "model": "mlp", "expMethod": "lime", "blocks": wine_blocks},
    # "Adult Income": {"appId": "adult", "model": "xgboost", "expMethod": "lime", "blocks": adult_blocks},
    # "Mushrooms": {"appId": "mushrooms", "model": "mlp", "expMethod": "shap", "blocks": mushroom_blocks},
}

STRATEGIES = {
    "Attribution Sum": AttributionSum,
    "Sensitive-features categorization": SensitiveFeatures,
    "Salient-features categorization": SalientFeatures,
    "Importance categorization": ImportanceCategorization,
}

STRATS_BY_XAI = {
    "None": ["Sensitive-features categorization"],  # keep empty to skip, matching your earlier logic
    "Importance": [
        "Sensitive-features categorization",
        "Salient-features categorization",
        "Importance categorization",
        "Attribution Sum",
    ],
    "Attribution": [
        "Sensitive-features categorization",
        "Attribution Sum",
    ],
}

# Strategy-specific skip rules you used earlier:
def should_skip_condition(xai_type: str, strat_name: str, param_regime: str) -> bool:
    # (1) Importance categorization w/o XAI
    if strat_name == "Importance categorization" and param_regime == "w/o XAI":
        return True
    # (2) Sensitive-features categorization w/ XAI if XAI type is Attribution
    if xai_type == "Attribution" and strat_name == "Sensitive-features categorization" and param_regime == "w/ XAI":
        return True
    return False

# Training-size sweep: 1..15 step ~3
TRAIN_SIZES = [1, 4, 7, 10, 13, 16]  # edit freely (e.g., include 15 if you want)

# Total simulated participants per (dataset, xai_type, param_regime, n_train)
TOTAL_PARTICIPANTS_PER_REGIME = 100  # scale overall N here

# Output
out_path = Path("results/CoAX simulated data (freq-matched, train-sweep).csv")
save_every = 20_000

# -------------------------------
# 4) CSV writing
# -------------------------------
CSV_COLUMNS = [
    "Participant Id",
    "appId",
    "Strategy",
    "k",
    "decay_param",
    "sensitivity",
    "retrieval_threshold",
    "scaling_factor",
    "N_train",
    "Instance Index",
    "trialType",
    "Tested w/ XAI",
    "predicted",
    "Prob correct",
    "Prob 0",
    "Prob 1",
    "Correct",
    "XAIType",
    "Agent",
    "Step",
    "expMethod",
]

def save_rows_to_csv(rows, path: Path, write_header: bool):
    if not rows:
        return
    df_out = pd.DataFrame(rows)
    for col in CSV_COLUMNS:
        if col not in df_out.columns:
            df_out[col] = np.nan
    df_out = df_out[CSV_COLUMNS]
    df_out.to_csv(path, mode="a", header=write_header, index=False)

# -------------------------------
# 5) Helpers
# -------------------------------
def make_mask(df: pd.DataFrame, **wanted):
    """True when every present column matches the requested value; ignore absent cols."""
    mask = pd.Series(True, index=df.index)
    for col, val in wanted.items():
        if col in df.columns:
            mask &= (df[col] == val)
    return mask

def sample_config_row(df_params: pd.DataFrame, strategy_name: str, tested_with_xai_value: str, appId: str, xai_type: str):
    """
    Sample ONE row for a participant config.

    - Try (Strategy == strategy_name) & (Tested w/ XAI == tested_with_xai_value) & (appId == appId) & (XAIType == xai_type)
    - If empty, relax Tested w/ XAI constraint.
    - Resample until required fields are non-null.
    """
    while True:
        sub = df_params[
            (df_params["Strategy"] == strategy_name) &
            (df_params["Tested w/ XAI"] == tested_with_xai_value) &
            (df_params["appId"] == appId) &
            (df_params["XAIType"] == xai_type)
        ]
        if sub.empty:
            sub = df_params[
                (df_params["Strategy"] == strategy_name) &
                (df_params["appId"] == appId) &
                (df_params["XAIType"] == xai_type)
            ]
            # keep the warning minimal (can be noisy at scale)
            # print(f"⚠️ no rows for Strategy='{strategy_name}' with Tested w/ XAI='{tested_with_xai_value}'; relaxing constraint.")
        if sub.empty:
            raise ValueError(f"No rows found in df_params for Strategy='{strategy_name}' (appId={appId}, XAIType={xai_type}).")

        row = sub.sample(1).iloc[0]

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

def strategy_percentages(df_params: pd.DataFrame, *, appId: str, xai_type: str, param_regime: str, eligible_strategies: list[str]):
    """
    Compute strategy proportions for (appId, XAIType, Tested w/ XAI=param_regime).
    Returns dict {strategy: p} summing to 1. Falls back to uniform if empty.
    """
    sub = df_params[
        (df_params["appId"] == appId) &
        (df_params["XAIType"] == xai_type) &
        (df_params["Tested w/ XAI"] == param_regime)
    ]
    counts = sub["Strategy"].value_counts().reindex(eligible_strategies).fillna(0).astype(int)
    if counts.sum() == 0:
        p = 1.0 / len(eligible_strategies)
        return {s: p for s in eligible_strategies}
    props = (counts / counts.sum()).astype(float)
    return {s: float(props.loc[s]) for s in eligible_strategies}

def allocate_from_percentages(perc: dict, total_n: int):
    """
    Convert percentages to integer allocation summing to total_n using largest remainder.
    """
    keys = list(perc.keys())
    p = np.array([perc[k] for k in keys], dtype=float)
    s = p.sum()
    if s <= 0:
        base = total_n // len(keys)
        rem = total_n - base * len(keys)
        alloc = {k: base for k in keys}
        for k in keys[:rem]:
            alloc[k] += 1
        return alloc

    p = p / s
    raw = p * total_n
    floored = np.floor(raw).astype(int)
    alloc = {k: int(floored[i]) for i, k in enumerate(keys)}
    remainder = total_n - sum(alloc.values())
    if remainder > 0:
        frac = raw - floored
        order = np.argsort(-frac)
        for idx in order[:remainder]:
            alloc[keys[idx]] += 1
    return alloc

# def get_instance_pool(df_values: pd.DataFrame, df_metadata: pd.DataFrame, *, appId: str, modelName: str, expMethod: str):
#     """
#     Return a list of instance IDs available for sampling training trials.

#     Tries:
#     - df_values filtered by (appId/modelName/expMethod) if columns exist
#     - id column preference: 'instance_id' > 'Instance Index' > index
#     Falls back to df_metadata similarly if needed.
#     """
#     # values
#     dfv = df_values
#     maskv = make_mask(dfv, appId=appId, modelName=modelName, expMethod=expMethod)
#     dfv = dfv[maskv]

#     if "instance_id" in dfv.columns:
#         pool = dfv["instance_id"].dropna().unique().tolist()
#     elif "Instance Index" in dfv.columns:
#         pool = dfv["Instance Index"].dropna().unique().tolist()
#     else:
#         pool = dfv.index.tolist()

#     if len(pool) > 0:
#         return pool

#     # metadata fallback
#     dfm = df_metadata
#     maskm = make_mask(dfm, appId=appId, modelName=modelName, expMethod=expMethod)
#     dfm = dfm[maskm]

#     if "instance_id" in dfm.columns:
#         pool = dfm["instance_id"].dropna().unique().tolist()
#     elif "Instance Index" in dfm.columns:
#         pool = dfm["Instance Index"].dropna().unique().tolist()
#     else:
#         pool = dfm.index.tolist()

#     return pool

def make_session_variable_train(blocks, *, instance_pool: list[int], n_train: int, train_with_explanation: bool):
    """
    For each block:
    - sample n_train training instances from instance_pool (excluding that block's test ids if possible)
    - append tests from block['testWithXAI'] and block['testWithoutXAI'] (shuffled)

    Returns:
    - session: list of dicts
    - session_meta: dict instance_id -> (is_training, with_explanation)
    """
    session = []
    session_meta = {}

    pool_set = set(range(1, 120))

    for block in blocks:
        test_ids = set(block.get("testWithXAI", [])) | set(block.get("testWithoutXAI", []))
        eligible_pool = list(pool_set - test_ids)
        if len(eligible_pool) < n_train:
            eligible_pool = list(pool_set)  # allow overlap if needed

        if len(eligible_pool) < n_train:
            raise ValueError(f"Not enough instances to sample n_train={n_train}. pool={len(eligible_pool)}")

        train_ids = random.sample(eligible_pool, n_train)

        for i in train_ids:
            session.append({"instance_id": i, "is_training": True, "with_explanation": bool(train_with_explanation)})
            session_meta[i] = (True, bool(train_with_explanation))

        for i in random.sample(block["testWithXAI"], len(block["testWithXAI"])):
            session.append({"instance_id": i, "is_training": False, "with_explanation": True})
            session_meta[i] = (False, True)

        for i in random.sample(block["testWithoutXAI"], len(block["testWithoutXAI"])):
            session.append({"instance_id": i, "is_training": False, "with_explanation": False})
            session_meta[i] = (False, False)

    return session, session_meta

# -------------------------------
# 6) Core runner (one participant, one regime, one n_train)
# -------------------------------
def run_one_param_regime(
    *,
    ds_cfg,
    xai_type: str,
    strat_name: str,
    StratCls,
    ai_loader,
    participant_idx: int,
    param_regime: str,           # "w/ XAI" or "w/o XAI"
    train_with_explanation: bool,
    n_train: int,
    instance_pool: list[int],
    rows_out: list[dict],
    expMethod = "lime",
):
    """
    Runs simulation and appends ONLY TEST infer rows matching param_regime.
    Returns:
    - kept: number of rows appended
    - -1 if skipped condition
    """
    if should_skip_condition(xai_type, strat_name, param_regime):
        return -1

    # cfg_row = sample_config_row(df_params, strat_name, param_regime, ds_cfg["appId"], xai_type)

    # config = {
    #     "k": int(cfg_row["k"]),
    #     "retrieval_threshold": float(cfg_row["retrieval_threshold"]),
    #     "decay_param": 0.5,
    # }
    # if strat_name == "Attribution Sum":
    #     config["scaling_factor"] = float(cfg_row["scaling_factor"]) + 2
    #     config["explanation_type"] = xai_type.lower()
    # else:
    #     config["sensitivity"] = float(cfg_row["sensitivity"])

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



    session, session_meta = make_session_variable_train(
        ds_cfg["blocks"],
        instance_pool=None,
        n_train=n_train,
        train_with_explanation=train_with_explanation,
    )

    strategy = StratCls(**config)
    runner = StrategyComparisonRunner(strategy, ai_loader, UI())
    logs = runner.generalized_run_experiment(session)

    kept = 0
    for lg in logs:
        inst_id = lg["instance_id"]
        is_tr, _with_xai = session_meta[inst_id]

        if lg.get("Step") != "infer" or is_tr:
            continue

        trial_tested = "w/ XAI" if lg.get("explanation") else "w/o XAI"
        if trial_tested != param_regime:
            continue

        probs = lg["response"]
        truth = lg["ai_prediction"]

        keys = np.array(list(probs.keys()))
        weights = np.array(list(probs.values()))
        choice = np.random.choice(keys, p=weights / weights.sum())

        rows_out.append({
            "Participant Id": participant_idx,
            "appId": ds_cfg["appId"],
            "XAIType": xai_type,
            "Strategy": strat_name,
            **config,
            "N_train": int(n_train),
            "Instance Index": inst_id,
            "trialType": "Test",
            "Tested w/ XAI": trial_tested,
            "predicted": choice,
            "Prob correct": probs.get(truth, 0.0),
            "Prob 0": probs.get(0, np.nan),
            "Prob 1": probs.get(1, np.nan),
            "Correct": int(choice == truth),
            "Agent": "CoAX",
            "Step": lg.get("Step", "infer"),
            "expMethod": expMethod
        })
        kept += 1

    return kept

# -------------------------------
# 7) Main sweep (freq-matched + train sweep)
# -------------------------------
rows = []
open(out_path, "w").close()
first_write = True
participant_counter = 0



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


for ds_name, ds_cfg in DATASETS.items():
    print(f"[RUN] Dataset ➜ {ds_name}")

    # Create a reusable instance pool for training sampling
    # instance_pool = get_instance_pool(
    #     df_values,
    #     df_metadata,
    #     appId=ds_cfg["appId"],
    #     modelName=ds_cfg["model"],
    #     expMethod=ds_cfg["expMethod"],
    # )
    # if len(instance_pool) == 0:
    #     raise ValueError(f"Empty instance_pool for {ds_cfg}")

    for expMethod in ["shap"]:
        for xai_type in ("None", "Importance", "Attribution"):
            if expMethod!="shap" and xai_type=="None":
                continue
            eligible = STRATS_BY_XAI.get(xai_type, [])
            if not eligible:
                print(f"  [RUN] XAI-type ➜ {xai_type} (no eligible strategies; skipping)")
                continue
            print(f"  [RUN] XAI-type ➜ {xai_type}")

            # Load explanation file for this xai_type
            file_expl = os.path.join(data_dir, f"{'attribution' if xai_type == 'Attribution' else 'importance'}.csv")
            df_expl = pd.read_csv(file_expl)

            ai_loader = (
                AIDatasetLoader(
                    feature_values_df=df_values,
                    metadata_df=df_metadata,
                    explanation_values_df=df_expl,
                    explanation_columns=["a0_i", "a1_i", "a2_i", "a3_i", "a4_i"],
                )
                .filter_loader(
                    lambda dff,
                        app=ds_cfg["appId"],
                        mdl=ds_cfg["model"],
                        exp=expMethod:
                    make_mask(dff, appId=app, modelName=mdl, expMethod=exp)
                )
            )

            for n_train in TRAIN_SIZES:
                print(f"    [RUN] N_train ➜ {n_train}")

                for param_regime in ("w/ XAI", "w/o XAI"):
                    if param_regime == "w/ XAI" and xai_type == "None":
                        continue
                    perc = strategy_percentages(
                        df_params,
                        appId=ds_cfg["appId"],
                        xai_type=xai_type,
                        param_regime=param_regime,
                        eligible_strategies=eligible,
                    )
                    alloc = allocate_from_percentages(perc, TOTAL_PARTICIPANTS_PER_REGIME)

                    print(f"      [RUN] Regime ➜ {param_regime} | Allocation: {alloc}")

                    for strat_name, n_part in alloc.items():
                        StratCls = STRATEGIES[strat_name]

                        for _ in range(n_part):
                            participant_counter += 1

                            kept = run_one_param_regime(
                                ds_cfg=ds_cfg,
                                xai_type=xai_type,
                                strat_name=strat_name,
                                StratCls=StratCls,
                                ai_loader=ai_loader,
                                participant_idx=participant_counter,
                                param_regime=param_regime,
                                train_with_explanation=True,
                                n_train=n_train,
                                instance_pool=None,
                                rows_out=rows,
                                expMethod=expMethod,
                            )

                            if kept == -1:
                                continue

                            # If nothing matched, relax training-with-XAI flag to match regime
                            if kept == 0:
                                _ = run_one_param_regime(
                                    ds_cfg=ds_cfg,
                                    xai_type=xai_type,
                                    strat_name=strat_name,
                                    StratCls=StratCls,
                                    ai_loader=ai_loader,
                                    participant_idx=participant_counter,
                                    param_regime=param_regime,
                                    train_with_explanation=(param_regime == "w/ XAI"),
                                    n_train=n_train,
                                    instance_pool=None,
                                    rows_out=rows,
                                    expMethod=expMethod,
                                )

                            if len(rows) >= save_every:
                                save_rows_to_csv(rows, out_path, write_header=first_write)
                                first_write = False
                                rows.clear()

# Final flush
save_rows_to_csv(rows, out_path, write_header=first_write)
print(f"✅ finished – data stored in {out_path.resolve()}")
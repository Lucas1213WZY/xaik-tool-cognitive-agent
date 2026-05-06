#!/usr/bin/env python3
"""Flask server for the CoAX interactive demo.

Usage (from repo root):
    conda run -n xaikit-coax python demo/server.py
    # Then open http://localhost:5000/demo/
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, request, send_file, send_from_directory

from src.cognitive_models import initialize_strategies
from src.virtual_experiment_executor.coax_executor import (
    DATASETS,
    DEFAULT_ASSETS_ROOT,
    STRATEGY_NAME_MAP,
    VirtualExperimentConfig,
    load_coax_loader,
    make_session,
    simulate_candidate,
    simulate_virtual_experiment,
)

DEMO_DIR = Path(__file__).parent

app = Flask(__name__)
initialize_strategies()

# Strategies available per XAI type (in display order)
STRATEGIES_FOR_XAI: dict[str, list[str]] = {
    "Importance": [
        "Sensitive-features categorization",
        "Salient-features categorization",
        "Importance categorization",
    ],
    "Attribution": [
        "Sensitive-features categorization",
        "Attribution Sum",
    ],
    "None": ["Sensitive-features categorization"],
}


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------

@app.route("/")
def root():
    return redirect("/demo/")


@app.route("/demo/")
def demo_index():
    return send_file(DEMO_DIR / "index.html")


@app.route("/demo/<path:path>")
def demo_static(path: str):
    return send_from_directory(str(DEMO_DIR), path)


@app.route("/assets/<path:path>")
def assets_static(path: str):
    return send_from_directory(str(REPO_ROOT / "assets"), path)


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _build_param_row(strategy_name: str, cog: dict, xai_type: str) -> pd.Series:
    return pd.Series({
        "Strategy": strategy_name,
        "k": int(round(float(cog.get("k", 3)))),
        "decay_param": float(cog.get("decay_param", 0.5)),
        "sensitivity": float(cog.get("sensitivity", 50.0)),
        "retrieval_threshold": float(cog.get("retrieval_threshold", -2.5)),
        "scaling_factor": float(cog.get("scaling_factor", 1.0)),
        "explanation_type": cog.get("explanation_type", xai_type.lower()),
        "NLL": np.nan,
        "Session": np.nan,
    })


def _summary(df: pd.DataFrame) -> list[dict]:
    rows = []
    for (strategy, condition, trial_type), g in df.groupby(
        ["Strategy", "Tested w/ XAI", "trialType"]
    ):
        n = len(g)
        correct = int(g["Correct"].sum())
        rows.append({
            "strategy": strategy,
            "condition": condition,
            "trial_type": trial_type,
            "accuracy": round(correct / n, 4) if n > 0 else 0.0,
            "n_trials": n,
            "n_correct": correct,
        })
    return rows


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.route("/api/strategies")
def api_strategies():
    xai_type = request.args.get("xai_type", "Importance")
    xai_cap = xai_type[:1].upper() + xai_type[1:].lower()
    return jsonify(STRATEGIES_FOR_XAI.get(xai_cap, list(STRATEGY_NAME_MAP.keys())))


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    data = request.get_json(force=True)
    app_id: str = data.get("app_id", "adult")
    raw_xai: str = data.get("xai_type", "Importance")
    xai_type: str = raw_xai[:1].upper() + raw_xai[1:].lower()
    seed: int = int(data.get("seed", 7))
    mode: str = data.get("mode", "custom")

    if app_id not in DATASETS:
        return jsonify({"error": f"Unknown app_id: {app_id}"}), 400

    all_rows: list[dict] = []

    if mode == "from_csv":
        max_p = int(data.get("max_participants", 5))
        for condition in ["w/ XAI", "w/o XAI"]:
            cfg = VirtualExperimentConfig(
                app_id=app_id,
                xai_type=xai_type,
                tested_with_xai=condition,
                max_participants=max_p,
                seed=seed,
            )
            part_df = simulate_virtual_experiment(cfg)
            if not part_df.empty:
                all_rows.extend(part_df.to_dict(orient="records"))

    else:
        cog = data.get("cognitive_parameters", {})
        strategies: list[str] = data.get(
            "strategies", STRATEGIES_FOR_XAI.get(xai_type, list(STRATEGY_NAME_MAP.keys()))
        )
        loader = load_coax_loader(str(DEFAULT_ASSETS_ROOT), app_id, xai_type)

        for condition in ["w/ XAI", "w/o XAI"]:
            session_rng = random.Random(seed)
            session = make_session(
                DATASETS[app_id]["blocks"], session_rng, xai_type != "None"
            )
            for strategy_name in strategies:
                if strategy_name not in STRATEGY_NAME_MAP:
                    continue
                param_row = _build_param_row(strategy_name, cog, xai_type)
                sim_rng = np.random.default_rng(seed)
                result = simulate_candidate(
                    loader=loader,
                    session=session,
                    param_row=param_row,
                    participant_id=1,
                    app_id=app_id,
                    xai_type=xai_type,
                    tested_with_xai=condition,
                    rng=sim_rng,
                )
                for row in result.rows:
                    row["Selected Strategy"] = ""
                    row["Selected Strategy NLL"] = np.nan
                    row["Strategy NLL"] = result.nll
                all_rows.extend(result.rows)

    if not all_rows:
        return jsonify({"summary": [], "trials": [], "total_rows": 0})

    df = pd.DataFrame(all_rows)
    summary = _summary(df)
    trials = (
        df.replace({np.nan: None})
        .head(1000)
        .to_dict(orient="records")
    )

    return jsonify({"summary": summary, "trials": trials, "total_rows": len(df)})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    print(f"\n  CoAX demo →  http://localhost:{port}/demo/\n")
    app.run(port=port, debug=False)

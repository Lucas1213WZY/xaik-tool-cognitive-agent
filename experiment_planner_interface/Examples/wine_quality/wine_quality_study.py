"""
wine_quality_study.py
=====================
XAI evaluation experiment using the wine_quality.csv dataset.

Input format
------------
The input file follows the fixed xai_tester convention:

    y       — AI prediction label  (0 = Bad Quality, 1 = Good Quality)
    v0–v5   — raw feature values   (Vinegar Taint, SO2, pH,
                                     Sulphates, Alcohol, Others)
    a0–a5   — SHAP attribution scores, one per feature

This maps directly to the .numbers file supplied (input_values.numbers),
which uses the same column layout.  Export that file to CSV from Numbers
(File → Export To → CSV) and pass it as the filepath below.

Usage
-----
    python wine_quality_study.py
    python wine_quality_study.py --csv path/to/input_values.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import xai_tester
from xai_tester import control, design, io, misc

# ── Feature names matching v0–v5 order ──────────────────────────────────────
FEATURE_NAMES = [
    "Vinegar Taint",   # v0 — volatile acidity
    "SO2",             # v1 — free sulfur dioxide
    "pH",              # v2 — pH
    "Sulphates",       # v3 — sulphates
    "Alcohol",         # v4 — alcohol content
]

# ── Map integer y codes to display labels ────────────────────────────────────
LABEL_MAP = {
    0: "Type1",
    1: "Type2",
}

# ── CLI: optionally pass a custom CSV path ───────────────────────────────────
parser = argparse.ArgumentParser(description="Wine Quality XAI Study")
parser.add_argument(
    "--csv",
    default=str(Path(__file__).parent / "wine_quality.csv"),
    help="Path to input CSV (y, v0–v4 a0–a4 columns)",
)
args = parser.parse_args()


# ── 0. CONFIGURE DEFAULTS ────────────────────────────────────────────────────

misc.defaults.instructions_text = """
WINE QUALITY — XAI EVALUATION STUDY
=====================================
You will review wines evaluated by an AI quality-prediction model.

For each wine you will see:
  • The wine's chemical measurements  (Vinegar Taint, SO2, pH, etc.)
  • The AI's quality prediction       (Good Quality / Bad Quality)
  • An XAI explanation                (why the AI made that prediction)

The XAI bar chart shows SHAP feature attributions:
  ████ green bars  → feature pushed AI toward Type1
  ████ red bars    → feature pushed AI toward Type2
  Longer bar = stronger influence on the prediction.

Your task: review all of the above, then predict what label the AI assigned.
Type your answer exactly and press ENTER:

        [Type1]   or   [Type2]
"""

misc.defaults.goodbye_text          = "Session complete — thank you!"
misc.defaults.data_directory        = "xai_results"
misc.defaults.response_timeout      = None    # unlimited
misc.defaults.pause_between_trials  = 0.4
misc.defaults.importance_bar_width  = 28


# ── 1. BUILD THE EXPERIMENT ──────────────────────────────────────────────────

exp = design.Experiment(
    name             = "WineQuality_XAI",
    labels           = list(LABEL_MAP.values()),  
    randomise_trials = True,
    
)

# load_csv_fixed handles the y / v0–v5 / a0–a5 column convention directly.
# feature_names maps v0→"Vinegar Taint", v1→"SO2", … so participants see
# readable labels instead of generic column codes.
exp.load_csv_fixed(
    filepath      = args.csv,
    feature_names = FEATURE_NAMES,
    label_map     = LABEL_MAP,
    n_features    = 5,
    n_trials = 10,
)

print(f"\nLoaded {exp.n_trials} wine trials.")
print(f"Labels: {exp.labels}")


# ── 2. INITIALISE ────────────────────────────────────────────────────────────

control.initialise(exp)


# ── 3. START ─────────────────────────────────────────────────────────────────

control.start()     # prompts for participant ID, shows instructions


# ── 4. TRIAL LOOP ────────────────────────────────────────────────────────────

for i, trial in enumerate(exp.session.trials, start=1):

    # Timestamp when trial appears on screen
    trial.presented_at = exp.clock.session_time

    # Display: original data + AI prediction + XAI bars
    io.present_trial(
        trial,
        trial_number = i,
        total_trials = exp.session.n_trials,
        labels       = exp.labels,
        bar_width    = misc.defaults.importance_bar_width,
    )

    # Collect response and measure reaction time
    exp.clock.reset()
    response, rt = io.get_response(
        trial,
        timeout = misc.defaults.response_timeout,
    )

    # Record — appends a CSV row immediately
    exp.data.record(trial, response, rt)

    # # Optional mid-session break
    # if i == exp.session.n_trials // 2:
    #     control.pause(
    #         "Halfway through!\n\n"
    #         "Take a short break if you need one,\n"
    #         "then press ENTER to continue."
    #     )

    # Brief blank pause between trials
    if i < exp.session.n_trials:
        io.presenter.show_pause(misc.defaults.pause_between_trials)


# ── 5. END ───────────────────────────────────────────────────────────────────

control.end(show_summary=True)

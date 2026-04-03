"""
loan_approval_study.py
======================
A complete XAI evaluation experiment using the xai_tester API.

Scenario
--------
A bank has deployed an AI model that approves or rejects loan applications.
The model uses a LIME explainer to generate per-feature importance scores.

In this study, participants see each application's data, the AI's prediction,
the real-world ground truth outcome, and the LIME explanation. Their task is
to predict the AI's label — allowing researchers to measure how well the XAI
explanation helps participants understand and anticipate the model's decisions.

Dataset
-------
loan_data.csv — 12 synthetic loan applications with 6 features:
  age, income_k, credit_score, employment_years, loan_amount_k, debt_ratio

  ai_label      : what the AI predicted     (shown as "AI Prediction")
  ground_truth  : real-world outcome         (shown as "Ground Truth")
  xai_<feature> : LIME importance scores    (shown as importance bars)

The AI uses credit_score + debt_ratio to decide; the ground truth also
requires employment_years >= 5, so the AI makes occasional errors.

Usage
-----
    python loan_approval_study.py
"""

import sys
import os
from pathlib import Path

# ── Make sure xai_tester is importable from this folder ────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import xai_tester
from xai_tester import control, design, io, misc


# ── 0. CONFIGURE DEFAULTS ──────────────────────────────────────────────────
# All of these are optional — shown here for clarity.

misc.defaults.instructions_text = """
LOAN APPROVAL — XAI EVALUATION STUDY
=====================================
You will review loan applications evaluated by an AI model.

For each case you will see:
  • The applicant's data    (age, income, credit score, etc.)
  • The AI's prediction     (what the model decided: Approved / Rejected)
  • The ground truth        (the real-world outcome — did the AI get it right?)
  • An XAI explanation      (why the AI made that decision)

The bar chart shows LIME feature importance:
  ████ green bars  → feature pushed the AI toward Approval
  ████ red bars    → feature pushed the AI toward Rejection
  Longer bar = stronger influence.

Your task: study all of the above, then predict what label the AI assigned.
Type your answer and press ENTER.

        [Approved]   or   [Rejected]
"""

misc.defaults.goodbye_text       = "Session complete — thank you for participating!"
misc.defaults.data_directory     = "xai_results"
misc.defaults.response_timeout   = None   # None = wait as long as needed
misc.defaults.pause_between_trials = 0.4  # seconds of blank screen between trials
misc.defaults.importance_bar_width = 30


# ── 1. BUILD THE EXPERIMENT ────────────────────────────────────────────────

exp = design.Experiment(
    name             = "LoanApproval_XAI",
    labels           = ["Approved", "Rejected"],
    randomise_trials = True,    # shuffle so order doesn't bias results
)

exp.load_csv(
    filepath         = Path(__file__).parent / "loan_data.csv",
    ai_label_col     = "ai_label",
    ground_truth_col = "ground_truth",
    xai_cols         = [
        "xai_age",
        "xai_income_k",
        "xai_credit_score",
        "xai_employment_years",
        "xai_loan_amount_k",
        "xai_debt_ratio",
    ],
    # feature_cols inferred automatically as:
    # age, income_k, credit_score, employment_years, loan_amount_k, debt_ratio
)

print(f"\nLoaded {exp.n_trials} trials.")


# ── 2. INITIALISE ─────────────────────────────────────────────────────────
# Validates config and starts the session clock.

control.initialise(exp)


# ── 3. START ──────────────────────────────────────────────────────────────
# Prompts for participant ID, shows instructions, opens output files.
# Pass participant_id="P01" to skip the input prompt.

control.start()
# After this call:
#   exp.session  → Session with ordered list of trials
#   exp.data     → DataRecorder (CSV + summary)
#   exp.clock    → Clock tracking session time


# ── 4. TRIAL LOOP ─────────────────────────────────────────────────────────

for i, trial in enumerate(exp.session.trials, start=1):

    # Record the clock time when this trial appears on screen
    trial.presented_at = exp.clock.session_time

    # Render: original data table + XAI importance bar chart
    io.present_trial(
        trial,
        trial_number = i,
        total_trials = exp.session.n_trials,
        labels       = exp.labels,
        bar_width    = misc.defaults.importance_bar_width,
    )

    # Collect free-text response and measure reaction time
    exp.clock.reset()
    response, rt = io.get_response(
        trial,
        timeout = misc.defaults.response_timeout,
    )

    # Persist: appends one row to the CSV immediately
    exp.data.record(trial, response, rt)

    # Optional mid-session break at the halfway point
    if i == exp.session.n_trials // 2:
        control.pause(
            "You're halfway through!\n\n"
            "Take a short break if you need one,\n"
            "then press ENTER to continue."
        )

    # Brief blank screen between trials
    if i < exp.session.n_trials:
        io.presenter.show_pause(misc.defaults.pause_between_trials)


# ── 5. END ────────────────────────────────────────────────────────────────
# Writes the summary report, prints output file paths, shows goodbye screen.

control.end(show_summary=True)

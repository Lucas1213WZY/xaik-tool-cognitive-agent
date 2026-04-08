"""XAI Tester — A Python library for human-subject XAI evaluation experiments.

XAI Tester provides a structured, script-based API for running experiments in
which human participants review original data alongside XAI feature-importance
explanations, then predict what label an AI model assigned to each case.

The design mirrors the hierarchical style of Expyriment:

    Experiment → Session → Trial

Typical usage::

    import xai_tester

    exp = xai_tester.design.Experiment(
        name="LIME Evaluation",
        labels=["Approved", "Rejected"],
    )
    exp.load_csv("data.csv",
                 ai_label_col="ai_label",
                 xai_cols=["feat_age", "feat_income", "feat_score"])

    xai_tester.control.initialise(exp)
    xai_tester.control.start(participant_id="P01")

    for trial in exp.session.trials:
        xai_tester.io.present_trial(trial)
        response = xai_tester.io.get_response(trial)
        exp.data.record(trial, response)

    xai_tester.control.end()

Website: https://github.com/your-org/xai-tester
"""

__version__ = "0.1.0"
__author__  = "Your Name <you@example.com>"

from . import control, design, io, misc

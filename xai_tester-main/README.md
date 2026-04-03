# XAI Tester

A Python library for running human-subject XAI (Explainable AI) evaluation experiments.

Participants are shown original data records alongside feature-importance
explanations (e.g. LIME, SHAP scores), then asked to predict what label the
AI model assigned to each case.  The library handles the full session
lifecycle, response collection, and output file generation.

## Requirements

- Python ≥ 3.10
- Standard library only (no third-party dependencies)

## Installation

```bash
# From source
pip install -e .
```

## Quick Start

```python
import xai_tester
from xai_tester import control, design, io

exp = design.Experiment(name="LIME Study", labels=["Approved", "Rejected"])
exp.load_csv("data.csv",
             ai_label_col="ai_label",
             xai_cols=["xai_age", "xai_income", "xai_score"])

control.initialise(exp)
control.start(participant_id="P01")

for i, trial in enumerate(exp.session.trials, start=1):
    trial.presented_at = exp.clock.session_time
    io.present_trial(trial, trial_number=i,
                     total_trials=exp.session.n_trials,
                     labels=exp.labels)
    response, rt = io.get_response(trial)
    exp.data.record(trial, response, rt)

control.end()
```

## Input CSV Format

| Column type    | Description                                      |
|----------------|--------------------------------------------------|
| Feature cols   | Any columns shown as original data to participants |
| `ai_label_col` | AI model's prediction — hidden during session     |
| XAI cols       | Feature-importance scores (floats, one per feature)|

Example:

```
age,income,credit_score,ai_label,xai_age,xai_income,xai_credit_score
34,72000,680,Approved,0.12,-0.34,0.55
...
```

## Output Files

Two files are written to `xai_results/` (configurable):

- `<experiment>_<participant>_<timestamp>.csv` — one row per trial with
  response, correctness flag, reaction time, all features and XAI scores.
- `<experiment>_<participant>_<timestamp>_summary.txt` — accuracy,
  response-time statistics, per-label breakdown, top features by importance.

## Project Structure

```
xai_tester/
├── control/              # Lifecycle: initialise, start, pause, end
├── design/               # Experiment → Session → Trial hierarchy
├── io/                   # Presenter (terminal UI) + DataRecorder
└── misc/                 # Clock, defaults
tests/
    test_xai_tester.py    # 33 unit + integration tests
example_experiment.py     # Runnable demo with synthetic data
```

## Running Tests

```bash
python -m unittest tests/test_xai_tester.py -v
```

## Customising Defaults

All settings live in `xai_tester.misc.defaults` and can be overridden before
calling `control.initialise()`:

```python
xai_tester.misc.defaults.response_timeout      = 30      # seconds
xai_tester.misc.defaults.pause_between_trials  = 0.5
xai_tester.misc.defaults.data_directory        = "results"
xai_tester.misc.defaults.instructions_text     = "Your custom instructions..."
xai_tester.misc.defaults.importance_bar_width  = 40
```

## Swapping the Display Backend

Replace the terminal presenter with your own subclass:

```python
from xai_tester.io._presenter import BasePresenter
import xai_tester.io as xio

class WebPresenter(BasePresenter):
    def show_trial(self, trial, trial_number, total_trials, labels, bar_width):
        ...  # serve a Flask page
    def get_response(self, trial, timeout):
        ...  # read from HTTP POST
    def show_message(self, text):
        ...
    def clear(self):
        ...

xio.presenter = WebPresenter()
```

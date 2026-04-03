"""
Default settings for XAI Tester.

All optional parameters across the library fall back to values defined here.
Override any default before calling ``control.initialise()``::

    import xai_tester
    xai_tester.misc.defaults.response_timeout = 60
    xai_tester.misc.defaults.display_mode = "terminal"

Attributes
----------
display_mode : str
    How trials are rendered to the participant.
    ``"terminal"``  — rich terminal UI (default, no GUI needed)
    ``"web"``       — local Flask web page (future)

response_timeout : int or None
    Maximum seconds a participant has to enter a response per trial.
    ``None`` means wait indefinitely.

show_progress : bool
    Whether to show a progress indicator (e.g. "Trial 3 / 20").

randomise_trials : bool
    Whether to shuffle trial order when loading from CSV.

pause_between_trials : float
    Seconds of blank pause shown between trials.

instructions_text : str
    Text shown on the instructions screen before the session begins.

ready_prompt : str
    Prompt shown after instructions, waiting for ENTER to continue.

goodbye_text : str
    Text shown at the end of the session.

data_directory : str
    Folder where output CSV and summary report are written.

csv_delimiter : str
    Delimiter used when writing the output data CSV.

importance_bar_width : int
    Number of characters used for the ASCII importance bar in terminal mode.

importance_positive_colour : str
    Rich markup colour for positive feature importance values.

importance_negative_colour : str
    Rich markup colour for negative feature importance values.

summary_decimal_places : int
    Decimal places used when formatting numbers in the summary report.
"""

__author__ = "Your Name <you@example.com>"

# --- Display ---
display_mode: str = "terminal"          # "terminal" | "web"
show_progress: bool = True
pause_between_trials: float = 0.5      # seconds

# --- Timing ---
response_timeout: int | None = None    # seconds; None = unlimited

# --- Trial presentation ---
importance_bar_width: int = 30
importance_positive_colour: str = "green"
importance_negative_colour: str = "red"

# --- Text / prompts ---
instructions_text: str = (
    "In this study you will be shown data records together with an AI "
    "explanation.\n\n"
    "The explanation shows how much each feature pushed the AI toward or "
    "away from a particular prediction.\n\n"
    "Your task: read the data and the explanation, then TYPE your best "
    "guess for the AI's predicted label and press ENTER."
)
ready_prompt: str = "Press ENTER to begin the session."
goodbye_text: str = "Session complete. Thank you for participating!"

# --- Output ---
data_directory: str = "xai_results"
csv_delimiter: str = ","
summary_decimal_places: int = 3

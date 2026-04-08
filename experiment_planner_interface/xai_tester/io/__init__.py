"""The io package — trial presentation, response collection, data recording."""

from ._presenter import BasePresenter, TerminalPresenter
from ._data_recorder import DataRecorder

# Default presenter instance — swap this out to change display mode:
#   xai_tester.io.presenter = MyCustomPresenter()
presenter: BasePresenter = TerminalPresenter()


def present_trial(trial, trial_number: int, total_trials: int,
                  labels=None, bar_width: int = 30) -> None:
    """Render *trial* using the active presenter.

    Shows four sections:
    1. Progress header.
    2. Original feature data table.
    3. AI Prediction label (``trial.ai_label``) — displayed prominently
       so participants know what the model decided.
    4. XAI feature-importance bar chart.

    This is a convenience wrapper around ``xai_tester.io.presenter.show_trial``.
    Swap ``xai_tester.io.presenter`` to change the display backend.

    Parameters
    ----------
    trial:
        The :class:`~xai_tester.design.Trial` to display.
    trial_number:
        1-based index for the progress indicator.
    total_trials:
        Total trials in the session.
    labels:
        Optional list of known label strings shown as a hint.
    bar_width:
        Width of the ASCII importance bar in characters.
    """
    presenter.show_trial(
        trial,
        trial_number=trial_number,
        total_trials=total_trials,
        labels=labels,
        bar_width=bar_width,
    )


def get_response(trial, timeout=None) -> tuple[str, float]:
    """Prompt the participant for a response using the active presenter.

    Parameters
    ----------
    trial:
        The current :class:`~xai_tester.design.Trial`.
    timeout:
        Maximum seconds to wait; ``None`` = unlimited.

    Returns
    -------
    (response_str, reaction_time_seconds)
    """
    return presenter.get_response(trial, timeout=timeout)

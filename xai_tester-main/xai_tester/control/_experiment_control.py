"""Experiment lifecycle functions.

Four functions control the full lifecycle of a session:

    initialise(exp)   — validate config, set up clock and recorder
    start(...)        — register participant, open output files
    pause(...)        — mid-session pause screen
    end(...)          — write summary, show goodbye, clean up

These functions operate on the *active experiment* registered via
``initialise()``.  Only one experiment can be active at a time.
"""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING

from .. import io, misc
from ..io._data_recorder import DataRecorder
from ..design._structure import Session

if TYPE_CHECKING:
    from ..design._structure import Experiment


# ---------------------------------------------------------------------------
# Module-level active experiment reference
# ---------------------------------------------------------------------------

_active_exp: "Experiment | None" = None


def _get_active() -> "Experiment":
    if _active_exp is None:
        raise RuntimeError(
            "No active experiment. Call control.initialise(exp) first."
        )
    return _active_exp


# ---------------------------------------------------------------------------
# initialise
# ---------------------------------------------------------------------------

def initialise(experiment: "Experiment") -> "Experiment":
    """Initialise an experiment and register it as the active experiment.

    This is the first lifecycle call.  It:

    * Validates that at least one trial has been loaded.
    * Creates the session :class:`~xai_tester.misc.Clock`.
    * Registers *experiment* as the module-level active experiment so that
      subsequent calls to :func:`start`, :func:`pause`, and :func:`end` know
      which experiment to act on.

    Does **not** open any output files or prompt for participant info — that
    happens in :func:`start`.

    Parameters
    ----------
    experiment:
        The :class:`~xai_tester.design.Experiment` to initialise.

    Returns
    -------
    experiment
        The same object, for chaining.

    Raises
    ------
    RuntimeError
        If *experiment* has already been initialised.
    ValueError
        If no trials have been loaded (``exp.n_trials == 0``).

    Examples
    --------
    >>> exp = xai_tester.design.Experiment(name="Study")
    >>> exp.load_csv("data.csv", ai_label_col="label", xai_cols=["f1","f2"])
    >>> xai_tester.control.initialise(exp)
    """
    global _active_exp

    if experiment.is_initialised:
        raise RuntimeError(
            f"Experiment '{experiment.name}' is already initialised."
        )
    if experiment.n_trials == 0:
        raise ValueError(
            "No trials loaded. Call exp.load_csv() or exp.add_trial() "
            "before initialise()."
        )

    experiment._clock        = misc.Clock()
    experiment._is_initialised = True
    _active_exp              = experiment

    print(
        f"\n[xai_tester] Experiment '{experiment.name}' initialised — "
        f"{experiment.n_trials} trials loaded."
    )
    return experiment


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------

def start(
    participant_id: str | int | None = None,
    auto_id: bool = False,
    skip_instructions: bool = False,
) -> "Experiment":
    """Start the active experiment session.

    Prompts for a participant ID (unless *participant_id* or *auto_id* is
    set), displays the instructions screen, then creates the
    :class:`~xai_tester.io.DataRecorder` and :class:`~xai_tester.design.Session`.

    After this call the following are available on the experiment object:

    * ``exp.session``  — the active :class:`~xai_tester.design.Session`
    * ``exp.data``     — the :class:`~xai_tester.io.DataRecorder`
    * ``exp.clock``    — the running :class:`~xai_tester.misc.Clock`

    Parameters
    ----------
    participant_id:
        Explicit participant identifier.  If given, no ID prompt is shown.
        Can be a string or integer.
    auto_id:
        If ``True``, a timestamp-based ID is generated automatically.
        Ignored if *participant_id* is provided.
    skip_instructions:
        If ``True``, the instructions screen is skipped (useful for testing).

    Returns
    -------
    experiment
        The active experiment.

    Raises
    ------
    RuntimeError
        If ``initialise()`` has not been called, or if the session has
        already been started.

    Examples
    --------
    >>> xai_tester.control.start(participant_id="P01")
    >>> xai_tester.control.start(auto_id=True)
    """
    exp = _get_active()

    if exp.is_started:
        raise RuntimeError(
            "Session already started. Call control.end() before restarting."
        )

    # ── Participant ID ───────────────────────────────────────────────────
    if participant_id is not None:
        pid = str(participant_id)
    elif auto_id:
        pid = f"auto_{int(time.time())}"
    else:
        print()
        try:
            pid = input("  Enter participant ID: ").strip()
        except (EOFError, KeyboardInterrupt):
            pid = f"auto_{int(time.time())}"
        if not pid:
            pid = f"auto_{int(time.time())}"

    # ── Session setup ────────────────────────────────────────────────────
    session = Session(trials=exp._trials, participant_id=pid)

    exp._session = session
    exp._data    = DataRecorder(
        experiment   = exp,
        session      = session,
        output_dir   = misc.defaults.data_directory,
        csv_delimiter= misc.defaults.csv_delimiter,
        decimal_places=misc.defaults.summary_decimal_places,
    )
    exp._is_started = True

    # ── Instructions screen ───────────────────────────────────────────────
    if not skip_instructions:
        text = misc.defaults.instructions_text
        text += f"\n\n{misc.defaults.ready_prompt}"
        io.presenter.show_message(text)

    # ── Start clock ───────────────────────────────────────────────────────
    exp._clock = misc.Clock()   # reset clock to session start
    session.started_at = exp._clock.session_time

    print(
        f"\n[xai_tester] Session started — participant '{pid}', "
        f"{session.n_trials} trials.\n"
    )
    return exp


# ---------------------------------------------------------------------------
# pause
# ---------------------------------------------------------------------------

def pause(message: str = "Session paused. Press ENTER to continue.") -> None:
    """Display a pause screen and wait for ENTER.

    Useful for inserting a rest break mid-session.

    Parameters
    ----------
    message:
        Text to display during the pause.

    Raises
    ------
    RuntimeError
        If the session has not been started.
    """
    _get_active()   # validates active experiment exists
    io.presenter.show_message(message)


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------

def end(
    show_summary: bool = True,
    goodbye_text: str | None = None,
) -> None:
    """End the active session, write all output files, and clean up.

    * Records session end time.
    * Writes the summary report (``<name>_<pid>_<ts>_summary.txt``).
    * Displays a goodbye message.
    * Resets the active experiment reference.

    Parameters
    ----------
    show_summary:
        If ``True``, prints the summary report to stdout before the goodbye
        screen.
    goodbye_text:
        Override for ``misc.defaults.goodbye_text``.

    Raises
    ------
    RuntimeError
        If the session has not been started.

    Examples
    --------
    >>> xai_tester.control.end()
    """
    
    global _active_exp

    exp = _get_active()

    if not exp.is_started:
        raise RuntimeError("Session has not been started yet.")

    # ── Record end time ──────────────────────────────────────────────────
    exp.session.ended_at = exp.clock.session_time

    # ── Write summary ────────────────────────────────────────────────────
    summary_path = exp.data.save_summary()

    if show_summary:
        summary_text = summary_path.read_text(encoding="utf-8")
        io.presenter.clear()
        print(summary_text)

    # ── Goodbye ──────────────────────────────────────────────────────────
    bye = goodbye_text or misc.defaults.goodbye_text
    io.presenter.show_message(bye)

    print(
        f"\n[xai_tester] Session ended.\n"
        f"  CSV     → {exp.data.csv_path}\n"
        f"  Summary → {summary_path}\n"
    )

    _active_exp = None

"""Trial presenter — renders each trial to the terminal.

The presenter draws:
  1. A progress header (trial N of M)
  2. A table of original feature values
  3. The AI model's prediction label (highlighted)
  4. A bar chart of XAI feature-importance scores
  5. A text-input prompt for the participant's label prediction

All rendering uses only the Python standard library so there are no
additional display dependencies.  A richer presenter (web, tkinter, etc.)
can be swapped in by subclassing :class:`BasePresenter` and assigning it
to ``xai_tester.io.presenter``.
"""

from __future__ import annotations

import os
import sys
import textwrap
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..design._structure import Trial


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _term_width() -> int:
    """Return current terminal width, defaulting to 80."""
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


def _hr(char: str = "─", width: int | None = None) -> str:
    w = width or _term_width()
    return char * w


def _centre(text: str, width: int | None = None) -> str:
    w = width or _term_width()
    return text.center(w)


def _importance_bar(
    score: float,
    max_abs: float,
    bar_width: int = 30,
) -> str:
    """Build a two-sided ASCII bar for a feature importance score.

    Example output (bar_width=20)::

        ████████░░░░░░░░░░░░  +0.42
        ░░░░░░░░░░░░░░░░████  -0.21
    """
    if max_abs == 0:
        filled = 0
    else:
        filled = int(abs(score) / max_abs * bar_width)
    empty = bar_width - filled

    FULL  = "█"
    EMPTY = "░"

    if score >= 0:
        bar = FULL * filled + EMPTY * empty
        sign = "+"
    else:
        bar = EMPTY * empty + FULL * filled
        sign = ""

    return f"{bar}  {sign}{score:+.4f}"


# ---------------------------------------------------------------------------
# ANSI colour helpers (no third-party lib required)
# ---------------------------------------------------------------------------

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_CYAN   = "\033[36m"
_YELLOW = "\033[33m"
_WHITE  = "\033[97m"

def _supports_colour() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    if _supports_colour():
        return f"{code}{text}{_RESET}"
    return text


# ---------------------------------------------------------------------------
# BasePresenter
# ---------------------------------------------------------------------------

class BasePresenter:
    """Abstract base class for trial presenters.

    Subclass this to implement a web, GUI, or alternative terminal presenter.
    Only :meth:`show_trial` and :meth:`get_response` are required.
    """

    def show_trial(
        self,
        trial: "Trial",
        trial_number: int,
        total_trials: int,
        labels: list[str] | None = None,
        bar_width: int = 30,
    ) -> None:
        """Render the trial to the participant.

        Displays four sections:
        1. Progress header.
        2. Original data table (``trial.features``).
        3. AI prediction label (``trial.ai_label``) — highlighted so the
           participant can see what the model decided before evaluating the
           explanation.
        4. XAI feature-importance bar chart (``trial.xai_scores``).

        Parameters
        ----------
        trial:
            The trial to display.
        trial_number:
            1-based index for the progress indicator.
        total_trials:
            Total number of trials in the session.
        labels:
            Known label list to show as a hint; ``None`` = no hint.
        bar_width:
            Character width of the importance bar.
        """
        raise NotImplementedError

    def get_response(
        self,
        trial: "Trial",
        timeout: int | None = None,
    ) -> tuple[str, float]:
        """Prompt the participant for a label prediction.

        Parameters
        ----------
        trial:
            The trial currently on screen.
        timeout:
            Maximum seconds to wait; ``None`` = unlimited.

        Returns
        -------
        (response, reaction_time_seconds)
        """
        raise NotImplementedError

    def show_message(self, text: str) -> None:
        """Show a plain message and wait for ENTER."""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear the display."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# TerminalPresenter
# ---------------------------------------------------------------------------

class TerminalPresenter(BasePresenter):
    """Default presenter — renders trials as formatted terminal output.

    Uses only ANSI escape codes and the standard library; works on Linux,
    macOS, and Windows 10+.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear the terminal screen."""
        os.system("cls" if os.name == "nt" else "clear")

    def show_trial(
        self,
        trial: "Trial",
        trial_number: int,
        total_trials: int,
        labels: list[str] | None = None,
        bar_width: int = 30,
    ) -> None:
        """Render the trial to the terminal.

        Displays four sections:
        1. **Header** — progress indicator and divider.
        2. **Original data** — feature name / value table.
        3. **AI Prediction** — the model's label, prominently highlighted.
        4. **XAI explanation** — two-sided importance bars, sorted by
           absolute importance descending.

        Parameters
        ----------
        trial:
            Trial to display.
        trial_number:
            1-based index for the progress indicator.
        total_trials:
            Total number of trials in the session.
        labels:
            If provided, shown as a hint line: "Possible labels: X, Y, Z".
        bar_width:
            Width of the ASCII importance bar in characters.
        """
        self.clear()
        w = _term_width()

        # ── Header ──────────────────────────────────────────────────────
        print(_c(_hr("═", w), _CYAN))
        progress = f"  Trial {trial_number} / {total_trials}"
        print(_c(progress, _BOLD + _WHITE))
        print(_c(_hr("─", w), _CYAN))
        print()

        # ── Original data table ──────────────────────────────────────────
        print(_c("  ORIGINAL DATA", _BOLD + _YELLOW))
        print(_c(_hr("─", w), _DIM))

        if trial.features:
            col_w = max(len(k) for k in trial.features) + 2
            val_w = w - col_w - 4
            for name, value in trial.features.items():
                val_str = str(value)
                if len(val_str) > val_w:
                    val_str = val_str[: val_w - 3] + "..."
                print(f"  {_c(name.ljust(col_w), _CYAN)}{val_str}")
        else:
            print("  (no features to display)")

        print()

        # ── AI Prediction + Ground Truth ─────────────────────────────────
        print(_c("  AI PREDICTION", _BOLD + _YELLOW))
        print(_c(_hr("─", w), _DIM))
        # Box the AI label so it stands out clearly
        box_width = max(len(trial.ai_label) + 6, 24)
        print(_c("  ┌" + "─" * (box_width - 2) + "┐", _GREEN))
        print(_c(f"  │  {trial.ai_label:<{box_width - 4}}│", _BOLD + _GREEN))
        print(_c("  └" + "─" * (box_width - 2) + "┘", _GREEN))

        if trial.ground_truth is not None:
            ai_correct = trial.ai_agrees
            gt_colour  = _GREEN if ai_correct else _RED
            tick       = "✓" if ai_correct else "✗"
            gt_line    = f"  Ground truth:  {trial.ground_truth}  {tick}"
            print(_c(gt_line, gt_colour))

        print()

        # ── XAI explanation ──────────────────────────────────────────────
        print(_c("  XAI EXPLANATION  ", _BOLD + _YELLOW))
        print(_c(_hr("─", w), _DIM))
        print(
            _c(
                "  ← pushes away from positive class"
                "   |   pushes toward positive class →",
                _DIM,
            )
        )
        print()

        # sorted_scores = sorted(
        #     trial.xai_scores.items(),
        #     key=lambda kv: abs(kv[1]),
        #     reverse=True,
        # )
        
        scores = trial.xai_scores.items()
        
        if trial.xai_scores.items():
            max_abs = max(abs(v) for _, v in scores) or 1.0
            name_w = max(len(k) for k, _ in scores) + 2
            for feat_name, score in scores:
                bar = _importance_bar(score, max_abs, bar_width)
                colour = _GREEN if score >= 0 else _RED
                print(
                    f"  {_c(feat_name.ljust(name_w), _CYAN)}"
                    f"{_c(bar, colour)}"
                )
        else:
            print("  (no XAI scores provided)")

        print()
        print(_c(_hr("─", w), _DIM))

        # ── Label hint ───────────────────────────────────────────────────
        if labels:
            hint = "  Possible labels: " + ", ".join(
                _c(f"[{lb}]", _YELLOW) for lb in labels
            )
            print(hint)
            print()

    def get_response(
        self,
        trial: "Trial",
        timeout: int | None = None,
    ) -> tuple[str, float]:
        """Prompt for a free-text response and measure reaction time.

        The timer starts the moment this method is called (i.e. from when
        the participant first sees the prompt).  If *timeout* is set and
        expires before ENTER is pressed, an empty string is recorded.

        Parameters
        ----------
        trial:
            Current trial (unused directly here; available for subclasses).
        timeout:
            Seconds before auto-recording an empty response.  ``None`` means
            wait indefinitely.

        Returns
        -------
        (response_str, reaction_time_seconds)
            *response_str* is stripped of leading/trailing whitespace.
            *reaction_time_seconds* is a float measured from prompt display.
        """
        prompt = _c("  ▶ Your prediction: ", _BOLD + _WHITE)

        if timeout is None:
            t0 = time.monotonic()
            try:
                response = input(prompt)
            except EOFError:
                response = ""
            rt = time.monotonic() - t0
        else:
            # Timeout via threading — falls back to empty response
            import threading

            result: list[str] = []
            event = threading.Event()

            def _read() -> None:
                try:
                    result.append(input(prompt))
                except EOFError:
                    result.append("")
                event.set()

            t0 = time.monotonic()
            thread = threading.Thread(target=_read, daemon=True)
            thread.start()
            fired = event.wait(timeout)
            rt = time.monotonic() - t0

            if not fired:
                print()  # newline after timeout
                response = ""
            else:
                response = result[0] if result else ""

        return response.strip(), rt

    def show_message(self, text: str) -> None:
        """Display *text* and wait for the participant to press ENTER.

        Parameters
        ----------
        text:
            Multi-line message to display.
        """
        self.clear()
        w = _term_width()
        print(_c(_hr("═", w), _CYAN))
        print()
        for line in text.splitlines():
            wrapped = textwrap.fill(line, width=w - 4)
            for wline in wrapped.splitlines():
                print(f"  {wline}")
        print()
        print(_c(_hr("─", w), _DIM))
        try:
            input(_c("  Press ENTER to continue… ", _DIM))
        except EOFError:
            pass

    def show_pause(self, seconds: float) -> None:
        """Show a brief blank pause between trials.

        Parameters
        ----------
        seconds:
            Duration of the pause.
        """
        self.clear()
        time.sleep(seconds)

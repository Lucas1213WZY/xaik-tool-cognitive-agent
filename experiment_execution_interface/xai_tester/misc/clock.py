"""Timing utilities for XAI Tester."""

import time


class Clock:
    """Monotonic wall-clock timer.

    Tracks elapsed time from a fixed start point. Used internally to measure
    per-trial response times and total session duration.

    Examples
    --------
    >>> clock = Clock()
    >>> clock.reset()
    >>> # ... show stimulus ...
    >>> rt = clock.elapsed()   # seconds since last reset()
    """

    def __init__(self) -> None:
        self._start: float = time.monotonic()
        self._split: float = self._start

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def session_time(self) -> float:
        """Seconds elapsed since this Clock was created (session start)."""
        return time.monotonic() - self._start

    def reset(self) -> None:
        """Reset the split timer (call just before presenting each trial)."""
        self._split = time.monotonic()

    def elapsed(self) -> float:
        """Seconds elapsed since the last :meth:`reset` call."""
        return time.monotonic() - self._split

    def wait(self, seconds: float) -> None:
        """Block for *seconds* (float precision).

        Parameters
        ----------
        seconds:
            Duration to wait.
        """
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(0.001)

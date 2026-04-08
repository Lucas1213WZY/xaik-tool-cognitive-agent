"""Core design structures: Experiment, Session, Trial.

These three classes mirror the three-level hierarchy of Expyriment
(Experiment → Block → Trial) but are simplified for XAI evaluation studies:

    Experiment
        └── Session          (one per participant run)
                └── Trial    (one per CSV row)
"""

from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import misc


# ---------------------------------------------------------------------------
# Trial
# ---------------------------------------------------------------------------

@dataclass
class Trial:
    """One XAI evaluation trial — a single row from the input CSV.

    Attributes
    ----------
    index : int
        Zero-based position in the original CSV.
    features : dict[str, Any]
        The original data fields shown to the participant.
    xai_scores : dict[str, float]
        Feature-importance scores keyed by feature name.
        Positive values indicate a push *toward* the positive class;
        negative values indicate a push *away*.
    ai_label : str
        The label assigned by the AI model.  Shown to participants as
        "AI Prediction" so they can evaluate whether the XAI explanation
        justifies that decision.  Also used for scoring (``is_correct``).
    ground_truth : str | None
        The real-world outcome for this case (e.g. whether the loan was
        actually repaid).  Shown to participants alongside the AI prediction
        so they can see whether the AI was right.  ``None`` if not provided.
    response : str | None
        The participant's free-text prediction (filled in during the run).
    response_time : float | None
        Seconds from trial presentation to ENTER keypress.
    presented_at : float | None
        ``Clock.session_time`` value when this trial was presented.
    """

    index: int
    features: dict[str, Any]
    xai_scores: dict[str, float]
    ai_label: str
    ground_truth: str | None = field(default=None)
    response: str | None = field(default=None, repr=False)
    response_time: float | None = field(default=None, repr=False)
    presented_at: float | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def is_correct(self) -> bool | None:
        """True if the participant's response matches the AI label.

        Returns ``None`` if no response has been recorded yet.
        Comparison is case-insensitive and strips leading/trailing whitespace.
        """
        if self.response is None:
            return None
        return self.response.strip().lower() == self.ai_label.strip().lower()

    @property
    def ai_agrees(self) -> bool | None:
        """True if the AI label matches the ground truth.

        Returns ``None`` if ``ground_truth`` was not provided.
        Useful for analysing whether participants are better at predicting
        AI errors vs. correct AI decisions.
        """
        if self.ground_truth is None:
            return None
        return self.ai_label.strip().lower() == self.ground_truth.strip().lower()

    def top_features(self, n: int = 5) -> list[tuple[str, float]]:
        """Return the *n* features with the highest absolute importance.

        Parameters
        ----------
        n:
            Number of features to return.

        Returns
        -------
        list of (feature_name, score) tuples, sorted by |score| descending.
        """
        return sorted(
            self.xai_scores.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )[:n]


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class Session:
    """A single participant's run through all trials.

    A Session is created automatically by :class:`Experiment` — you rarely
    need to instantiate it directly.

    Parameters
    ----------
    trials:
        Ordered list of :class:`Trial` objects for this session.
    participant_id:
        Identifier for the participant (string or integer).
    """

    def __init__(
        self,
        trials: list[Trial],
        participant_id: str | int = "unknown",
    ) -> None:
        self._trials: list[Trial] = list(trials)
        self.participant_id: str | int = participant_id
        self.started_at: float | None = None
        self.ended_at: float | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def trials(self) -> list[Trial]:
        """Ordered list of trials in this session (read-only view)."""
        return list(self._trials)

    @property
    def n_trials(self) -> int:
        """Total number of trials."""
        return len(self._trials)

    @property
    def n_completed(self) -> int:
        """Number of trials that have a recorded response."""
        return sum(1 for t in self._trials if t.response is not None)

    @property
    def duration(self) -> float | None:
        """Total session duration in seconds, or None if not yet ended."""
        if self.started_at is None or self.ended_at is None:
            return None
        return self.ended_at - self.started_at


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class Experiment:
    """Top-level container for an XAI evaluation experiment.

    An Experiment holds the configuration, loads the CSV data, builds
    :class:`Trial` objects, and (after :func:`~xai_tester.control.start` is
    called) exposes a :class:`Session` and a
    :class:`~xai_tester.io.DataRecorder` as ``exp.session`` and ``exp.data``.

    Parameters
    ----------
    name:
        Human-readable experiment name (used in output filenames).
    labels:
        Exhaustive list of possible AI prediction labels that participants
        may type.  Used for validation and for showing a hint on screen.
        If ``None``, any free-text response is accepted.
    filename_suffix:
        Optional string appended to output filenames before the extension.
    randomise_trials:
        Whether to shuffle trial order.  Overrides
        ``misc.defaults.randomise_trials`` when set explicitly.

    Examples
    --------
    >>> exp = Experiment(name="LIME Study", labels=["Approved", "Rejected"])
    >>> exp.load_csv(
    ...     "data.csv",
    ...     ai_label_col="ai_label",
    ...     xai_cols=["feat_age", "feat_income"],
    ... )
    """

    def __init__(
        self,
        name: str = "XAI Experiment",
        labels: list[str] | None = None,
        filename_suffix: str | None = None,
        randomise_trials: bool = True,
    ) -> None:
        self.name: str = name
        self.labels: list[str] | None = labels
        self.filename_suffix: str | None = filename_suffix
        self.randomise_trials: bool = randomise_trials

        self._trials: list[Trial] = []
        self._session: Session | None = None
        self._data = None           # set by control.start()
        self._clock: misc.Clock | None = None
        self._is_initialised: bool = False
        self._is_started: bool = False

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Fixed-format loader  (y / v0–v5 / a0–a5 convention)
    # ------------------------------------------------------------------

    def load_fixed_format(
        self,
        rows: list[dict],
        feature_names: list[str],
        label_map: dict | None = None,
        ground_truth_col: str | None = None,
        n_features: int = 5,
        max_rows: int | None = None,
        n_trials: int | None = None,
    ) -> "Experiment":
        """Load trials from a list of dicts in the fixed v/a/y format.

        This is the canonical input format assumed by XAI Tester:

        * ``y``          — AI prediction label (integer 0/1 or string)
        * ``v0``–``v4``  — raw feature values
        * ``a0``–``a4``  — feature attribution / importance scores (floats)

        Column names beyond ``v4``/``a4`` are supported via *n_features*.

        Parameters
        ----------
        rows:
            List of dicts as produced by :meth:`load_csv` or
            :meth:`load_numbers`.  Each dict must have keys ``y``,
            ``v0``…``v{n-1}``, ``a0``…``a{n-1}``.
        feature_names:
            Human-readable names for each feature, in the same order as
            ``v0``…``v{n-1}``.  Length must equal *n_features*.
        label_map:
            Optional mapping from raw ``y`` values to display strings, e.g.
            ``{0: "Rejected", 1: "Approved"}``.  If ``None``, the raw value
            is used as-is.
        ground_truth_col:
            Name of a key in each row that holds the real-world outcome.
            When provided, the value is stored in ``trial.ground_truth``.
        n_features:
            Number of features (default 5 → v0–v4 / a0–a4).
        n_trials:
            If set, randomly sample exactly *n_trials* from all loaded rows.
            Sampling happens after *max_rows* is applied, and before the
            final shuffle.  Raises ``ValueError`` if *n_trials* exceeds the 
            number of available rows.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If *feature_names* length does not match *n_features*, or if a
            required key is missing from a row.
        """
        if len(feature_names) != n_features:
            raise ValueError(
                f"feature_names has {len(feature_names)} entries but "
                f"n_features={n_features}."
            )

        v_cols = [f"v{i}" for i in range(n_features)]
        a_cols = [f"a{i}" for i in range(n_features)]

        self._trials.clear()

        for idx, row in enumerate(rows):
            if max_rows is not None and idx >= max_rows:
                break

            # AI label — map integer codes to strings if label_map provided
            raw_y = str(row.get("y", "")).strip()
            if label_map is not None:
                # Try both int and string keys
                ai_label = str(
                    label_map.get(int(raw_y), label_map.get(raw_y, raw_y))
                )
            else:
                ai_label = raw_y

            # Feature values — use human-readable names as keys
            try:
                features = {
                    feature_names[i]: row[v_cols[i]]
                    for i in range(n_features)
                }
            except KeyError as exc:
                raise ValueError(
                    f"Row {idx}: missing feature column {exc}"
                ) from exc

            # Attribution scores — use human-readable names as keys
            try:
                xai_scores = {
                    feature_names[i]: float(row[a_cols[i]])
                    for i in range(n_features)
                }
            except KeyError as exc:
                raise ValueError(
                    f"Row {idx}: missing attribution column {exc}"
                ) from exc
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Row {idx}: could not convert attribution to float — {exc}"
                ) from exc

            # Ground truth
            ground_truth = (
                str(row[ground_truth_col]).strip()
                if ground_truth_col is not None and ground_truth_col in row
                else None
            )
            if ground_truth is not None and label_map is not None:
                ground_truth = str(
                    label_map.get(int(ground_truth),
                                  label_map.get(ground_truth, ground_truth))
                )

            self._trials.append(
                Trial(
                    index=idx,
                    features=features,
                    xai_scores=xai_scores,
                    ai_label=ai_label,
                    ground_truth=ground_truth,
                )
            )

        # ── Random trial subset ──────────────────────────────────────
        if n_trials is not None:
            if n_trials > len(self._trials):
                raise ValueError(
                    f"n_trials={n_trials} exceeds the number of available "
                    f"trials ({len(self._trials)})."
                )
            self._trials = random.sample(self._trials, n_trials)

        if self.randomise_trials:
            random.shuffle(self._trials)

        return self

    # ------------------------------------------------------------------
    # CSV loader  (flexible / general-purpose)
    # ------------------------------------------------------------------

    def load_csv(
        self,
        filepath: str | Path,
        ai_label_col: str,
        xai_cols: list[str],
        ground_truth_col: str | None = None,
        feature_cols: list[str] | None = None,
        label_map: dict | None = None,
        delimiter: str = ",",
        encoding: str = "utf-8",
        max_rows: int | None = None,
        n_trials: int | None = None,
    ) -> "Experiment":
        """Load trials from a CSV file with arbitrary column names.

        For CSVs in the standard fixed format (columns ``y``, ``v0``–``v5``,
        ``a0``–``a5``) use :meth:`load_csv_fixed` instead — it handles the
        column-name-to-feature-name mapping automatically.

        Parameters
        ----------
        filepath:
            Path to the input CSV.
        ai_label_col:
            Column holding the AI model's prediction.  Shown to participants
            as "AI Prediction" and used to score ``trial.is_correct``.
        xai_cols:
            Columns with feature-importance scores (floats).  Column names
            become the bar-chart labels shown to participants.
        ground_truth_col:
            Column holding the real-world outcome.  Shown with ✓/✗ next to
            the AI prediction.  ``None`` = not provided.
        feature_cols:
            Columns to display as original data.  Auto-inferred if ``None``:
            any column not in ``ai_label_col``, ``xai_cols``, or
            ``ground_truth_col``.
        label_map:
            Optional ``{raw_value: display_string}`` mapping applied to both
            ``ai_label_col`` and ``ground_truth_col``.  Useful when the CSV
            stores integer codes (e.g. ``0``/``1``) that should be displayed
            as human-readable strings.
        delimiter:
            CSV field delimiter.  Default ``","``
        encoding:
            File encoding.  Default ``"utf-8"``.
        max_rows:
            Load only the first N rows.
        n_trials:
            If set, randomly sample exactly *n_trials* from all loaded rows.
            Raises ``ValueError`` if *n_trials* exceeds available rows.
            
        Returns
        -------
        self

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist.
        KeyError
            If a required column is missing from the CSV header.
        ValueError
            If an XAI column value cannot be converted to float.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        self._trials.clear()

        with open(path, newline="", encoding=encoding) as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            if reader.fieldnames is None:
                raise ValueError("CSV file appears to be empty.")

            fieldnames = list(reader.fieldnames)

            # Validate required columns
            required = [ai_label_col, *xai_cols]
            if ground_truth_col is not None:
                required.append(ground_truth_col)
            missing = [c for c in required if c not in fieldnames]
            if missing:
                raise KeyError(
                    f"Columns not found in CSV: {missing}. "
                    f"Available: {fieldnames}"
                )

            # Determine which columns are plain features
            hidden = {ai_label_col} | set(xai_cols)
            if ground_truth_col is not None:
                hidden.add(ground_truth_col)
            if feature_cols is None:
                feature_cols = [c for c in fieldnames if c not in hidden]

            def _apply_label_map(raw: str) -> str:
                if label_map is None:
                    return raw
                try:
                    return str(label_map.get(int(raw), label_map.get(raw, raw)))
                except (ValueError, TypeError):
                    return str(label_map.get(raw, raw))

            for idx, row in enumerate(reader):
                if max_rows is not None and idx >= max_rows:
                    break

                features = {c: row[c] for c in feature_cols}
                ai_label = _apply_label_map(row[ai_label_col].strip())
                ground_truth = (
                    _apply_label_map(row[ground_truth_col].strip())
                    if ground_truth_col is not None
                    else None
                )

                try:
                    xai_scores = {c: float(row[c]) for c in xai_cols}
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"Row {idx + 1}: could not convert XAI score to "
                        f"float — {exc}"
                    ) from exc

                self._trials.append(
                    Trial(
                        index=idx,
                        features=features,
                        xai_scores=xai_scores,
                        ai_label=ai_label,
                        ground_truth=ground_truth,
                    )
                )
                
        # ── Random trial subset ──────────────────────────────────────
        if n_trials is not None:
            if n_trials > len(self._trials):
                raise ValueError(
                    f"n_trials={n_trials} exceeds the number of available "
                    f"trials ({len(self._trials)})."
                )
            self._trials = random.sample(self._trials, n_trials)

        # ── Random trial subset ──────────────────────────────────────
        if n_trials is not None:
            if n_trials > len(self._trials):
                raise ValueError(
                    f"n_trials={n_trials} exceeds the number of available "
                    f"trials ({len(self._trials)})."
                )
            self._trials = random.sample(self._trials, n_trials)

        if self.randomise_trials:
            random.shuffle(self._trials)

        return self

    # ------------------------------------------------------------------
    # Fixed-format CSV loader  (y / v0–vN / a0–aN convention)
    # ------------------------------------------------------------------

    def load_csv_fixed(
        self,
        filepath: str | Path,
        feature_names: list[str],
        label_map: dict | None = None,
        ground_truth_col: str | None = None,
        n_features: int | None = None,
        delimiter: str = ",",
        encoding: str = "utf-8",
        max_rows: int | None = None,
        n_trials: int | None = None,
    ) -> "Experiment":
        """Load a CSV whose columns follow the fixed ``y / v0–vN / a0–aN`` convention.

        This is the standard input format for XAI Tester:

        +--------------+---------------------------------------------+
        | Column       | Description                                 |
        +==============+=============================================+
        | ``y``        | AI prediction (integer code or string)      |
        +--------------+---------------------------------------------+
        | ``v0``–``vN``| Raw feature values shown to participants    |
        +--------------+---------------------------------------------+
        | ``a0``–``aN``| Attribution / importance scores (floats)    |
        +--------------+---------------------------------------------+

        Parameters
        ----------
        filepath:
            Path to the CSV file.
        feature_names:
            Human-readable names for each feature, matched positionally to
            ``v0``, ``v1``, …  Length determines *n_features* when
            *n_features* is ``None``.
        label_map:
            ``{raw_y_value: display_string}`` mapping.  Example:
            ``{0: "Rejected", 1: "Approved"}``.  Accepts int or string keys.
            Applied to both ``y`` and ``ground_truth_col``.
        ground_truth_col:
            Name of an extra column holding the real-world outcome (optional).
        n_features:
            Override the number of features.  Defaults to
            ``len(feature_names)``.
        delimiter:
            CSV field delimiter.  Default ``","``
        encoding:
            File encoding.  Default ``"utf-8"``.
        max_rows:
            Load only the first N rows.

        Returns
        -------
        self

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist.
        ValueError
            If required columns are missing or attribution values are
            non-numeric.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        
        print(feature_names)
        
        n = n_features if n_features is not None else len(feature_names)
        if len(feature_names) != n:
            raise ValueError(
                f"feature_names has {len(feature_names)} entries but "
                f"n_features={n}."
            )

        with open(path, newline="", encoding=encoding) as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            rows = list(reader)

        return self.load_fixed_format(
            rows=rows,
            feature_names=feature_names,
            label_map=label_map,
            ground_truth_col=ground_truth_col,
            n_features=n,
            max_rows=max_rows,
            n_trials=n_trials
        )

    def add_trial(self, trial: Trial) -> None:
        """Manually add a single :class:`Trial` (for programmatic setups).

        Parameters
        ----------
        trial:
            The trial to append.
        """
        self._trials.append(trial)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_trials(self) -> int:
        """Number of loaded trials."""
        return len(self._trials)

    @property
    def session(self) -> Session:
        """The active :class:`Session`.

        Raises
        ------
        RuntimeError
            If :func:`~xai_tester.control.start` has not been called yet.
        """
        if self._session is None:
            raise RuntimeError(
                "No active session. Call control.start() first."
            )
        return self._session

    @property
    def data(self):
        """The :class:`~xai_tester.io.DataRecorder` for this session.

        Raises
        ------
        RuntimeError
            If :func:`~xai_tester.control.start` has not been called yet.
        """
        if self._data is None:
            raise RuntimeError(
                "No data recorder. Call control.start() first."
            )
        return self._data

    @property
    def clock(self) -> misc.Clock:
        """Session clock.

        Raises
        ------
        RuntimeError
            If :func:`~xai_tester.control.initialise` has not been called yet.
        """
        if self._clock is None:
            raise RuntimeError(
                "Clock not available. Call control.initialise() first."
            )
        return self._clock

    @property
    def is_initialised(self) -> bool:
        """True after :func:`~xai_tester.control.initialise` succeeds."""
        return self._is_initialised

    @property
    def is_started(self) -> bool:
        """True after :func:`~xai_tester.control.start` succeeds."""
        return self._is_started

    def __repr__(self) -> str:
        return (
            f"Experiment(name={self.name!r}, "
            f"n_trials={self.n_trials}, "
            f"labels={self.labels!r})"
        )

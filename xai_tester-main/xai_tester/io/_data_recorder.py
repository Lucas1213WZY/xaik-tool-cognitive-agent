"""DataRecorder — records trial responses and writes output files.

Two output files are produced at the end of each session:

``<name>_<participant_id>_<timestamp>.csv``
    One row per trial.  Columns: trial_index, ai_label, response,
    is_correct, response_time_s, presented_at_s, plus all feature
    values and XAI scores.

``<name>_<participant_id>_<timestamp>_summary.txt``
    Human-readable summary: accuracy, mean RT, per-label breakdown.
"""

from __future__ import annotations

import csv
import statistics
import textwrap
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..design._structure import Experiment, Session, Trial


# ---------------------------------------------------------------------------
# DataRecorder
# ---------------------------------------------------------------------------

class DataRecorder:
    """Stores trial responses in memory and flushes them to disk on demand.

    Typically accessed as ``exp.data`` after :func:`~xai_tester.control.start`
    is called.

    Parameters
    ----------
    experiment:
        The parent :class:`~xai_tester.design.Experiment`.
    session:
        The active :class:`~xai_tester.design.Session`.
    output_dir:
        Directory where output files are written.
    csv_delimiter:
        Delimiter for the output CSV.
    decimal_places:
        Precision for numeric fields in the summary report.
    """

    def __init__(
        self,
        experiment: "Experiment",
        session: "Session",
        output_dir: str | Path = "xai_results",
        csv_delimiter: str = ",",
        decimal_places: int = 3,
    ) -> None:
        self._exp = experiment
        self._session = session
        self._delimiter = csv_delimiter
        self._decimal = decimal_places
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in experiment.name
        )
        safe_pid = str(session.participant_id).replace(" ", "_")
        stem = f"{safe_name}_{safe_pid}_{ts}"

        self._csv_path     = self._output_dir / f"{stem}.csv"
        self._summary_path = self._output_dir / f"{stem}_summary.txt"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, trial: "Trial", response: str, rt: float) -> None:
        """Attach a response to *trial* and persist immediately.

        Writes a new row to the CSV on every call so data is not lost if
        the session is interrupted.

        Parameters
        ----------
        trial:
            The trial that was just completed.
        response:
            Participant's free-text prediction (already stripped).
        rt:
            Reaction time in seconds.
        """
        trial.response      = response
        trial.response_time = rt

        self._append_csv_row(trial)

    def save_summary(self) -> Path:
        """Write the human-readable summary report and return its path.

        Returns
        -------
        Path
            Path to the written summary file.
        """
        lines = self._build_summary()
        self._summary_path.write_text("\n".join(lines), encoding="utf-8")
        return self._summary_path

    @property
    def csv_path(self) -> Path:
        """Path to the response CSV file."""
        return self._csv_path

    @property
    def summary_path(self) -> Path:
        """Path to the summary report file."""
        return self._summary_path

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def _csv_fieldnames(self, trial: "Trial") -> list[str]:
        base = [
            "trial_index",
            "ai_label",
            "ground_truth",
            "ai_agrees",
            "response",
            "is_correct",
            "response_time_s",
            "presented_at_s",
        ]
        feature_fields = [f"feat_{k}" for k in trial.features]
        xai_fields     = [f"xai_{k}"  for k in trial.xai_scores]
        return base + feature_fields + xai_fields

    def _csv_row(self, trial: "Trial") -> dict:
        row: dict = {
            "trial_index":     trial.index,
            "ai_label":        trial.ai_label,
            "ground_truth":    trial.ground_truth if trial.ground_truth is not None else "",
            "ai_agrees":       "" if trial.ai_agrees is None else str(trial.ai_agrees),
            "response":        trial.response or "",
            "is_correct":      trial.is_correct,
            "response_time_s": (
                round(trial.response_time, self._decimal)
                if trial.response_time is not None else ""
            ),
            "presented_at_s": (
                round(trial.presented_at, self._decimal)
                if trial.presented_at is not None else ""
            ),
        }
        for k, v in trial.features.items():
            row[f"feat_{k}"] = v
        for k, v in trial.xai_scores.items():
            row[f"xai_{k}"] = round(v, self._decimal + 2)
        return row

    def _append_csv_row(self, trial: "Trial") -> None:
        fieldnames = self._csv_fieldnames(trial)
        file_exists = self._csv_path.exists()

        with open(self._csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=fieldnames,
                delimiter=self._delimiter,
                extrasaction="ignore",
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(self._csv_row(trial))

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def _build_summary(self) -> list[str]:
        dp = self._decimal
        completed = [t for t in self._session.trials if t.response is not None]

        lines: list[str] = []
        W = 60

        def hr(c: str = "─") -> str:
            return c * W

        lines += [
            hr("═"),
            f"  XAI TESTER — SESSION SUMMARY",
            hr("─"),
            f"  Experiment    : {self._exp.name}",
            f"  Participant   : {self._session.participant_id}",
            f"  Trials total  : {self._session.n_trials}",
            f"  Trials done   : {len(completed)}",
        ]
        if self._session.duration is not None:
            lines.append(
                f"  Session time  : {self._session.duration:.{dp}f} s"
            )
        lines.append(hr())

        if not completed:
            lines.append("  No completed trials to analyse.")
            lines.append(hr("═"))
            return lines

        # ── Accuracy ────────────────────────────────────────────────────
        correct   = [t for t in completed if t.is_correct]
        incorrect = [t for t in completed if not t.is_correct]
        accuracy  = len(correct) / len(completed)

        lines += [
            "",
            "  PARTICIPANT ACCURACY  (predicted AI label correctly?)",
            hr("─"),
            f"  Correct   : {len(correct)} / {len(completed)}",
            f"  Incorrect : {len(incorrect)} / {len(completed)}",
            f"  Accuracy  : {accuracy:.{dp+2}%}",
        ]

        # ── AI accuracy vs ground truth ──────────────────────────────────
        gt_trials = [t for t in completed if t.ground_truth is not None]
        if gt_trials:
            ai_correct_trials   = [t for t in gt_trials if t.ai_agrees]
            ai_incorrect_trials = [t for t in gt_trials if not t.ai_agrees]
            ai_acc = len(ai_correct_trials) / len(gt_trials)
            lines += [
                "",
                "  AI MODEL ACCURACY  (AI label matched ground truth?)",
                hr("─"),
                f"  AI correct   : {len(ai_correct_trials)} / {len(gt_trials)}",
                f"  AI incorrect : {len(ai_incorrect_trials)} / {len(gt_trials)}",
                f"  AI accuracy  : {ai_acc:.{dp+2}%}",
            ]

            # How well did participants predict on AI-correct vs AI-wrong cases?
            if ai_correct_trials and ai_incorrect_trials:
                p_on_correct = (
                    sum(1 for t in ai_correct_trials if t.is_correct)
                    / len(ai_correct_trials)
                )
                p_on_wrong = (
                    sum(1 for t in ai_incorrect_trials if t.is_correct)
                    / len(ai_incorrect_trials)
                )
                lines += [
                    "",
                    "  PARTICIPANT ACCURACY BY AI CORRECTNESS",
                    hr("─"),
                    f"  When AI was correct : {p_on_correct:.{dp+2}%}",
                    f"  When AI was wrong   : {p_on_wrong:.{dp+2}%}",
                ]

        # ── Response times ───────────────────────────────────────────────
        rts = [t.response_time for t in completed if t.response_time is not None]
        if rts:
            lines += [
                "",
                "  RESPONSE TIMES (seconds)",
                hr("─"),
                f"  Mean   : {statistics.mean(rts):.{dp}f}",
                f"  Median : {statistics.median(rts):.{dp}f}",
                f"  SD     : {statistics.pstdev(rts):.{dp}f}",
                f"  Min    : {min(rts):.{dp}f}",
                f"  Max    : {max(rts):.{dp}f}",
            ]

        # ── Per-label breakdown ──────────────────────────────────────────
        label_stats: dict[str, dict] = {}
        for trial in completed:
            lbl = trial.ai_label
            if lbl not in label_stats:
                label_stats[lbl] = {"total": 0, "correct": 0, "rts": []}
            label_stats[lbl]["total"]  += 1
            if trial.is_correct:
                label_stats[lbl]["correct"] += 1
            if trial.response_time is not None:
                label_stats[lbl]["rts"].append(trial.response_time)

        if label_stats:
            lines += ["", "  PER-LABEL BREAKDOWN", hr("─")]
            for lbl, stats in sorted(label_stats.items()):
                acc_lbl = stats["correct"] / stats["total"]
                mean_rt = (
                    f"{statistics.mean(stats['rts']):.{dp}f} s"
                    if stats["rts"] else "—"
                )
                lines.append(
                    f"  [{lbl}]  "
                    f"n={stats['total']}  "
                    f"acc={acc_lbl:.{dp+2}%}  "
                    f"mean_RT={mean_rt}"
                )

        # ── XAI feature importance diagnostics ──────────────────────────
        # Show which features had the highest mean absolute importance
        all_xai: dict[str, list[float]] = {}
        for trial in completed:
            for feat, score in trial.xai_scores.items():
                all_xai.setdefault(feat, []).append(abs(score))

        if all_xai:
            mean_abs = {k: statistics.mean(v) for k, v in all_xai.items()}
            top = sorted(mean_abs.items(), key=lambda kv: kv[1], reverse=True)
            lines += ["", "  TOP FEATURES BY MEAN |IMPORTANCE|", hr("─")]
            for feat, val in top[:10]:
                lines.append(f"  {feat:<30s}  {val:.{dp}f}")

        lines += ["", hr("═"), ""]
        return lines

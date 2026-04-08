"""tests/test_xai_tester.py — full API test suite (no human input needed).

Run with:
    python tests/test_xai_tester.py
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Make sure we import from the local source tree
sys.path.insert(0, str(Path(__file__).parent.parent))

import xai_tester
from xai_tester import control, design, io, misc
from xai_tester.design._structure import Experiment, Session, Trial
from xai_tester.io._data_recorder import DataRecorder
from xai_tester.io._presenter import TerminalPresenter
from xai_tester.misc.clock import Clock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _sample_rows(n: int = 4) -> list[dict]:
    return [
        {
            "age": 30 + i, "income": 50 + i * 5,
            "ai_label": "Yes" if i % 2 == 0 else "No",
            "xai_age": 0.1 * i, "xai_income": -0.05 * i,
        }
        for i in range(n)
    ]


def _make_exp(tmp_path: Path, n: int = 4, randomise: bool = False) -> Experiment:
    rows = _sample_rows(n)
    csv_path = tmp_path / "data.csv"
    _make_csv(rows, csv_path)
    exp = Experiment(name="Test", labels=["Yes", "No"], randomise_trials=randomise)
    exp.load_csv(csv_path, ai_label_col="ai_label", xai_cols=["xai_age", "xai_income"])
    return exp


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------

class TestClock(unittest.TestCase):

    def test_elapsed_increases(self):
        c = Clock()
        c.reset()
        t1 = c.elapsed()
        t2 = c.elapsed()
        self.assertGreaterEqual(t2, t1)

    def test_session_time_starts_near_zero(self):
        c = Clock()
        self.assertLess(c.session_time, 1.0)

    def test_wait(self):
        import time
        c = Clock()
        t0 = time.monotonic()
        c.wait(0.05)
        self.assertGreaterEqual(time.monotonic() - t0, 0.04)


# ---------------------------------------------------------------------------
# Trial
# ---------------------------------------------------------------------------

class TestTrial(unittest.TestCase):

    def _trial(self, ai="Yes", response=None):
        t = Trial(
            index=0,
            features={"age": "30", "income": "50k"},
            xai_scores={"age": 0.4, "income": -0.2},
            ai_label=ai,
            response=response,
        )
        return t

    def test_is_correct_none_when_no_response(self):
        self.assertIsNone(self._trial().is_correct)

    def test_is_correct_true(self):
        t = self._trial(ai="Yes", response="yes")
        self.assertTrue(t.is_correct)

    def test_is_correct_false(self):
        t = self._trial(ai="Yes", response="No")
        self.assertFalse(t.is_correct)

    def test_is_correct_strips_whitespace(self):
        t = self._trial(ai="Yes", response="  Yes  ")
        self.assertTrue(t.is_correct)

    def test_ai_agrees_none_when_no_ground_truth(self):
        t = self._trial()
        self.assertIsNone(t.ai_agrees)

    def test_ai_agrees_true(self):
        t = Trial(0, {}, {}, ai_label="Yes", ground_truth="yes")
        self.assertTrue(t.ai_agrees)

    def test_ai_agrees_false(self):
        t = Trial(0, {}, {}, ai_label="Yes", ground_truth="No")
        self.assertFalse(t.ai_agrees)

    def test_top_features_sorted_by_abs(self):
        t = Trial(0, {}, {"a": 0.1, "b": -0.9, "c": 0.5}, "X")
        top = t.top_features(n=2)
        self.assertEqual(top[0][0], "b")
        self.assertEqual(top[1][0], "c")

    def test_top_features_respects_n(self):
        t = Trial(0, {}, {f"f{i}": float(i) for i in range(10)}, "X")
        self.assertEqual(len(t.top_features(n=3)), 3)


# ---------------------------------------------------------------------------
# Experiment / design
# ---------------------------------------------------------------------------

class TestExperiment(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_load_csv_basic(self):
        exp = _make_exp(self.tmp)
        self.assertEqual(exp.n_trials, 4)

    def test_load_csv_feature_cols_inferred(self):
        exp = _make_exp(self.tmp)
        t = exp._trials[0]
        self.assertIn("age", t.features)
        self.assertIn("income", t.features)
        self.assertNotIn("ai_label", t.features)
        self.assertNotIn("xai_age", t.features)

    def test_load_csv_xai_scores(self):
        exp = _make_exp(self.tmp)
        t = exp._trials[0]
        self.assertIn("xai_age", t.xai_scores)
        self.assertIsInstance(t.xai_scores["xai_age"], float)

    def test_load_csv_ground_truth(self):
        rows = _sample_rows(4)
        for r in rows:
            r["gt"] = "Yes" if int(r["age"]) > 30 else "No"
        path = self.tmp / "gt.csv"
        _make_csv(rows, path)
        exp = Experiment(randomise_trials=False)
        exp.load_csv(path, ai_label_col="ai_label",
                     xai_cols=["xai_age", "xai_income"],
                     ground_truth_col="gt")
        self.assertIsNotNone(exp._trials[0].ground_truth)
        self.assertNotIn("gt", exp._trials[0].features)

    def test_load_csv_ground_truth_not_in_features(self):
        rows = _sample_rows(3)
        for r in rows:
            r["actual"] = "Yes"
        path = self.tmp / "gt2.csv"
        _make_csv(rows, path)
        exp = Experiment(randomise_trials=False)
        exp.load_csv(path, ai_label_col="ai_label",
                     xai_cols=["xai_age", "xai_income"],
                     ground_truth_col="actual")
        for t in exp._trials:
            self.assertNotIn("actual", t.features)

    def test_load_csv_missing_ground_truth_col_raises(self):
        exp = Experiment()
        rows = [{"a": 1, "b": 2, "label": "Y", "xai_a": 0.1}]
        path = self.tmp / "mini.csv"
        _make_csv(rows, path)
        with self.assertRaises(KeyError):
            exp.load_csv(path, ai_label_col="label",
                         xai_cols=["xai_a"], ground_truth_col="nonexistent")

    def test_load_csv_missing_file_raises(self):
        exp = Experiment()
        with self.assertRaises(FileNotFoundError):
            exp.load_csv("nonexistent.csv", ai_label_col="x", xai_cols=["y"])

    def test_load_csv_missing_column_raises(self):
        exp = Experiment()
        rows = [{"a": 1, "b": 2}]
        path = self.tmp / "mini.csv"
        _make_csv(rows, path)
        with self.assertRaises(KeyError):
            exp.load_csv(path, ai_label_col="label", xai_cols=["xai_a"])

    def test_load_csv_bad_xai_value_raises(self):
        rows = [{"feat": "x", "label": "Y", "xai": "not_a_number"}]
        path = self.tmp / "bad.csv"
        _make_csv(rows, path)
        exp = Experiment()
        with self.assertRaises(ValueError):
            exp.load_csv(path, ai_label_col="label", xai_cols=["xai"])

    def test_max_rows(self):
        exp = Experiment(randomise_trials=False)
        rows = _sample_rows(10)
        path = self.tmp / "big.csv"
        _make_csv(rows, path)
        exp.load_csv(path, ai_label_col="ai_label",
                     xai_cols=["xai_age", "xai_income"], max_rows=3)
        self.assertEqual(exp.n_trials, 3)

    def test_add_trial_manual(self):
        exp = Experiment()
        t = Trial(0, {"x": "1"}, {"xai_x": 0.5}, "A")
        exp.add_trial(t)
        self.assertEqual(exp.n_trials, 1)

    def test_load_csv_label_map_int_keys(self):
        """label_map with int keys converts numeric y values to strings."""
        rows = [{"feat": "x", "label": "0", "xai": "0.1"},
                {"feat": "y", "label": "1", "xai": "-0.2"}]
        path = self.tmp / "lm.csv"
        _make_csv(rows, path)
        exp = Experiment(randomise_trials=False)
        exp.load_csv(path, ai_label_col="label", xai_cols=["xai"],
                     label_map={0: "Bad", 1: "Good"})
        self.assertEqual(exp._trials[0].ai_label, "Bad")
        self.assertEqual(exp._trials[1].ai_label, "Good")

    def test_session_raises_before_start(self):
        exp = _make_exp(self.tmp)
        with self.assertRaises(RuntimeError):
            _ = exp.session

    def test_data_raises_before_start(self):
        exp = _make_exp(self.tmp)
        with self.assertRaises(RuntimeError):
            _ = exp.data


# ---------------------------------------------------------------------------
# Fixed-format loaders  (y / v0–v5 / a0–a5 convention)
# ---------------------------------------------------------------------------

def _fixed_rows(n: int = 4) -> list[dict]:
    """Generate minimal rows in the fixed y/v0-v5/a0-a5 format."""
    return [
        {
            "y": i % 2,
            "v0": 0.3 + i * 0.1, "v1": 20.0 + i, "v2": 3.2 + i * 0.05,
            "v3": 0.5 + i * 0.1, "v4": 11.0 + i * 0.3, "v5": 5.0 + i,
            "a0": 0.1 * i, "a1": -0.05 * i, "a2": 0.08 * i,
            "a3": -0.03 * i, "a4": 0.15 * i, "a5": 0.01 * i,
        }
        for i in range(n)
    ]


class TestFixedFormat(unittest.TestCase):

    FEATURES = ["Vinegar Taint", "SO2", "pH", "Sulphates", "Alcohol", "Others"]
    LABEL_MAP = {0: "Bad Quality", 1: "Good Quality"}

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _fixed_csv(self, n: int = 4) -> Path:
        path = self.tmp / "wine.csv"
        rows = _fixed_rows(n)
        _make_csv(rows, path)
        return path

    # ── load_fixed_format ────────────────────────────────────────────────────

    def test_load_fixed_format_basic(self):
        exp = Experiment(randomise_trials=False)
        exp.load_fixed_format(
            rows=_fixed_rows(4),
            feature_names=self.FEATURES,
            label_map=self.LABEL_MAP,
        )
        self.assertEqual(exp.n_trials, 4)

    def test_load_fixed_format_feature_names_as_keys(self):
        exp = Experiment(randomise_trials=False)
        exp.load_fixed_format(_fixed_rows(2), self.FEATURES)
        t = exp._trials[0]
        self.assertIn("Vinegar Taint", t.features)
        self.assertIn("Alcohol", t.features)
        self.assertNotIn("v0", t.features)

    def test_load_fixed_format_xai_uses_feature_names(self):
        exp = Experiment(randomise_trials=False)
        exp.load_fixed_format(_fixed_rows(2), self.FEATURES)
        t = exp._trials[0]
        self.assertIn("Vinegar Taint", t.xai_scores)
        self.assertIn("Alcohol", t.xai_scores)
        self.assertIsInstance(t.xai_scores["Vinegar Taint"], float)

    def test_load_fixed_format_label_map_applied(self):
        exp = Experiment(randomise_trials=False)
        exp.load_fixed_format(_fixed_rows(4), self.FEATURES, label_map=self.LABEL_MAP)
        labels = {t.ai_label for t in exp._trials}
        self.assertIn("Bad Quality", labels)
        self.assertIn("Good Quality", labels)
        self.assertNotIn("0", labels)
        self.assertNotIn("1", labels)

    def test_load_fixed_format_no_label_map_raw_values(self):
        exp = Experiment(randomise_trials=False)
        exp.load_fixed_format(_fixed_rows(2), self.FEATURES, label_map=None)
        self.assertIn(exp._trials[0].ai_label, ["0", "1"])

    def test_load_fixed_format_max_rows(self):
        exp = Experiment(randomise_trials=False)
        exp.load_fixed_format(_fixed_rows(10), self.FEATURES, max_rows=3)
        self.assertEqual(exp.n_trials, 3)

    def test_load_fixed_format_wrong_feature_names_count_raises(self):
        exp = Experiment()
        with self.assertRaises(ValueError):
            exp.load_fixed_format(_fixed_rows(2), ["Only", "Two", "Names"])

    def test_load_fixed_format_missing_v_col_raises(self):
        rows = [{"y": 0, "v0": 1.0}]  # missing v1-v5 and a0-a5
        exp = Experiment()
        with self.assertRaises(ValueError):
            exp.load_fixed_format(rows, self.FEATURES)

    def test_load_fixed_format_ground_truth(self):
        rows = _fixed_rows(4)
        for r in rows:
            r["actual"] = r["y"]  # ground truth = same as prediction
        exp = Experiment(randomise_trials=False)
        exp.load_fixed_format(rows, self.FEATURES,
                              label_map=self.LABEL_MAP,
                              ground_truth_col="actual")
        for t in exp._trials:
            self.assertIsNotNone(t.ground_truth)
            self.assertTrue(t.ai_agrees)  # AI matches ground truth here

    # ── load_csv_fixed ───────────────────────────────────────────────────────

    def test_load_csv_fixed_basic(self):
        exp = Experiment(randomise_trials=False)
        exp.load_csv_fixed(self._fixed_csv(), self.FEATURES,
                           label_map=self.LABEL_MAP)
        self.assertEqual(exp.n_trials, 4)

    def test_load_csv_fixed_feature_names_as_keys(self):
        exp = Experiment(randomise_trials=False)
        exp.load_csv_fixed(self._fixed_csv(), self.FEATURES)
        t = exp._trials[0]
        self.assertIn("Vinegar Taint", t.features)
        self.assertNotIn("v0", t.features)

    def test_load_csv_fixed_label_map(self):
        exp = Experiment(randomise_trials=False)
        exp.load_csv_fixed(self._fixed_csv(), self.FEATURES,
                           label_map=self.LABEL_MAP)
        labels = {t.ai_label for t in exp._trials}
        self.assertIn("Bad Quality", labels)

    def test_load_csv_fixed_missing_file_raises(self):
        exp = Experiment()
        with self.assertRaises(FileNotFoundError):
            exp.load_csv_fixed("nonexistent.csv", self.FEATURES)

    def test_load_csv_fixed_max_rows(self):
        path = self._fixed_csv(n=10)
        exp = Experiment(randomise_trials=False)
        exp.load_csv_fixed(path, self.FEATURES, max_rows=5)
        self.assertEqual(exp.n_trials, 5)

    def test_load_csv_fixed_is_correct_after_session(self):
        """End-to-end: load fixed CSV, run session, check is_correct."""
        import xai_tester.control._experiment_control as _ec
        _ec._active_exp = None

        misc.defaults.data_directory = str(self.tmp / "out")
        exp = Experiment(name="FixedTest", randomise_trials=False,
                         labels=list(self.LABEL_MAP.values()))
        exp.load_csv_fixed(self._fixed_csv(4), self.FEATURES,
                           label_map=self.LABEL_MAP)
        control.initialise(exp)
        control.start(participant_id="test_fixed", skip_instructions=True)

        for trial in exp.session.trials:
            trial.presented_at = exp.clock.session_time
            with patch("builtins.input", return_value=trial.ai_label):
                response, rt = io.get_response(trial)
            exp.data.record(trial, response, rt)

        with patch.object(io.presenter, "show_message"):
            with patch.object(io.presenter, "clear"):
                control.end(show_summary=False)

        completed = [t for t in exp.session.trials if t.response is not None]
        self.assertEqual(len(completed), 4)
        self.assertTrue(all(t.is_correct for t in completed))


# ---------------------------------------------------------------------------
# control — lifecycle
# ---------------------------------------------------------------------------

class TestControl(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # Reset active experiment before each test
        import xai_tester.control._experiment_control as _ec
        _ec._active_exp = None

    def _fresh_exp(self, n=3):
        return _make_exp(self.tmp, n=n, randomise=False)

    def test_initialise_sets_flag(self):
        exp = self._fresh_exp()
        control.initialise(exp)
        self.assertTrue(exp.is_initialised)

    def test_initialise_twice_raises(self):
        exp = self._fresh_exp()
        control.initialise(exp)
        with self.assertRaises(RuntimeError):
            control.initialise(exp)

    def test_initialise_empty_exp_raises(self):
        exp = Experiment()
        with self.assertRaises(ValueError):
            control.initialise(exp)

    def test_start_creates_session_and_data(self):
        exp = self._fresh_exp()
        misc.defaults.data_directory = str(self.tmp / "out")
        control.initialise(exp)
        control.start(participant_id="tester", skip_instructions=True)
        self.assertIsNotNone(exp._session)
        self.assertIsNotNone(exp._data)
        self.assertTrue(exp.is_started)

    def test_start_twice_raises(self):
        exp = self._fresh_exp()
        misc.defaults.data_directory = str(self.tmp / "out")
        control.initialise(exp)
        control.start(participant_id="P1", skip_instructions=True)
        with self.assertRaises(RuntimeError):
            control.start(participant_id="P2", skip_instructions=True)

    def test_end_writes_files(self):
        out = self.tmp / "results"
        misc.defaults.data_directory = str(out)
        exp = self._fresh_exp()
        control.initialise(exp)
        control.start(participant_id="P1", skip_instructions=True)

        # Simulate trial responses
        for trial in exp.session.trials:
            trial.presented_at = exp.clock.session_time
            exp.data.record(trial, trial.ai_label, 1.5)

        with patch.object(io.presenter, "show_message"):
            with patch.object(io.presenter, "clear"):
                control.end(show_summary=False)

        csvs = list(out.glob("*.csv"))
        txts = list(out.glob("*_summary.txt"))
        self.assertEqual(len(csvs), 1)
        self.assertEqual(len(txts), 1)

    def test_end_before_start_raises(self):
        exp = self._fresh_exp()
        misc.defaults.data_directory = str(self.tmp / "out")
        control.initialise(exp)
        with self.assertRaises(RuntimeError):
            control.end()


# ---------------------------------------------------------------------------
# DataRecorder
# ---------------------------------------------------------------------------

class TestDataRecorder(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _recorder(self, n=3):
        exp = _make_exp(self.tmp, n=n, randomise=False)
        session = Session(trials=exp._trials, participant_id="P99")
        return DataRecorder(exp, session, output_dir=self.tmp / "out")

    def test_record_writes_csv(self):
        rec = self._recorder()
        trial = Trial(0, {"age": "30"}, {"xai_age": 0.3}, "Yes")
        rec.record(trial, "Yes", 2.5)
        self.assertTrue(rec.csv_path.exists())

    def test_csv_has_correct_columns(self):
        rec = self._recorder()
        trial = Trial(0, {"age": "30", "income": "50"},
                      {"xai_age": 0.3, "xai_income": -0.1}, "Yes",
                      ground_truth="Yes")
        rec.record(trial, "Yes", 1.0)
        with open(rec.csv_path) as fh:
            header = fh.readline().strip().split(",")
        self.assertIn("ai_label", header)
        self.assertIn("ground_truth", header)
        self.assertIn("ai_agrees", header)
        self.assertIn("response", header)
        self.assertIn("is_correct", header)
        self.assertIn("response_time_s", header)
        self.assertIn("feat_age", header)
        self.assertIn("xai_xai_age", header)

    def test_ai_agrees_recorded_true(self):
        rec = self._recorder()
        trial = Trial(0, {"x": "1"}, {"xai_x": 0.1}, "Yes", ground_truth="Yes")
        rec.record(trial, "Yes", 0.8)
        with open(rec.csv_path) as fh:
            row = next(csv.DictReader(fh))
        self.assertEqual(row["ai_agrees"], "True")

    def test_ai_agrees_recorded_false(self):
        rec = self._recorder()
        trial = Trial(0, {"x": "1"}, {"xai_x": 0.1}, "Yes", ground_truth="No")
        rec.record(trial, "Yes", 0.8)
        with open(rec.csv_path) as fh:
            row = next(csv.DictReader(fh))
        self.assertEqual(row["ai_agrees"], "False")

    def test_ai_agrees_none_when_no_ground_truth(self):
        rec = self._recorder()
        trial = Trial(0, {"x": "1"}, {"xai_x": 0.1}, "Yes")
        rec.record(trial, "Yes", 0.8)
        with open(rec.csv_path) as fh:
            row = next(csv.DictReader(fh))
        # ground_truth=None → both columns written as empty string
        self.assertEqual(row["ground_truth"], "")
        self.assertEqual(row["ai_agrees"], "")

    def test_is_correct_recorded(self):
        rec = self._recorder()
        trial = Trial(0, {"x": "1"}, {"xai_x": 0.1}, "Yes")
        rec.record(trial, "Yes", 0.8)
        with open(rec.csv_path) as fh:
            reader = csv.DictReader(fh)
            row = next(reader)
        self.assertEqual(row["is_correct"], "True")

    def test_summary_written(self):
        rec = self._recorder()
        for i, t in enumerate(rec._session.trials):
            t.ground_truth = t.ai_label  # AI always correct in this fixture
            rec.record(t, t.ai_label if i % 2 == 0 else "Wrong", float(i + 1))
        path = rec.save_summary()
        self.assertTrue(path.exists())
        content = path.read_text()
        self.assertIn("PARTICIPANT ACCURACY", content)
        self.assertIn("AI MODEL ACCURACY", content)
        self.assertIn("RESPONSE TIMES", content)
        self.assertIn("PER-LABEL BREAKDOWN", content)


# ---------------------------------------------------------------------------
# TerminalPresenter (non-interactive paths)
# ---------------------------------------------------------------------------

class TestTerminalPresenter(unittest.TestCase):

    def test_show_trial_does_not_crash(self):
        p = TerminalPresenter()
        t = Trial(0, {"age": "35", "score": "720"},
                  {"xai_age": 0.4, "xai_score": -0.25}, "Approved")
        with patch("builtins.print"), patch("os.system"):
            p.show_trial(t, trial_number=1, total_trials=5,
                         labels=["Approved", "Rejected"], bar_width=20)

    def test_get_response_captures_input(self):
        p = TerminalPresenter()
        t = Trial(0, {}, {}, "X")
        with patch("builtins.input", return_value="  Approved  "):
            response, rt = p.get_response(t)
        self.assertEqual(response, "Approved")
        self.assertGreaterEqual(rt, 0.0)


# ---------------------------------------------------------------------------
# Integration test — full loop with mocked input
# ---------------------------------------------------------------------------

class TestIntegration(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        import xai_tester.control._experiment_control as _ec
        _ec._active_exp = None

    def test_full_session(self):
        out = self.tmp / "results"
        misc.defaults.data_directory = str(out)

        exp = _make_exp(self.tmp, n=4, randomise=False)
        control.initialise(exp)
        control.start(participant_id="integration_test", skip_instructions=True)

        responses = [t.ai_label for t in exp.session.trials]

        for i, trial in enumerate(exp.session.trials):
            trial.presented_at = exp.clock.session_time
            with patch("builtins.input", return_value=responses[i]):
                response, rt = io.get_response(trial)
            exp.data.record(trial, response, rt)

        with patch.object(io.presenter, "show_message"):
            with patch.object(io.presenter, "clear"):
                control.end(show_summary=False)

        # All trials answered correctly
        completed = [t for t in exp.session.trials if t.response is not None]
        self.assertEqual(len(completed), 4)
        self.assertTrue(all(t.is_correct for t in completed))

        csvs = list(out.glob("*.csv"))
        txts = list(out.glob("*_summary.txt"))
        self.assertEqual(len(csvs), 1)
        self.assertEqual(len(txts), 1)

        # Verify CSV row count
        with open(csvs[0]) as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)

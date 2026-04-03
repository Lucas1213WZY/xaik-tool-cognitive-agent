"""
population_em_tuner.py

Fast population-level fitting of:
- strategy assignment per (Participant ID, Session, Tested w/ XAI)
- Gaussian parameter distribution per (Strategy, XAIType, Tested w/ XAI)

This avoids per-participant Bayesian optimization. Instead it runs a small number of
forward simulations per unit (participant×session×condition) and updates group means/covariances
with an EM-like loop.

Assumptions:
- Parameters are consistent across datasets (do NOT condition on appId/modelName/expMethod).
- Parameters do NOT vary by session.
- Strategy CAN vary by session (assignment is per participant×session×condition).
- You have:
    - ParticipantExperimentRunner
    - Evaluator
    - participant_loader with .df and .list_all_participants()
    - loader_getter(app_id, exp_method, ai_model_name, xai_type) -> AIDatasetLoader
    - models dict: {strategy_name: strategy_class}
    - param_spaces dict: {strategy_name: list[skopt.space.Dimension]}  (names must match __init__ args)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

from skopt.space import Real, Integer, Categorical

from experiment_runner import ParticipantExperimentRunner
from evaluator import Evaluator


# ---------------------------
# Helpers for skopt spaces
# ---------------------------

def _dim_names(dims: List[Any]) -> List[str]:
    return [d.name for d in dims]

def _dim_by_name(dims: List[Any]) -> Dict[str, Any]:
    return {d.name: d for d in dims}

def _midpoint(dim: Any):
    if isinstance(dim, Real):
        a, b = float(dim.low), float(dim.high)
        return 0.5 * (a + b)
    if isinstance(dim, Integer):
        a, b = int(dim.low), int(dim.high)
        return int(round(0.5 * (a + b)))
    if isinstance(dim, Categorical):
        # pick first category as deterministic init
        return dim.categories[0]
    # fallback
    return None

def _clip_to_dim(v: Any, dim: Any):
    if isinstance(dim, Real):
        return float(np.clip(float(v), float(dim.low), float(dim.high)))
    if isinstance(dim, Integer):
        return int(np.clip(int(round(v)), int(dim.low), int(dim.high)))
    if isinstance(dim, Categorical):
        # if unknown, fallback to first
        return v if v in dim.categories else dim.categories[0]
    return v


# ---------------------------
# Population state
# ---------------------------

@dataclass
class GroupGaussian:
    param_names: List[str]               # includes only numeric params (we keep explanation_type out)
    mu: np.ndarray                       # (P,)
    Sigma: np.ndarray                    # (P,P), usually diagonal
    # For integer params (e.g., k), we still store in mu/Sigma but will round when using.


class PopulationEMTuner:
    """
    Fast EM-like tuner that:
    1) assigns strategies per unit (pid, session, condition)
    2) estimates Gaussian parameter distributions per (strategy, xai_type, condition)
    """

    def __init__(
        self,
        *,
        models: Dict[str, Any],                  # {strategy_name: class}
        param_spaces: Dict[str, List[Any]],      # {strategy_name: [Real/Integer/Categorical dims]}
        loader_getter,                           # (app_id, exp_method, ai_model_name, xai_type) -> ai_loader
        participant_loader,                      # has .df and .list_all_participants()
        ui,
        available_conditions: Tuple[str, ...] = ("w/ XAI", "w/o XAI"),
        optimization_metric: str = "nll_model_participant",
        max_participants: Optional[int] = None,
        seed: int = 1234,
    ):
        self.models = models
        self.param_spaces = param_spaces
        self.loader_getter = loader_getter
        self.participant_loader = participant_loader
        self.ui = ui

        self.available_conditions = list(available_conditions)
        self.optimization_metric = optimization_metric
        self.max_participants = max_participants

        self.rng = np.random.default_rng(seed)

        # learned
        self.group_gaussians: Dict[Tuple[str, str, str], GroupGaussian] = {}
        self.assignments_rows: List[Dict[str, Any]] = []

        # small cache: key=(pid, session, condition, strategy, tuple(params rounded/clipped)) -> score
        self._score_cache: Dict[Tuple[Any, ...], float] = {}

        self.verbose=True
        self.print_every = 200

        self._logs_cache: Dict[Tuple[Any, ...], Any] = {}
        self._metrics_cache: Dict[Tuple[Any, ...], float] = {}
    
    def _log(self, msg: str, level: int = 1):
        if getattr(self, "verbose", False):
            print(msg)


    def _run_and_get_logs_cached(
        self,
        *,
        pid,
        condition: str,
        strategy_name: str,
        hyperparams: Dict[str, Any],
    ):
        prow = self._get_participant_row(pid)
        if prow is None:
            return None

        app_id = prow["appId"]
        exp_method = prow["expMethod"]
        ai_model_name = prow["modelName"]
        xai_type = (prow.get("XAIType", "") or "").lower()

        if not self._allowed_for_condition(strategy_name, xai_type, condition):
            return None

        params = dict(hyperparams)
        if "decay_param" not in params:
            params["decay_param"] = 0.5
        if strategy_name == "Attribution Sum":
            params["explanation_type"] = xai_type

        key = (pid, condition, strategy_name, xai_type) + self._params_cache_key(strategy_name, params)
        if key in self._logs_cache:
            return self._logs_cache[key]

        ai_loader = self.loader_getter(app_id, exp_method, ai_model_name, xai_type)
        human_model_class = self.models[strategy_name]
        human_model = human_model_class(**params)

        runner = ParticipantExperimentRunner(
            human_model=human_model,
            ai_dataset_loader=ai_loader,
            participant_dataset_loader=self.participant_loader,
            ui=self.ui,
        )
        human_model.time = runner.time
        logs = runner.run_experiment(pid)

        self._logs_cache[key] = logs
        return logs

    def _run_and_score(
        self,
        *,
        pid,
        session_num: int,
        condition: str,
        strategy_name: str,
        hyperparams: Dict[str, Any],
    ) -> Tuple[float, float, float]:
        prow = self._get_participant_row(pid)
        if prow is None:
            return float("inf")

        xai_type = (prow.get("XAIType", "") or "").lower()

        # build params exactly as before (so cache keys match)
        params = dict(hyperparams)
        if "decay_param" not in params:
            params["decay_param"] = 0.5
        if strategy_name == "Attribution Sum":
            params["explanation_type"] = xai_type

        # metric cache key MUST include session_num
        metric_key = (pid, session_num, condition, strategy_name, xai_type) + self._params_cache_key(strategy_name, params)
        if metric_key in self._metrics_cache:
            return self._metrics_cache[metric_key]

        logs = self._run_and_get_logs_cached(
            pid=pid,
            condition=condition,
            strategy_name=strategy_name,
            hyperparams=params,
        )
        if logs is None:
            return 1e8

        evaluator = Evaluator(logs, num_parameters=len(params))
        metrics = evaluator.compute_metrics(session_num=session_num)
        nll = float(metrics.get(condition, {}).get(self.optimization_metric, 1e9))

        cond_metrics = metrics.get(condition, {})
        p_ai = cond_metrics.get("accuracy_participant_ai", np.nan)
        m_ai = cond_metrics.get("accuracy_model_ai", np.nan)

        self._metrics_cache[metric_key] = (nll, p_ai, m_ai)
        return nll, p_ai, m_ai


    # ---------------------------
    # Applicability logic (same as yours)
    # ---------------------------

    def _allowed_for_condition(self, model_name: str, xai_type: str, condition: str) -> bool:
        # Always-allowed baselines
        if model_name in ["MLP", "KNN", "DT"]:
            return True

        xai_type = (xai_type or "").lower()

        if xai_type == "none":
            return model_name == "Sensitive-features categorization"

        if xai_type == "importance":
            if condition == "w/ XAI":
                return True
            if condition == "w/o XAI":
                return model_name != "Importance categorization"
            return True

        if xai_type == "attribution":
            if condition == "w/ XAI":
                return model_name == "Attribution Sum"
            if condition == "w/o XAI":
                return model_name in ["Attribution Sum", "Sensitive-features categorization"]
            return True

        return False

    # ---------------------------
    # Participant row helpers
    # ---------------------------

    def _get_participant_row(self, pid) -> Optional[pd.Series]:
        df = self.participant_loader.df
        row = df[df["Participant ID"] == pid]
        if row.empty:
            return None
        return row.iloc[0]

    def _participant_xai_type(self, pid) -> str:
        r = self._get_participant_row(pid)
        if r is None:
            return "unknown"
        return (r.get("XAIType", "") or "").lower()

    # ---------------------------
    # Group Gaussian init
    # ---------------------------

    def _numeric_param_names(self, strategy_name: str) -> List[str]:
        """
        Keep only numeric dims (Real/Integer). Skip categorical dims and "explanation_type".
        """
        dims = self.param_spaces[strategy_name]
        names = []
        for d in dims:
            if d.name == "explanation_type":
                continue
            if isinstance(d, (Real, Integer)):
                names.append(d.name)
        return names

    def _init_group_gaussians(self, xai_types: List[str], conditions: List[str]):
        """
        Initialize mu at midpoints of each param range; Sigma diagonal small.
        """
        for strategy_name in self.models.keys():
            if strategy_name not in self.param_spaces:
                raise ValueError(f"param_spaces missing entry for strategy '{strategy_name}'")

        for xai_type in xai_types:
            for condition in conditions:
                for strategy_name in self.models.keys():
                    if not self._allowed_for_condition(strategy_name, xai_type, condition):
                        continue

                    dims = self.param_spaces[strategy_name]
                    dim_map = _dim_by_name(dims)

                    pnames = self._numeric_param_names(strategy_name)
                    if not pnames:
                        continue

                    mu0 = []
                    for pn in pnames:
                        mu0.append(float(_midpoint(dim_map[pn])))
                    mu0 = np.array(mu0, dtype=float)

                    # diagonal covariance
                    # set variance to (range/5.0)^2 for Reals, and 1 for Integers
                    var = []
                    for pn in pnames:
                        d = dim_map[pn]
                        if isinstance(d, Real):
                            rng = float(d.high) - float(d.low)
                            var.append(max((rng / 5.0) ** 2, 1e-3))
                        elif isinstance(d, Integer):
                            var.append(1.0)
                        else:
                            var.append(1.0)
                    Sigma0 = np.diag(np.array(var, dtype=float))

                    self.group_gaussians[(strategy_name, xai_type, condition)] = GroupGaussian(
                        param_names=pnames,
                        mu=mu0,
                        Sigma=Sigma0,
                    )

    # ---------------------------
    # Parameter vector <-> dict
    # ---------------------------

    def _vec_to_params(self, strategy_name: str, vec: np.ndarray) -> Dict[str, Any]:
        """
        Convert numeric vector to params dict and clip/round to bounds.
        """
        dims = self.param_spaces[strategy_name]
        dim_map = _dim_by_name(dims)

        pnames = self._numeric_param_names(strategy_name)
        assert len(pnames) == len(vec)

        params = {}
        for i, pn in enumerate(pnames):
            params[pn] = _clip_to_dim(vec[i], dim_map[pn])

        return params

    def _params_cache_key(self, strategy_name: str, params: Dict[str, Any]) -> Tuple[Any, ...]:
        """
        Cache key uses numeric params in fixed order, clipped/rounded.
        """
        dims = self.param_spaces[strategy_name]
        dim_map = _dim_by_name(dims)
        pnames = self._numeric_param_names(strategy_name)

        key = []
        for pn in pnames:
            key.append(_clip_to_dim(params[pn], dim_map[pn]))
        return tuple(key)

    def _propose_candidate_vecs(
        self,
        *,
        strategy_name: str,
        base_mu: np.ndarray,
        Sigma: np.ndarray,
        jitter_K: int,
        jitter_scale: float,
        cov_diagonal: bool,
    ) -> List[np.ndarray]:
        """
        Generate candidate parameter vectors around base_mu.
        Returns: [base_mu] + jittered vectors. No scoring here.

        If you call this ONCE and reuse the returned vecs for both sessions,
        both sessions use identical candidate points.
        """
        pnames = self._numeric_param_names(strategy_name)
        P = len(pnames)

        # base always included
        vecs = [np.array(base_mu, dtype=float)]

        if jitter_K <= 0:
            return vecs

        dims = self.param_spaces[strategy_name]
        dim_map = _dim_by_name(dims)

        # std per-dimension
        if cov_diagonal:
            std = np.sqrt(np.clip(np.diag(Sigma), 1e-8, None))
        else:
            std = np.zeros(P, dtype=float)
            for i, pn in enumerate(pnames):
                d = dim_map[pn]
                if isinstance(d, Real):
                    std[i] = (float(d.high) - float(d.low)) / 10.0
                else:
                    std[i] = 1.0

        std = jitter_scale * std

        # IMPORTANT: draw all z here once so sessions share the same jitters
        Z = self.rng.normal(0.0, 1.0, size=(jitter_K, P))
        for k in range(jitter_K):
            vecs.append(base_mu + Z[k] * std)

        return vecs


    def _best_from_candidate_vecs(
        self,
        *,
        pid,
        session_num: int,
        condition: str,
        strategy_name: str,
        cand_vecs: List[np.ndarray],
    ) -> Tuple[Dict[str, Any], float, float, float]:
        """
        Evaluate a fixed set of candidate vectors and return best params + best NLL.
        """
        best_params = None
        best_nll = float("inf")
        best_p_ai = float("nan")
        best_m_ai = float("nan")

        for vec in cand_vecs:
            params = self._vec_to_params(strategy_name, vec)
            nll, p_ai, m_ai = self._run_and_score(
                pid=pid,
                session_num=session_num,
                condition=condition,
                strategy_name=strategy_name,
                hyperparams=params,
            )
            if nll < best_nll:
                best_nll = float(nll)
                best_params = params
                best_p_ai = p_ai
                best_m_ai = m_ai

        return best_params, best_nll, best_p_ai, best_m_ai


    # ---------------------------
    # Cheap local search around group mean
    # ---------------------------

    def _best_near_mean(
        self,
        *,
        pid,
        session_num: int,
        condition: str,
        strategy_name: str,
        xai_type: str,
        base_mu: np.ndarray,
        jitter_K: int,
        jitter_scale: float,
        cov_diagonal: bool,
        Sigma: np.ndarray,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Evaluate NLL at base_mu and a few jittered points; return best params and best NLL.
        """
        # base
        base_params = self._vec_to_params(strategy_name, base_mu)
        best_params = dict(base_params)
        best_nll, _, _ = self._run_and_score(
            pid=pid,
            session_num=session_num,
            condition=condition,
            strategy_name=strategy_name,
            hyperparams=best_params,
        )

        if jitter_K <= 0:
            return best_params, best_nll

        dims = self.param_spaces[strategy_name]
        dim_map = _dim_by_name(dims)
        pnames = self._numeric_param_names(strategy_name)

        # choose jitter std:
        # - if cov_diagonal, use sqrt(diag(Sigma)) scaled
        # - else fall back to per-dim scale based on range
        if cov_diagonal:
            std = np.sqrt(np.clip(np.diag(Sigma), 1e-8, None))
        else:
            std = np.zeros(len(pnames), dtype=float)
            for i, pn in enumerate(pnames):
                d = dim_map[pn]
                if isinstance(d, Real):
                    std[i] = (float(d.high) - float(d.low)) / 10.0
                else:
                    std[i] = 1.0

        std = jitter_scale * std

        for _ in range(jitter_K):
            z = self.rng.normal(0.0, 1.0, size=len(pnames))
            cand_vec = base_mu + z * std
            cand_params = self._vec_to_params(strategy_name, cand_vec)

            nll, _, _ = self._run_and_score(
                pid=pid,
                session_num=session_num,
                condition=condition,
                strategy_name=strategy_name,
                hyperparams=cand_params,
            )
            if nll < best_nll:
                best_nll = nll
                best_params = cand_params

        return best_params, best_nll

    # ---------------------------
    # Main fitter
    # ---------------------------

    def fit(
        self,
        *,
        sessions: List[int],
        xai_types: Optional[List[str]] = None,
        conditions: Optional[List[str]] = None,
        n_rounds: int = 3,
        jitter_K: int = 8,
        jitter_scale: float = 0.5,
        soft: bool = True,
        beta: float = 3.0,
        cov_diagonal: bool = True,
        cov_floor: float = 1e-3,
        min_group_n: int = 5,
        reset_cache_each_round: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns:
          group_df: one row per (Strategy, XAIType, Tested w/ XAI) with mu + Sigma (diag)
          assign_df: one row per unit (Participant ID, Session, Tested w/ XAI) with chosen strategy + params
        """
        if xai_types is None:
            xai_types = ["none", "importance", "attribution"]
        if conditions is None:
            conditions = [c for c in self.available_conditions]

        # init
        self.group_gaussians.clear()
        self.assignments_rows.clear()
        self._score_cache.clear()
        self._init_group_gaussians(xai_types=xai_types, conditions=conditions)

        pids = self.participant_loader.list_all_participants()
        if self.max_participants is not None:
            pids = pids[: self.max_participants]

        for rnd in range(n_rounds):
            print(f"=== EM Round {rnd + 1} / {n_rounds} ===")
            if reset_cache_each_round:
                self._score_cache.clear()

            # collect weighted samples per group
            group_samples: Dict[Tuple[str, str, str], List[Tuple[np.ndarray, float]]] = {}
            group_counts: Dict[Tuple[str, str, str], int] = {}

            # we re-build assignments each round; keep only final after loop
            round_assignments = []

            for pid in pids:
                prow = self._get_participant_row(pid)
                if prow is None:
                    continue
                xai_type = (prow.get("XAIType", "") or "").lower()
                if xai_type not in xai_types:
                    continue

                app_id = prow.get("appId", None)
                # optionally also:
                exp_method = prow.get("expMethod", None)
                model_name = prow.get("modelName", None)

                # for session_num in sessions:
                for condition in conditions:
                    # candidate strategies for this xai+condition
                    candidates = [
                        s for s in self.models.keys()
                        if self._allowed_for_condition(s, xai_type, condition)
                    ]

                    if not candidates:
                        self._log(f"[WARN] No candidates for pid={pid}, cond={condition}, xai={xai_type}")
                        continue

                    cand_vecs_by_strategy = {}
                    for s in candidates:
                        gkey = (s, xai_type, condition)


                        gg = self.group_gaussians.get(gkey, None)
                        if gg is None:
                            self._log(f"[WARN] Missing group gaussian for {gkey} (skipping candidate)")
                            continue

                        cand_vecs_by_strategy[s] = self._propose_candidate_vecs(
                            strategy_name=s,
                            base_mu=gg.mu,
                            Sigma=gg.Sigma,
                            jitter_K=jitter_K,
                            jitter_scale=jitter_scale,
                            cov_diagonal=cov_diagonal,
                        )

                    for session_num in sessions:
                        nlls = []
                        best_params_per_candidate = []
                        used_candidates = []
                        p_ai_per_candidate = []
                        m_ai_per_candidate = []

                        for s, cand_vecs in cand_vecs_by_strategy.items():
                            # if s not in cand_vecs_by_strategy:
                            #     print("[WARN] Skipping candidate {} for pid={} cond={} xai={}".format(
                            #         s, pid, condition, xai_type
                            #     ))
                            
                            best_params, best_nll, best_p_ai, best_m_ai = self._best_from_candidate_vecs(
                                pid=pid,
                                session_num=session_num,
                                condition=condition,
                                strategy_name=s,
                                cand_vecs=cand_vecs_by_strategy[s],
                            )
                            nlls.append(best_nll)
                            best_params_per_candidate.append(best_params)
                            p_ai_per_candidate.append(best_p_ai)
                            m_ai_per_candidate.append(best_m_ai)
                            used_candidates.append(s)

                        if not nlls:
                            print("[WARN] No valid candidates after evaluation for pid={} sess={} cond={} xai={}".format(
                                pid, session_num, condition, xai_type
                            ))
                            continue

                        nlls_arr = np.array(nlls, dtype=float)
                        best_idx = int(np.argmin(nlls_arr))
                        

                        # tie logging (optional)
                        if len(nlls_arr) > 1:
                            order = np.argsort(nlls_arr)
                            best, second = nlls_arr[order[0]], nlls_arr[order[1]]
                            # if second - best < 0.2:
                            #     self._log(
                            #         f"[TIE?] pid={pid} sess={session_num} cond={condition} xai={xai_type} "
                            #         f"best={used_candidates[order[0]]}:{best:.3f} second={used_candidates[order[1]]}:{second:.3f}"
                            #     )
                            self._log(
                                f"pid={pid} sess={session_num} cond={condition} xai={xai_type} "
                                f"best={used_candidates[order[0]]}:{best:.3f} second={used_candidates[order[1]]}:{second:.3f}"
                            )


                        if soft:
                            # responsibilities
                            # stabilize by subtracting min to avoid overflow
                            delta = nlls_arr - nlls_arr.min()
                            w = np.exp(-beta * delta)
                            w = w / (w.sum() + 1e-12)

                            # save weighted samples for M-step
                            for j, s in enumerate(used_candidates):
                                gkey = (s, xai_type, condition)
                                gg = self.group_gaussians[gkey]
                                vec = np.array([best_params_per_candidate[j][pn] for pn in gg.param_names], dtype=float)

                                group_samples.setdefault(gkey, []).append((vec, float(w[j])))
                                group_counts[gkey] = group_counts.get(gkey, 0) + 1

                            chosen_s = used_candidates[best_idx]
                            chosen_params = best_params_per_candidate[best_idx]
                            chosen_nll = float(nlls_arr[best_idx])
                            chosen_pai = p_ai_per_candidate[best_idx]
                            chosen_mai = m_ai_per_candidate[best_idx]
                            # also store "confidence" in assignment
                            chosen_prob = float(w[best_idx])
                        else:
                            chosen_s = used_candidates[best_idx]
                            chosen_params = best_params_per_candidate[best_idx]
                            chosen_nll = float(nlls_arr[best_idx])
                            chosen_pai = p_ai_per_candidate[best_idx]
                            chosen_mai = m_ai_per_candidate[best_idx]
                            chosen_prob = 1.0

                            gkey = (chosen_s, xai_type, condition)
                            gg = self.group_gaussians[gkey]
                            vec = np.array([chosen_params[pn] for pn in gg.param_names], dtype=float)
                            group_samples.setdefault(gkey, []).append((vec, 1.0))
                            group_counts[gkey] = group_counts.get(gkey, 0) + 1

                        # record assignment row
                        round_assignments.append({
                            "Participant ID": pid,
                            "appId": app_id,
                            "expMethod": exp_method,
                            "modelName": model_name,
                            "Session": session_num,
                            "Tested w/ XAI": condition,
                            "XAIType": xai_type,
                            "Strategy": chosen_s,
                            "Assignment Prob": chosen_prob,
                            "NLL": chosen_nll,
                            **chosen_params,
                            "PAI": chosen_pai,
                            "MAI": chosen_mai,
                        })

            # M-step: update group Gaussians
            for gkey, samples in group_samples.items():
                if not samples:
                    continue
                n_used = group_counts.get(gkey, 0)
                if n_used < min_group_n:
                    continue

                X = np.stack([v for v, w in samples], axis=0)
                W = np.array([w for v, w in samples], dtype=float)
                W = W / (W.sum() + 1e-12)

                mu = (W[:, None] * X).sum(axis=0)
                Xc = X - mu[None, :]

                if cov_diagonal:
                    var = (W[:, None] * (Xc ** 2)).sum(axis=0)
                    var = np.maximum(var, cov_floor)
                    Sigma = np.diag(var)
                else:
                    Sigma = (Xc.T * W) @ Xc
                    Sigma = Sigma + cov_floor * np.eye(Sigma.shape[0])

                gg = self.group_gaussians[gkey]
                gg.mu = mu
                gg.Sigma = Sigma
                self.group_gaussians[gkey] = gg

            # keep only last round assignments
            self.assignments_rows = round_assignments

        group_df = self.group_params_dataframe()
        assign_df = pd.DataFrame(self.assignments_rows)
        return group_df, assign_df

    # ---------------------------
    # Export helpers
    # ---------------------------

    def group_params_dataframe(self) -> pd.DataFrame:
        rows = []
        for (strategy, xai_type, condition), gg in self.group_gaussians.items():
            r = {
                "Strategy": strategy,
                "XAIType": xai_type,
                "Tested w/ XAI": condition,
            }
            for i, pn in enumerate(gg.param_names):
                r[f"mu_{pn}"] = float(gg.mu[i])
                r[f"var_{pn}"] = float(gg.Sigma[i, i])
            rows.append(r)
        return pd.DataFrame(rows)

    def save_results(self, out_dir: str, prefix: str = "pop_em") -> Tuple[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")

        group_path = os.path.join(out_dir, f"{prefix}_group_gaussians_{ts}.csv")
        assign_path = os.path.join(out_dir, f"{prefix}_assignments_{ts}.csv")

        self.group_params_dataframe().to_csv(group_path, index=False)
        pd.DataFrame(self.assignments_rows).to_csv(assign_path, index=False)

        return group_path, assign_path

    # ---------------------------
    # Sampling for simulation
    # ---------------------------

    # def sample_params(
    #     self,
    #     *,
    #     strategy: str,
    #     xai_type: str,
    #     condition: str,
    #     n: int = 1,
    # ) -> List[Dict[str, Any]]:
    #     """
    #     Sample hyperparams from learned group Gaussian.
    #     Note: integer params (e.g., k) are rounded/clipped to bounds.
    #     """
    #     gkey = (strategy, xai_type.lower(), condition)
    #     if gkey not in self.group_gaussians:
    #         raise KeyError(f"No group gaussian for {gkey}")

    #     gg = self.group_gaussians[gkey]
    #     # diagonal covariance assumed by default; still ok if full
    #     draws = self.rng.multivariate_normal(mean=gg.mu, cov=gg.Sigma, size=n)

    #     out = []
    #     for i in range(n):
    #         params = self._vec_to_params(strategy, draws[i])
    #         if "decay_param" not in params:
    #             params["decay_param"] = 0.5
    #         if strategy == "Attribution Sum":
    #             params["explanation_type"] = xai_type.lower()
    #         out.append(params)
    #     return out

    def sample_params(self, *, strategy: str, xai_type: str, condition: str, n: int = 1):
        gkey = (strategy, xai_type.lower(), condition)
        if gkey not in self.group_gaussians:
            raise KeyError(f"No group gaussian for {gkey}")

        gg0 = self.group_gaussians[gkey]

        # --- REALIGN to strategy param order ---
        strategy_pnames = self._numeric_param_names(strategy)
        if gg0.param_names != strategy_pnames:
            mu_map = {pn: gg0.mu[i] for i, pn in enumerate(gg0.param_names)}
            var_map = {pn: gg0.Sigma[i, i] for i, pn in enumerate(gg0.param_names)}

            mu = np.array([float(mu_map[pn]) for pn in strategy_pnames], dtype=float)
            var = np.array([max(float(var_map[pn]), 1e-8) for pn in strategy_pnames], dtype=float)
            Sigma = np.diag(var)

            gg = GroupGaussian(param_names=strategy_pnames, mu=mu, Sigma=Sigma)
        else:
            gg = gg0

        draws = self.rng.multivariate_normal(mean=gg.mu, cov=gg.Sigma, size=n)

        out = []
        for i in range(n):
            params = self._vec_to_params(strategy, draws[i])
            if "decay_param" not in params:
                params["decay_param"] = 0.5
            if strategy == "Attribution Sum":
                params["explanation_type"] = xai_type.lower()
            out.append(params)
        return out


    def load_group_gaussians(self, group_csv_path: str) -> None:
        """
        Load group Gaussian parameters from a CSV previously saved by save_results().
        Expected columns:
        Strategy, XAIType, Tested w/ XAI, mu_<param>, var_<param>
        Reconstructs diagonal covariance matrices.

        After calling this, you can call assign_all_participants_* without running EM.
        """
        df = pd.read_csv(group_csv_path)
        print(df.head())
        required = {"Strategy", "XAIType", "Tested w/ XAI"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"group csv missing required columns: {missing}")

        self.group_gaussians.clear()

        for _, r in df.iterrows():
            strategy = r["Strategy"]
            xai_type = (r["XAIType"] or "").lower()
            condition = r["Tested w/ XAI"]

            # collect param names from columns
            mu_cols = [c for c in df.columns if c.startswith("mu_")]
            var_cols = [c for c in df.columns if c.startswith("var_")]

            # Determine param names present for THIS row (some strategies differ)
            param_names = []
            mu = []
            var = []
            for c in mu_cols:
                pn = c[len("mu_"):]
                vc = f"var_{pn}"
                if vc not in df.columns:
                    continue
                # skip NaNs (e.g., if CSV has columns for other strategies)
                if pd.isna(r[c]) or pd.isna(r[vc]):
                    continue
                param_names.append(pn)
                mu.append(float(r[c]))
                var.append(max(float(r[vc]), 1e-8))

            if not param_names:
                # strategy may have no numeric params; skip
                continue

            mu = np.array(mu, dtype=float)
            Sigma = np.diag(np.array(var, dtype=float))

            self.group_gaussians[(strategy, xai_type, condition)] = GroupGaussian(
                param_names=param_names,
                mu=mu,
                Sigma=Sigma,
            )

    def _best_near_mean_deterministic(
        self,
        *,
        pid,
        session_num: int,
        condition: str,
        strategy_name: str,
        base_mu: np.ndarray,
        Sigma: np.ndarray,
        max_evals: int = 9,
        step_scale: float = 1.0,
        shrink: float = 0.5,
        min_step_frac: float = 0.05,
    ):
        dims = self.param_spaces[strategy_name]
        dim_map = _dim_by_name(dims)
        pnames = self._numeric_param_names(strategy_name)
        P = len(pnames)

        diag = np.diag(Sigma) if Sigma.ndim == 2 else np.array(Sigma, dtype=float)
        natural = np.zeros(P, dtype=float)
        for i, pn in enumerate(pnames):
            d = dim_map[pn]
            if i < len(diag) and np.isfinite(diag[i]) and diag[i] > 0:
                natural[i] = np.sqrt(diag[i])
            else:
                natural[i] = (float(d.high) - float(d.low)) / 10.0 if isinstance(d, Real) else 1.0

        for i, pn in enumerate(pnames):
            if natural[i] < 1e-8:
                d = dim_map[pn]
                natural[i] = (float(d.high) - float(d.low)) / 10.0 if isinstance(d, Real) else 1.0

        step0 = step_scale * natural
        min_step = min_step_frac * step0

        # --- helper to eval a candidate vec ---
        def eval_vec(vec):
            params = self._vec_to_params(strategy_name, vec)
            nll, pai, mai = self._run_and_score(
                pid=pid,
                session_num=session_num,
                condition=condition,
                strategy_name=strategy_name,
                hyperparams=params,
            )
            return params, float(nll), float(pai), float(mai)

        # 1) evaluate base
        evals = 0
        current = np.array(base_mu, dtype=float)
        best_params, best_nll, best_pai, best_mai = eval_vec(current)
        evals += 1

        step = step0.copy()
        order = np.argsort(-step)

        while evals < max_evals:
            improved = False

            for idx in order:
                if evals >= max_evals:
                    break
                if step[idx] < min_step[idx]:
                    continue

                # try plus
                cand = current.copy()
                cand[idx] += step[idx]
                cand_params, cand_nll, cand_pai, cand_mai = eval_vec(cand)
                evals += 1
                if cand_nll + 1e-12 < best_nll:
                    best_nll, best_params = cand_nll, cand_params
                    best_pai, best_mai = cand_pai, cand_mai
                    current = np.array([best_params[pn] for pn in pnames], dtype=float)
                    improved = True
                    continue

                if evals >= max_evals:
                    break

                # try minus
                cand = current.copy()
                cand[idx] -= step[idx]
                cand_params, cand_nll, cand_pai, cand_mai = eval_vec(cand)
                evals += 1
                if cand_nll + 1e-12 < best_nll:
                    best_nll, best_params = cand_nll, cand_params
                    best_pai, best_mai = cand_pai, cand_mai
                    current = np.array([best_params[pn] for pn in pnames], dtype=float)
                    improved = True
                    continue

            if not improved:
                step = shrink * step
                if np.all(step < min_step):
                    break
                order = np.argsort(-step)

        # return pai/mai too
        return best_params, best_nll, best_pai, best_mai

    # ------------------------------------------------------------------
    # Update assign_all_participants to support search="deterministic"
    # ------------------------------------------------------------------
    def assign_all_participants(
        self,
        *,
        sessions: List[int],
        conditions: Optional[List[str]] = None,
        xai_types: Optional[List[str]] = None,
        subset_pids: Optional[List[Any]] = None,
        search: str = "mean",
        max_evals_per_strategy: int = 9,
        step_scale: float = 1.0,
        shrink: float = 0.5,
        min_step_frac: float = 0.05,
        soft: bool = False,
        beta: float = 3.0,
        reset_cache: bool = True,
    ) -> pd.DataFrame:
        if not self.group_gaussians:
            raise RuntimeError("group_gaussians is empty. Call load_group_gaussians() first.")

        if conditions is None:
            conditions = [c for c in self.available_conditions]
        if xai_types is None:
            xai_types = ["none", "importance", "attribution"]

        if reset_cache:
            self._score_cache.clear()

        if subset_pids is None:
            pids = self.participant_loader.list_all_participants()
            if self.max_participants is not None:
                pids = pids[: self.max_participants]
        else:
            pids = subset_pids

        rows = []

        for i, pid in enumerate(pids):
            print(f"Processing participant {i+1}/{len(pids)}: {pid}")
            prow = self._get_participant_row(pid)
            if prow is None:
                continue
            xai_type = (prow.get("XAIType", "") or "").lower()
            if xai_type not in xai_types:
                continue

            app_id = prow.get("appId", None)
            exp_method = prow.get("expMethod", None)
            model_name = prow.get("modelName", None)

            for session_num in sessions:
                for condition in conditions:
                    candidates = [s for s in self.models.keys() if self._allowed_for_condition(s, xai_type, condition)]
                    if not candidates:
                        continue

                    nlls = []
                    best_params_per_candidate = []
                    pai_per_candidate = []
                    mai_per_candidate = []

                    for s in candidates:
                        gkey = (s, xai_type, condition)
                        gg = self.group_gaussians.get(gkey, None)
                        if gg is None:
                            continue

                        # align gaussian ordering if needed (same as your code)
                        strategy_pnames = self._numeric_param_names(s)
                        if gg.param_names != strategy_pnames:
                            mu_map = {pn: gg.mu[i] for i, pn in enumerate(gg.param_names)}
                            var_map = {pn: gg.Sigma[i, i] for i, pn in enumerate(gg.param_names)}
                            mu_vec = np.array([float(mu_map.get(pn, 0.0)) for pn in strategy_pnames], dtype=float)

                            var_vec = []
                            dim_map = _dim_by_name(self.param_spaces[s])
                            for pn in strategy_pnames:
                                if pn in var_map:
                                    var_vec.append(max(float(var_map[pn]), 1e-8))
                                else:
                                    d = dim_map[pn]
                                    if isinstance(d, Real):
                                        rng = float(d.high) - float(d.low)
                                        var_vec.append(max((rng / 10.0) ** 2, 1e-3))
                                    else:
                                        var_vec.append(1.0)
                            Sigma_mat = np.diag(np.array(var_vec, dtype=float))
                            base_mu, Sigma_use = mu_vec, Sigma_mat
                        else:
                            base_mu, Sigma_use = gg.mu, gg.Sigma

                        if search == "mean":
                            params = self._vec_to_params(s, base_mu)
                            nll, pai, mai = self._run_and_score(
                                pid=pid,
                                session_num=session_num,
                                condition=condition,
                                strategy_name=s,
                                hyperparams=params,
                            )
                            best_params, best_nll = params, float(nll)
                            best_pai, best_mai = float(pai), float(mai)

                        elif search == "deterministic":
                            best_params, best_nll, best_pai, best_mai = self._best_near_mean_deterministic(
                                pid=pid,
                                session_num=session_num,
                                condition=condition,
                                strategy_name=s,
                                base_mu=base_mu,
                                Sigma=Sigma_use,
                                max_evals=max_evals_per_strategy,
                                step_scale=step_scale,
                                shrink=shrink,
                                min_step_frac=min_step_frac,
                            )
                        else:
                            raise ValueError("search must be 'mean' or 'deterministic'")

                        nlls.append(float(best_nll))
                        best_params_per_candidate.append(best_params)
                        pai_per_candidate.append(float(best_pai))
                        mai_per_candidate.append(float(best_mai))

                    if not nlls:
                        continue

                    nlls_arr = np.array(nlls, dtype=float)
                    best_idx = int(np.argmin(nlls_arr))

                    if soft:
                        delta = nlls_arr - nlls_arr.min()
                        w = np.exp(-beta * delta)
                        w = w / (w.sum() + 1e-12)
                        chosen_prob = float(w[best_idx])
                    else:
                        chosen_prob = 1.0

                    chosen_s = candidates[best_idx]
                    chosen_params = best_params_per_candidate[best_idx]
                    chosen_nll = float(nlls_arr[best_idx])
                    chosen_pai = pai_per_candidate[best_idx]
                    chosen_mai = mai_per_candidate[best_idx]

                    rows.append({
                        "Participant ID": pid,
                        "appId": app_id,
                        "expMethod": exp_method,
                        "modelName": model_name,
                        "Session": session_num,
                        "Tested w/ XAI": condition,
                        "XAIType": xai_type,
                        "Strategy": chosen_s,
                        "Assignment Prob": chosen_prob,
                        "NLL": chosen_nll,
                        "PAI": chosen_pai,
                        "MAI": chosen_mai,
                        **chosen_params,
                    })

        return pd.DataFrame(rows)





    # ---------------------------
    # Core evaluation
    # ---------------------------

    # def _run_and_score(
    #     self,
    #     *,
    #     pid,
    #     session_num: int,
    #     condition: str,
    #     strategy_name: str,
    #     hyperparams: Dict[str, Any],
    # ) -> float:
    #     """
    #     Run one participant once and score NLL for given session+condition.
    #     Uses a cache to avoid repeated identical evaluations.
    #     """
    #     prow = self._get_participant_row(pid)
    #     if prow is None:
    #         return float("inf")

    #     app_id = prow["appId"]
    #     exp_method = prow["expMethod"]
    #     ai_model_name = prow["modelName"]
    #     xai_type = (prow.get("XAIType", "") or "").lower()

    #     # enforce applicability
    #     if not self._allowed_for_condition(strategy_name, xai_type, condition):
    #         return 1e8

    #     # ensure required fixed params
    #     params = dict(hyperparams)
    #     if "decay_param" not in params:
    #         params["decay_param"] = 0.5

    #     if strategy_name == "Attribution Sum":
    #         params["explanation_type"] = xai_type  # required by your model

    #     # cache
    #     cache_key = (pid, session_num, condition, strategy_name, xai_type) + self._params_cache_key(strategy_name, params)
    #     if cache_key in self._score_cache:
    #         return self._score_cache[cache_key]

    #     ai_loader = self.loader_getter(app_id, exp_method, ai_model_name, xai_type)

    #     human_model_class = self.models[strategy_name]
    #     human_model = human_model_class(**params)

    #     runner = ParticipantExperimentRunner(
    #         human_model=human_model,
    #         ai_dataset_loader=ai_loader,
    #         participant_dataset_loader=self.participant_loader,
    #         ui=self.ui,
    #     )
    #     human_model.time = runner.time

    #     logs = runner.run_experiment(pid)

    #     evaluator = Evaluator(logs, num_parameters=len(params))
    #     metrics = evaluator.compute_metrics(session_num=session_num)

    #     nll = float(metrics.get(condition, {}).get(self.optimization_metric, 1e9))
    #     self._score_cache[cache_key] = nll
    #     return nll
    
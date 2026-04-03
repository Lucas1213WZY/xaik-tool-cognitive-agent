import numpy as np

# ----- unchanged binning logic (kept as-is) -----
def bin_relative_importance(partial_contrib, all_partials):
    total_abs = np.sum(np.abs(all_partials))
    if total_abs == 0:
        return "low", 0.5
    percentage = (abs(partial_contrib) / total_abs) * 100
    if percentage < 33:
        return "low", 0.5
    elif percentage < 67:
        return "medium", 1.0
    else:
        return "high", 1.5
import numpy as np

# ========== BUILD/STORE (no explicit digit storage) ==========
def add_lr_heuristic_to_memory(lr_exp, memory, initial_var: float = 1.0):
    """Store probabilistic chunks using ONLY (mu, var)."""
    import numpy as np

    intercept_mu = np.sign(lr_exp.intercept) if lr_exp.intercept != 0 else 0.0
    memory.add_chunk(
        "LR_intercept_prob",
        {"type": "intercept_prob", "mu": float(0), "var": float(initial_var)}
    )

    for key, coef in lr_exp.coefficients.items():
        mu_coef = np.sign(coef) if coef != 0 else 0.0
        memory.add_chunk(
            f"LR_coef_prob_{key.replace('=', '_')}",
            {
                "type": "coef_prob",
                "feature_key": key,
                "feature_name": lr_exp._format_feature(key),
                "mu": float(mu_coef),
                "var": float(initial_var),
            },
        )


def _draw_from_topk(ret, rng):
    """ret from topk_retrievals_with_prob_refresh; return chosen chunk or None."""
    top = ret["top_k"]  # [(chunk, p), ...]
    p_none = float(ret["p_none"])
    choices = [ch for ch, _ in top] + [None]
    probs   = [float(p) for _, p in top] + [p_none]
    probs   = np.array(probs, dtype=float)
    probs   = probs / probs.sum() if probs.sum() > 0 else np.array([1.0] + [0.0]*(len(choices)-1))
    idx = rng.choice(len(choices), p=probs)
    return choices[idx]

from .lr_memory import ddm_prob_rt, evidence_lr_divnorm
from typing import Dict, Any, Tuple, List, Optional
import math

def _base_idx_from_key(key: str) -> int:
    """'x3' -> 3, 'x3=1' -> 3."""
    return int(key.split('=')[0][1:])

import numpy as np

def _draw_from_topk(ret, rng):
    """
    ret from topk_retrievals_with_prob_refresh; return chosen chunk or None.
    Matches semantics:
      - choices = [chunks..., None]
      - probs   = [p_chunk..., p_none]
      - if sum(probs) == 0 -> pick first choice deterministically
    """
    top = ret.get("top_k", [])            # [(chunk, p), ...]
    p_none = float(ret.get("p_none", 0.0))

    choices = [ch for ch, _ in top] + [None]
    probs = [float(p) for _, p in top] + [p_none]
    probs = np.array(probs, dtype=float)

    # sanitize any NaNs/negatives
    probs[~np.isfinite(probs)] = 0.0
    probs[probs < 0] = 0.0

    s = probs.sum()
    if s > 0.0:
        probs = probs / s
    else:
        # exact fallback behavior: pick index 0 with prob 1
        probs = np.array([1.0] + [0.0] * (len(choices) - 1), dtype=float)

    idx = rng.choice(len(choices), p=probs)
    return choices[idx]

def _read_value_from_key(key: str, x: np.ndarray) -> Tuple[float, bool, int]:
    """
    Return (value, is_numeric, col_index) given feature key and raw feature vector x.
    - For categorical keys like 'x3=1': returns 0/1 indicator, is_numeric=False.
    - For numeric keys like 'x3': returns x[3], is_numeric=True.
    """
    if '=' in key:
        base, cat = key.split('=')
        col = int(base[1:])
        val = 1.0 if int(x[col]) == int(cat) else 0.0
        return float(val), False, col
    col = int(key[1:])
    return float(x[col]), True, col

# --------------------------- Main function -----------------------------------

def lr_heuristic(
    feature_vector,
    memory,
    lr_exp,
    *,
    num_samples: int = 40,
    K_top: int = 3,                 # top-k to consider per retrieval
    T_enc: float = 2.0,             # per numeric/categorical read
    T_INTUITIVE_OP: float = 0.5,    # per internal +/* cost
    # DDM params
    ddm_a: float = 1.5,
    ddm_s: float = 1.0,
    ddm_Tnd: float = 0.30,
    ddm_norm: str = "l2",
    active_indices: Optional[List[int]] = None,
    verbose: bool = False,
):
    """
    Heuristic LR with memory-based noisy retrieval:
      - Retrieve intercept and each coefficient distribution via memory.topk_retrievals_with_prob_refresh.
      - Read each feature value once (deterministic).
      - Monte Carlo `num_samples`: sample intercept/coefficients, form terms, run DDM.
      - Time = sum(retrieval expected RTs) + reads + intuitive ops + mean DDM RT.

    Returns:
      probs: np.array([p0, p1])
      total_time: float
      info: dict with minimal diagnostics
    """
    rng = np.random.default_rng()
    x = np.asarray(feature_vector, dtype=float)
    active_set = set(active_indices) if active_indices is not None else None

    # ---------------- Retrieval of intercept and coefficients -----------------
    r_int = memory.topk_retrievals_with_prob_refresh(
        {"type": "intercept_prob"}, k=K_top, refresh_prob=1.0, add_refresh=True
    )
    retrieval_cost = float(r_int.get("expected_rt", 0.0))

    feat_meta = []  # list of (key, value, is_numeric, r_coef)
    for key in lr_exp.coefficients.keys():
        if active_set and _base_idx_from_key(key) not in active_set:
            continue

        r_coef = memory.topk_retrievals_with_prob_refresh(
            {"type": "coef_prob", "feature_key": key},
            k=K_top, refresh_prob=1.0, add_refresh=True
        )
        retrieval_cost += float(r_coef.get("expected_rt", 0.0))

        val, is_numeric, _ = _read_value_from_key(key, x)
        feat_meta.append((key, float(val), bool(is_numeric), r_coef))

    # Read cost: one feature value read per feature (intercept has no read)
    read_cost = T_enc * len(feat_meta)

    # ------------------------- Monte Carlo sampling ---------------------------
    p1_list, rt_list, v_list = [], [], []
    for _ in range(num_samples):
        terms = []

        # Intercept sample
        ch_int = _draw_from_topk(r_int, rng)
        if ch_int is None:
            mu_i, var_i = 0.0, 0.01
        else:
            mu_i = float(ch_int.slots.get("mu", 0.0))
            var_i = float(ch_int.slots.get("var", ch_int.slots.get("sigma", 0.0) ** 2))
        sigma_i = math.sqrt(max(var_i, 1e-12))
        intercept_sample = rng.normal(mu_i, sigma_i)
        terms.append(intercept_sample)

        # Coef * value samples
        for key, val, _is_num, r_coef in feat_meta:
            ch = _draw_from_topk(r_coef, rng)
            if ch is None:
                mu_c, var_c = 0.0, 0.0
            else:
                mu_c = float(ch.slots.get("mu", 0.0))
                var_c = float(ch.slots.get("var", ch.slots.get("sigma", 0.0) ** 2))
            coef = rng.normal(mu_c, math.sqrt(max(var_c, 0.0))) if var_c > 0.0 else mu_c
            terms.append(coef * val)

        # DDM per sample
        evidence = evidence_lr_divnorm(terms, mode=ddm_norm)
        p_up, rt_dec, v_val = ddm_prob_rt(evidence, a=ddm_a, s=ddm_s, Tnd=ddm_Tnd, gain=1.0)
        p1_list.append(p_up)
        rt_list.append(rt_dec)
        v_list.append(v_val)

    p1 = float(np.mean(p1_list)) if p1_list else 0.5
    probs = np.array([1.0 - p1, p1], dtype=float)

    intuitive_ops = len(feat_meta)  # simple count of “internal ops”
    computation_cost = T_INTUITIVE_OP * max(0, intuitive_ops)
    ddm_rt_mean = float(np.mean(rt_list)) if rt_list else 0.0

    total_time = retrieval_cost + read_cost + computation_cost + ddm_rt_mean

    memory.tick(total_time)

    info = {
        "decision": {
            "p1": p1,
            "v_ratio_mean": float(np.mean(v_list)) if v_list else 0.0,
        },
        "timing": {
            "retrieval_rt_sum": retrieval_cost,
            "read_time_sum": read_cost,
            "intuitive_ops": intuitive_ops,
            "intuitive_op_cost": T_INTUITIVE_OP,
            "ddm_rt_mean": ddm_rt_mean,
            "total_time": total_time,
        },
        "chunks": {
            # RESTORED to original shape expected by `refresh_lr_heuristic_in_memory`
            "intercept": {
                "chosen_name": (r_int.get("top_k", [])[0][0].name if r_int.get("top_k") else None),
            },
            "features": [
                {
                    "key": key,
                    "value": float(val),
                    "chosen_name": (r_coef.get("top_k", [])[0][0].name if r_coef.get("top_k") else None),
                    "is_numeric": bool(is_numeric),
                }
                for (key, val, is_numeric, r_coef) in feat_meta
            ],
        },
    }

    if verbose:
        print(f"[LR_HEUR] p1={p1:.4f}, total_time={total_time:.3f}s "
            f"(retrieval={retrieval_cost:.3f}, read={read_cost:.3f}, "
            f"ops={intuitive_ops}×{T_INTUITIVE_OP:.2f}, ddm={ddm_rt_mean:.3f})")

    return probs, total_time, info


def refresh_lr_heuristic_in_memory(
    memory,
    lr_exp,
    info,                 # ← from predict()
    actual: int,
    *,
    active_indices: list[int] = None,  # optional feature filter
    w_min: float = 1e-4,                      # min curvature p(1-p) for stability
    verbose: bool = False,
):
    """
    Incremental Bayesian logistic update with diagonal covariance, storing only (mu, var).
    No process noise / forgetting is applied.

      w = p*(1-p)  (clipped at w_min)
      lambda_post = 1/var + w * x^2
      var_post    = 1/lambda_post
      mu_post     = mu + var_post * x * (y - p)

    Only updates:
      • the retrieved intercept (if any)
      • retrieved coefficient chunks whose base feature index is in active_indices (if provided)
    """
    import math

    y = int(actual)
    p = float(info["decision"]["p1"])
    active_set = set(active_indices) if active_indices is not None else None

    def _upd(mu, var, xj, y, p, w_min):
        var = max(float(var), 1e-12)          # ensure positive
        w = p * (1.0 - p)
        if w < w_min:
            w = w_min
        lam_post = (1.0 / var) + w * (xj * xj)
        var_new  = 1.0 / lam_post
        mu_new   = float(mu) + var_new * xj * (y - p)
        return mu_new, var_new

    # ---- Intercept (x0 = 1) ----
    cint_name = info["chunks"]["intercept"]["chosen_name"]
    if cint_name:
        ch = memory.get_chunk(cint_name)
        if ch:
            mu  = float(ch.slots.get("mu", 0.0))
            var = float(ch.slots.get("var", 1.0))
            mu_new, var_new = _upd(mu, var, 1.0, y, p, w_min)
            ch.slots["mu"]  = mu_new
            ch.slots["var"] = var_new
            # if verbose:
            #     print(f"[INT] {cint_name}: mu {mu:.4f}->{mu_new:.4f}, sd {math.sqrt(var):.4f}->{math.sqrt(var_new):.4f}")

    # ---- Coefficients (only retrieved per feature, filtered by active_indices) ----
    for f in info["chunks"]["features"]:
        cname = f.get("chosen_name")
        if not cname:
            continue

        if active_set is not None:
            key = f.get("key", "")        # e.g., "x7" or "x7=2"
            base = key.split('=')[0]      # "x7"
            try:
                base_idx = int(base[1:])  # 7
            except Exception:
                base_idx = None
            if (base_idx is None) or (base_idx not in active_set):
                continue

        ch = memory.get_chunk(cname)
        if not ch:
            continue

        xj  = float(f["value"])
        mu  = float(ch.slots.get("mu", 0.0))
        var = float(ch.slots.get("var", 1.0))

        mu_new, var_new = _upd(mu, var, xj, y, p, w_min)
        ch.slots["mu"]  = mu_new
        ch.slots["var"] = var_new

        # if verbose:
        #     print(f"[COEF] {cname}: mu {mu:.4f}->{mu_new:.4f}, sd {math.sqrt(var):.4f}->{math.sqrt(var_new):.4f}, x={xj:.4f}")

    memory.tick(20)



# # ======================================================== #

# from typing import Dict, Tuple, List, Any, Optional
# import numpy as np, math


# def _sf_halfwidth(v: float, sf: int = 2) -> float:
#     """Half width of the rounding bin for v at 'sf' significant figures."""
#     a = abs(v)
#     if a == 0:
#         # choose a tiny neighborhood when v == 0 (keeps code robust)
#         return 0.5 * 10 ** (-sf)
#     k = math.floor(math.log10(a))
#     unit = 10 ** (k - (sf - 1))  # e.g., sf=2 => 10^(k-1)
#     return 0.5 * unit

# def _slider_step(bounds: Tuple[float, float]) -> float:
#     lo, hi = bounds
#     r = max(hi - lo, 1e-12)
#     return 10 ** (math.floor(math.log10(r)) - 1)

# def _snap_to_step(x: float, bounds: Tuple[float, float], step: float) -> float:
#     lo, hi = bounds
#     x = min(max(x, lo), hi)
#     # Snap to nearest step from 'lo' (consistent slider behavior)
#     return lo + round((x - lo) / step) * step

# def _mental_division_time(a: float, b: float, compute_sf: int = 2, ability: float = 1.0) -> float:
#     """Very simple time proxy if you don't want to hook your ACT-R timing:
#     scales with difficulty and requested significant figures."""
#     a, b = float(a), float(b)
#     mag = 1.0 + math.log10(1.0 + abs(a)) + math.log10(1.0 + abs(b))
#     return (compute_sf / max(ability, 1e-6)) * 0.25 * mag  # seconds-ish

from .lr_memory import _base_index_from_key, _slider_step, _snap_to_step, round_to_sf

def _denorm(xn: float, lo: float, hi: float) -> float:
    """Map normalized value in [0,1] to original units using bounds."""
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float(xn)
    return float(lo + xn * (hi - lo))

def _norm_delta_to_orig(dxn: float, lo: float, hi: float) -> float:
    """Convert normalized delta to original-units delta via bounds."""
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float(dxn)
    return float(dxn * (hi - lo))

def _draw_from_topk(ret: Dict[str, Any], rng: np.random.Generator):
    top = ret.get("top_k", [])
    p_none = float(ret.get("p_none", 0.0))
    choices = [ch for ch, _ in top] + [None]
    probs   = [float(p) for _, p in top] + [p_none]
    probs   = np.asarray(probs, float)
    probs[~np.isfinite(probs)] = 0.0
    s = probs.sum()
    if s > 0.0:
        probs /= s
    else:
        probs = np.array([1.0] + [0.0] * (len(choices) - 1), float)
    idx = rng.choice(len(choices), p=probs)
    return choices[idx]
def cf_lr_heuristic(
    feature_vector,                   # NORMALIZED: numerics in [0,1]
    memory,
    lr_exp,
    bounds: Dict[str, Tuple[float, float]],   # per numeric key, e.g., "x3": (lo, hi)
    *,
    K_top: int = 3,
    T_enc: float = 2.0,
    T_INTUITIVE_OP: float = 0.5,
    value_display_sf: int = 2,       # only the final sum is rounded
    active_indices: Optional[List[int]] = None,
    feas_leeway_norm: float = 2.0,   # allow [-feas_leeway_norm, 1+feas_leeway_norm]
    y_actual: Optional[int] = None,  # NEW: desired label (0/1). If provided and sign disagrees with z_user, invert direction.
    verbose: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Forward-then-CF with NORMALIZED inputs.
    If y_actual is provided (0/1) and sign(z_user) ≠ (−1 for 0, +1 for 1), the CF direction is inverted:
      - Numerics: use delta_norm = (+ z_user / coef) instead of (− z_user / coef)
      - Leeway/boundary push: desired_up is flipped
      - Categoricals: invert default flip policy
    """
    rng = np.random.default_rng()
    x_norm = np.asarray(feature_vector, dtype=float)
    active_set = set(active_indices) if active_indices is not None else None

    # ---------- Retrieval ----------
    r_int = memory.topk_retrievals_with_prob_refresh(
        {"type": "intercept_prob"}, k=K_top, refresh_prob=1.0, add_refresh=True
    )
    retrieval_cost = float(r_int.get("expected_rt", 0.0))

    feat_meta: List[Tuple[str, float, bool, Dict[str, Any]]] = []  # (key, val_norm, is_numeric, r_coef)
    coef_stats: Dict[str, Tuple[float, float]] = {}

    for key in lr_exp.coefficients.keys():
        if active_set and _base_idx_from_key(key) not in active_set:
            continue

        r_coef = memory.topk_retrievals_with_prob_refresh(
            {"type": "coef_prob", "feature_key": key}, k=K_top, refresh_prob=1.0, add_refresh=True
        )
        retrieval_cost += float(r_coef.get("expected_rt", 0.0))

        # Read normalized value once:
        if '=' in key:
            base, cat = key.split('=')
            col = int(base[1:])
            val_norm = float(x_norm[col])   # assumes indicator in {0,1}
            is_numeric = False
        else:
            col = int(key[1:])
            val_norm = float(x_norm[col])   # numeric already normalized 0..1
            is_numeric = True

        feat_meta.append((key, val_norm, is_numeric, r_coef))

        ch = _draw_from_topk(r_coef, rng)
        if ch is None:
            mu_c, var_c = 0.0, 1.0
        else:
            mu_c = float(ch.slots.get("mu", 0.0))
            var_c = float(ch.slots.get("var", ch.slots.get("sigma", 0.0) ** 2))
        coef_stats[key] = (mu_c, var_c)

    # One deterministic read per feature (forward-like timing)
    read_cost = T_enc * len(feat_meta)

    # ---------- ONE forward-simulation sample on normalized x ----------
    ch_int = _draw_from_topk(r_int, rng)
    if ch_int is None:
        mu_i, var_i = 0.0, 0.01
    else:
        mu_i = float(ch_int.slots.get("mu", 0.0))
        var_i = float(ch_int.slots.get("var", ch_int.slots.get("sigma", 0.0) ** 2))
    sigma_i = math.sqrt(max(var_i, 1e-12))
    intercept_sample = rng.normal(mu_i, sigma_i)

    z_user = float(intercept_sample)
    sampled_coefs: Dict[str, float] = {}
    for key, val_norm, _is_num, _r_coef in feat_meta:
        mu_c, var_c = coef_stats.get(key, (0.0, 0.0))
        coef = mu_c   # mean for stability
        sampled_coefs[key] = float(coef)
        z_user += float(coef) * float(val_norm)

    # ----- NEW: compute alignment with y_actual -----
    def _sign(v: float) -> int:
        return 1 if v >= 0.0 else -1

    flip_direction = False  # whether to invert the default direction
    if y_actual is not None:
        # y_actual = float(y_actual) - 0.5
        actual_y = 1 if int(y_actual) == 1 else -1
        flip_direction = (_sign(z_user) != actual_y)

    # ---------- CF per feature (analytic) ----------
    out: Dict[str, Dict[str, float]] = {
        key: dict(p_selected=0.0, mean_delta=0.0, mean_time=0.0)
        for (key, *_ ) in feat_meta
    }
    feas_weights: Dict[str, float] = {}
    any_feasible = False

    # Base time (no DDM): retrieval + reads + simple ops
    intuitive_ops = len(feat_meta)
    base_time_const = retrieval_cost + read_cost + T_INTUITIVE_OP * max(0, intuitive_ops)

    EPS = 1e-8

    for key, val_norm, is_numeric, _r_coef in feat_meta:
        coef = sampled_coefs.get(key, 0.0)

        feasible = True
        delta_orig = 0.0
        extra_time = 0.0

        if not is_numeric:
            # Default: try to turn ON if currently OFF
            if not flip_direction:
                if val_norm < 0.5:
                    delta_orig = 1.0 - val_norm
                    if abs(delta_orig) > 0.0:
                        extra_time += T_INTUITIVE_OP
                else:
                    delta_orig = 0.0
            else:
                # Opposite: try to turn OFF if currently ON
                if val_norm >= 0.5:
                    delta_orig = 0.0 - val_norm  # results in -1.0 if val_norm==1.0
                    if abs(delta_orig) > 0.0:
                        extra_time += T_INTUITIVE_OP
                else:
                    delta_orig = 0.0

        else:
            if verbose:
                print(f"Feature: {key}  val_norm={val_norm:.4f}  coef={coef:.6f}  flip_dir={flip_direction}")
            if abs(coef) < 1e-12:
                feasible = False
            else:
                # Default delta moves z_user to 0; if flip_direction, move in the opposite direction.
                delta_norm = (- z_user / coef)
                if flip_direction:
                    delta_norm = -delta_norm

                x_target_norm = val_norm + delta_norm

                if np.isfinite(x_target_norm) and (0.0 <= x_target_norm <= 1.0):
                    lo, hi = bounds.get(key, (-np.inf, np.inf))
                    step = _slider_step((lo, hi))
                    x_orig = _denorm(val_norm, lo, hi)
                    x_target_orig = _denorm(x_target_norm, lo, hi)
                    x_landed_orig = _snap_to_step(x_target_orig, (lo, hi), step)
                    if lo <= x_landed_orig <= hi:
                        delta_orig = float(x_landed_orig - x_orig)
                        if abs(delta_orig) > 0.0:
                            extra_time += T_INTUITIVE_OP
                    else:
                        feasible = False

                elif np.isfinite(x_target_norm) and (-feas_leeway_norm <= x_target_norm <= 1.0 + feas_leeway_norm):
                    # within leeway: push to boundary in the intended direction
                    desired_up = (delta_norm > 0.0)
                    lo, hi = bounds.get(key, (-np.inf, np.inf))
                    step = _slider_step((lo, hi))
                    x_orig = _denorm(val_norm, lo, hi)
                    x_target_orig = _denorm(1.0 if desired_up else 0.0, lo, hi)
                    x_landed_orig = _snap_to_step(x_target_orig, (lo, hi), step)
                    if lo <= x_landed_orig <= hi:
                        delta_orig = float(x_landed_orig - x_orig)
                        feasible = True
                        if abs(delta_orig) > 0.0:
                            extra_time += T_INTUITIVE_OP
                    else:
                        feasible = False
                else:
                    feasible = False

        # record per-feature stats
        out[key]["mean_delta"] = float(delta_orig)
        out[key]["mean_time"]  = float(base_time_const + extra_time)

        # Wald / SNR weight
        mu, var = coef_stats.get(key, (0.0, 1.0))
        w = abs(mu) / math.sqrt(var + EPS)

        if feasible and abs(delta_orig) > 0.0:
            feas_weights[key] = float(max(0.0, w))
            any_feasible = True
        else:
            feas_weights[key] = 0.0

    # ---------- Fallback when nothing feasible ----------
    if not any_feasible:
        for key, val_norm, is_numeric, _r_coef in feat_meta:
            coef = sampled_coefs.get(key, 0.0)
            mu, var = coef_stats.get(key, (0.0, 1.0))
            w = abs(mu) / math.sqrt(var + EPS)

            delta_orig = 0.0
            extra_time = 0.0

            if not is_numeric:
                # Opposite/Default policy mirrored here too
                if not flip_direction:
                    if val_norm < 0.5:
                        delta_orig = 1.0 - val_norm
                        if abs(delta_orig) > 0.0:
                            extra_time += T_INTUITIVE_OP
                else:
                    if val_norm >= 0.5:
                        delta_orig = 0.0 - val_norm
                        if abs(delta_orig) > 0.0:
                            extra_time += T_INTUITIVE_OP
            else:
                if abs(coef) >= 1e-12:
                    # desired direction derived from (possibly flipped) delta w.r.t. z_user
                    base_delta = (- z_user / coef)
                    if flip_direction:
                        base_delta = -base_delta
                    desired_up = (base_delta > 0.0)
                    lo, hi = bounds.get(key, (-np.inf, np.inf))
                    step = _slider_step((lo, hi))
                    x_orig = _denorm(val_norm, lo, hi)
                    x_target_orig = _denorm(1.0 if desired_up else 0.0, lo, hi)
                    x_landed_orig = _snap_to_step(x_target_orig, (lo, hi), step)
                    if lo <= x_landed_orig <= hi:
                        delta_orig = float(x_landed_orig - x_orig)
                        if abs(delta_orig) > 0.0:
                            extra_time += T_INTUITIVE_OP

            out[key]["mean_delta"] = float(delta_orig)
            out[key]["mean_time"]  = float(base_time_const + extra_time)
            feas_weights[key] = float(w if abs(delta_orig) > 0.0 else 0.0)

    # ---------- Normalize weights to probabilities ----------
    total_w = sum(feas_weights.values())
    if total_w > 0.0:
        for key in out.keys():
            out[key]["p_selected"] = feas_weights[key] / total_w
    else:
        u = 1.0 / max(1, len(out))
        for key in out.keys():
            out[key]["p_selected"] = u

    if verbose:
        picked = [k for k, w in feas_weights.items() if w > 0]
        print(f"[CF_HEUR(norm)] z_user={z_user:.6f}, "
              f"{'fallback(max-delta)' if not any_feasible else 'feasible'}, "
              f"flip_dir={flip_direction}, selected={picked or 'none'}, "
              f"leeway={feas_leeway_norm}, time_base={base_time_const:.3f}s")

    # === Memory combo chunks (unchanged schema) ===
    if memory is not None and len(out) > 0:
        mem_core = getattr(memory, "dm", memory)
        mem_time = getattr(mem_core, "time", 0.0)

        feats_inc, feats_dec = {}, {}
        for feat_key, v in out.items():
            p = float(max(0.0, min(1.0, v.get("p_selected", 0.0))))
            if p <= 0.0:
                continue
            delta = float(v.get("mean_delta", 0.0))
            tcost = float(v.get("mean_time",  0.0))
            is_categorical = ("=" in feat_key)

            if is_categorical:
                feats_inc[feat_key] = dict(p=p, delta=delta, time=tcost)
                feats_dec[feat_key] = dict(p=p, delta=delta, time=tcost)
            else:
                if delta > 0:
                    feats_inc[feat_key] = dict(p=p, delta=delta, time=tcost)
                elif delta < 0:
                    feats_dec[feat_key] = dict(p=p, delta=delta, time=tcost)

        def _update_combo(direction: str, feats: Dict[str, Dict[str, float]]):
            if not feats:
                return
            raw_mass = float(sum(f["p"] for f in feats.values()))
            refresh_p = max(0.0, min(1.0, raw_mass))
            chunk_name = f"lr_change_combo:{direction}"
            ch = memory.get_chunk(chunk_name) if hasattr(memory, "get_chunk") else None

            if ch is None:
                total_p = float(sum(f["p"] for f in feats.values())) or 1.0
                p_norm = {k: f["p"] / total_p for k, f in feats.items()}
                slots = {
                    "type": "lr_change_combo",
                    "direction": direction,
                    "features": list(feats.keys()),
                    "p_select": p_norm,
                    "delta":   {k: float(f["delta"]) for k, f in feats.items()},
                    "time":    {k: float(f["time"])  for k, f in feats.items()},
                    "mass":    float(total_p),
                    "n_updates": 1,
                    "expected_time": float(sum(p_norm[k] * feats[k]["time"] for k in feats)),
                }
                ch = memory.add_chunk(chunk_name, slots, update_retrieval=False)
            else:
                s = ch.slots
                w_old = float(s.get("mass", 0.0))
                w_new = w_old + raw_mass if (w_old + raw_mass) > 0 else raw_mass + 1e-12
                total_p = float(sum(f["p"] for f in feats.values())) or 1.0
                p_norm_new = {k: f["p"] / total_p for k, f in feats.items()}

                old_feats = set(s.get("features", []))
                new_feats = set(feats.keys())
                all_feats = sorted(old_feats | new_feats)

                p_old = dict(s.get("p_select", {}))
                d_old = dict(s.get("delta", {}))
                t_old = dict(s.get("time", {}))

                p_upd, d_upd, t_upd = {}, {}, {}
                for k in all_feats:
                    p_k_old = float(p_old.get(k, 0.0))
                    d_k_old = float(d_old.get(k, 0.0))
                    t_k_old = float(t_old.get(k, 0.0))

                    p_k_new = float(p_norm_new.get(k, 0.0))
                    d_k_new = float(feats.get(k, {}).get("delta", d_k_old))
                    t_k_new = float(feats.get(k, {}).get("time",  t_k_old))

                    p_upd[k] = (p_k_old * w_old + p_k_new * raw_mass) / w_new
                    d_upd[k] = (d_k_old * w_old + d_k_new * raw_mass) / w_new
                    t_upd[k] = (t_k_old * w_old + t_k_new * raw_mass) / w_new

                norm = sum(p_upd.values()) or 1.0
                for k in p_upd:
                    p_upd[k] /= norm

                s["direction"] = direction
                s["features"] = list(all_feats)
                s["p_select"] = {k: float(p_upd[k]) for k in all_feats}
                s["delta"]    = {k: float(t) for k, t in d_upd.items()}
                s["time"]     = {k: float(t) for k, t in t_upd.items()}
                s["mass"]     = float(w_new)
                s["n_updates"] = int(s.get("n_updates", 0)) + 1
                s["expected_time"] = float(sum(p_upd[k] * t_upd[k] for k in all_feats))

            if ch is not None and hasattr(ch, "add_prob_refresh"):
                ch.add_prob_refresh(mem_time, refresh_p)

        _update_combo("increase", feats_inc)
        _update_combo("decrease", feats_dec)

    expected_time = sum(v.get("p_selected", 0.0) * v.get("mean_time", 0.0) for v in out.values())
    out["expected_time"] = float(expected_time)
    return out

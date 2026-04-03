# ===== lr_memory_module.py =====
import math
import numpy as np
from typing import Optional, Sequence, Tuple
from .memory import remember_number_to_sf, build_number_profile, digits_to_value
import random

# ---------------------------------------------------------------------
# Utilities for storing/reconstructing numbers as meta + digits
# ---------------------------------------------------------------------
def round_to_sf(value: float, sf: int = 2) -> float:
    """Round to 'sf' significant figures (deterministic, no noise)."""
    if value == 0:
        return 0.0
    sign = -1 if value < 0 else 1
    v = abs(value)
    order = int(np.floor(np.log10(v)))
    factor = 10 ** (sf - order - 1)
    return sign * (round(v * factor) / factor)

# ---------------------------------------------------------------------
# Ratio-normalized DDM (so 900-600 and 90-60 behave the same)
# ---------------------------------------------------------------------

def ddm_prob_rt_ratio(
    terms: Sequence[float],
    *,
    a: float = 1.5,     # boundary separation (tune ~1..2)
    s: float = 1.0,     # diffusion SD
    Tnd: float = 0.30,  # non-decision time
    norm: str = "l2",   # "l2" | "l1" | "max"
    eps: float = 1e-9
) -> Tuple[float, float, float, float]:
    """
    Ratio-normalized DDM.
    Returns: (p_upper, E_RT, v_ratio, denom)

    v_ratio = sum(terms) / denom, where denom = scale(terms) chosen by `norm`.
    """
    num = float(sum(terms))
    abs_terms = [abs(t) for t in terms]
    if norm == "l2":
        denom = math.sqrt(sum(t*t for t in terms))
    elif norm == "l1":
        denom = sum(abs_terms)
    elif norm == "max":
        denom = max(abs_terms) if abs_terms else 0.0
    else:
        raise ValueError("norm must be 'l2', 'l1', or 'max'")
    denom = max(denom, eps)
    v_ratio = num / denom

    if s <= 0:
        p_up = 1.0 if v_ratio > 0 else 0.0 if v_ratio < 0 else 0.5
        E_T  = Tnd
        return p_up, E_T, v_ratio, denom

    k = (2.0 * a * v_ratio) / (s**2)
    p_up = 1.0 / (1.0 + math.exp(-k))

    avs = a * abs(v_ratio) / (s**2 + 1e-12)
    if abs(v_ratio) < 1e-9:
        E_T = (a*a)/(s*s)
    else:
        E_T = (a / (abs(v_ratio) + 1e-12)) * math.tanh(avs)

    return p_up, E_T, v_ratio, denom

# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _base_index_from_key(key: str) -> int:
    base = key.split('=')[0]  # "aN" from "aN" or "aN=K"
    return int(base[1:])

def add_lr_calculation_to_memory(
    lr_exp,
    memory,
    intercept_sf: int = 2,
    factor_sf: int = 2
):
    """
    Store intercept and each coefficient of an LR explanation into memory
    as digit-wise chunks (sign, scale, digits).
    
    Keys:
      - Intercept:   "lr:intercept"
      - Coefs:       "lr:coef:{feat_key}"
    """
    # Intercept
    remember_number_to_sf(
        memory,
        key="lr:intercept",
        value=lr_exp.intercept,
        max_sf=intercept_sf
    )

    # Coefficients
    for feat_key, coef_val in lr_exp.coefficients.items():
        remember_number_to_sf(
            memory,
            key=f"lr:coef:{feat_key}",
            value=coef_val,
            max_sf=factor_sf
        )



# ---------------------------------------------------------------------
# MAIN: lr_calculation (single entry point, LR-only)
# ---------------------------------------------------------------------
def lr_calculation(
    feature_vector,
    memory,
    lr_exp,
    *,
    mode: str = "retrieve",     # "retrieve" or "read"
    compute_sf: int = 2,
    T_enc: float = 2.0,
    T_op: float  = 3.0,
    # DDM params (ratio-based)
    ddm_a: float = 1.5,
    ddm_s: float = 1.0,
    ddm_Tnd: float = 0.30,
    ddm_norm: str = "l2",
    active_indices: Optional[list] = None,
    # Monte Carlo controls (used only in retrieve mode)
    n_mc: int = 64,
    topk_k: int = 3,
    refresh_prob: float = 1.0,
    verbose: bool = False,

    factor_display_sf: int = 2,
):
    """
    Returns: (probs(np.array([p0,p1])), total_time, info(dict))

    Assumptions:
      - build_number_profile(memory, key, sf_req, k, refresh_prob, verbose)
        returns {"meta": [( (sign,scale) or None, p ), ...],
                 "digits": [ [(digit or None, p), ...] for pos=1..sf_req ]}
      - CombinedMemory.topk_retrievals_with_prob_refresh(...) exists (used to estimate expected RT per step if needed)
      - Helpers: round_to_sf, _base_index_from_key, ddm_prob_rt_ratio exist
    """
    rng_choices = random.choices  # local alias
    x = np.asarray(feature_vector, dtype=float)
    idx_set = set(active_indices) if active_indices else None
    use_subset = idx_set is not None

    def _tick(mem, dt, acc_box):
        mem.tick(dt)
        acc_box[0] += float(dt)

    def _sample_number_from_profile(profile) -> float:
        """Sequential, conditional sampling of one number draw."""
        # meta
        m_vals, m_probs = zip(*profile["meta"])
        meta_choice = rng_choices(m_vals, weights=m_probs, k=1)[0]
        if meta_choice is None:
            return 0.0
        sign, p10 = meta_choice

        # digits chain
        digits = []
        for opts in profile["digits"]:
            d_vals, d_probs = zip(*opts)
            d_choice = rng_choices(d_vals, weights=d_probs, k=1)[0]
            if d_choice is None:
                break
            digits.append(int(d_choice))
        if len(digits) == 0:
            return 0.0
        return float(digits_to_value(sign, p10, digits, len(digits)))

    def _get_x_used(key: str) -> float:
        """Value channel: indicator for categorical (=), rounded value otherwise."""
        if '=' in key:
            base, cat_idx = key.split('=')
            col_idx = int(base[1:])
            return 1.0 if int(x[col_idx]) == int(cat_idx) else 0.0
        else:
            col_idx = int(key[1:])
            return float(round_to_sf(x[col_idx], compute_sf))

    total_time_box = [0.0]
    # We will aggregate Monte Carlo samples of (probs, time)
    mc_probs_p1 = []   # store p_up per sample
    mc_times    = []   # store total time per sample

    # Precompute the feature/value encodings (these don’t vary across MC)
    # And keys list respecting active_indices
    coef_items = []
    if use_subset:
        for key, coef_true in lr_exp.coefficients.items():
            base_idx = _base_index_from_key(key)
            if base_idx in idx_set:
                coef_items.append((key, coef_true))
    else:
        coef_items = list(lr_exp.coefficients.items())

    # READ mode = deterministic single pass (no MC)
    if mode == "read":
        if verbose: print("[lr_calculation] mode=read (deterministic)")
        terms = []
        ops_count = 0

        # Intercept (factor read)
        _tick(memory, T_enc, total_time_box)
        c0 = float(round_to_sf(float(lr_exp.intercept), 1))
        if verbose: print(f"  read intercept={lr_exp.intercept} -> {c0}")

        # Intercept value (1.0)
        # _tick(memory, T_enc, total_time_box)
        x_used = 1.0
        if c0 != 0.0 and x_used != 0.0:
            ops_count += 1
            # _tick(memory, T_op, total_time_box)
        terms.append(c0 * x_used)

        # Coefficients
        for key, coef_true in coef_items:
            # factor
            _tick(memory, T_enc, total_time_box)
            c = float(round_to_sf(float(coef_true), factor_display_sf))
            if verbose: print(f"  read coef[{key}]={coef_true} -> {c}")

            # value
            _tick(memory, T_enc, total_time_box)
            x_used = _get_x_used(key)

            # multiply (if both nonzero)
            if c != 0.0 and x_used != 0.0:
                ops_count += 1
                _tick(memory, T_op, total_time_box)

            terms.append(c * x_used)

        # DDM (decision)
        p_up, rt_dec, v_ratio, denom = ddm_prob_rt_ratio(
            terms, a=ddm_a, s=ddm_s, Tnd=ddm_Tnd, norm=ddm_norm
        )
        _tick(memory, rt_dec, total_time_box)
        probs = np.array([1.0 - p_up, p_up], dtype=float)

        info = {
            "mode": "read",
            "terms": terms,
            "sum": float(sum(terms)),
            "ops_count": ops_count,
            "ddm": {"a": ddm_a, "s": ddm_s, "Tnd": ddm_Tnd, "norm": ddm_norm,
                    "p_up": p_up, "rt_dec": rt_dec, "v_ratio": v_ratio, "denom": denom},
            "T_enc": T_enc, "T_op": T_op,
        }
        return probs, total_time_box[0], info

    # RETRIEVE mode = MC over noisy retrieval
    if verbose: print(f"[lr_calculation] mode=retrieve, n_mc={n_mc}, topk_k={topk_k}")

    # RETRIEVE mode
    inter_profile = build_number_profile(
        memory, "lr:intercept", compute_sf,
        k=topk_k, refresh_prob=refresh_prob, verbose=verbose
    )
    coef_profiles = {
        key: build_number_profile(
            memory, f"lr:coef:{key}", factor_display_sf,
            k=topk_k, refresh_prob=refresh_prob, verbose=verbose
        )
        for key, _ in coef_items
    }

    # Monte Carlo sampling
    for _ in range(int(max(1, n_mc))):
        this_time = 0.0
        terms = []

        # Intercept
        c0 = _sample_number_from_profile(inter_profile)
        this_time += inter_profile["expected_rt"]  # retrieval latency
        # this_time += T_enc                        # value read
        # if c0 != 0.0:
        #     this_time += T_op
        terms.append(c0)

        # Coefficients
        for key, _coef_true in coef_items:
            prof = coef_profiles[key]
            c = _sample_number_from_profile(prof)
            this_time += prof["expected_rt"]      # retrieval latency
            this_time += T_enc                    # value read
            x_used = _get_x_used(key)
            if c != 0.0 and x_used != 0.0:
                this_time += T_op
            terms.append(c * x_used)

        # DDM
        p_up, rt_dec, v_ratio, denom = ddm_prob_rt_ratio(
            terms, a=ddm_a, s=ddm_s, Tnd=ddm_Tnd, norm=ddm_norm
        )
        this_time += rt_dec

        mc_probs_p1.append(float(p_up))
        mc_times.append(float(this_time))

    # Aggregate MC
    p1 = float(np.mean(mc_probs_p1)) if mc_probs_p1 else 0.5
    avg_time = float(np.mean(mc_times)) if mc_times else 0.0
    probs = np.array([1.0 - p1, p1], dtype=float)

    info = {
        "mode": "retrieve",
        "n_mc": int(n_mc),
        "topk_k": int(topk_k),
        "compute_sf": int(compute_sf),
        "avg_p_up": p1,
        "avg_time": avg_time,
        "ddm": {"a": ddm_a, "s": ddm_s, "Tnd": ddm_Tnd, "norm": ddm_norm},
    }
    # if explain:
    #     # For retrieve+MC, terms vary per draw; we report only configuration-level info.
    #     info["explain_rows"] = []  # left empty by design for MC retrieve

    # Advance model time by the average (like expectation pass)
    memory.tick(avg_time)
    return probs, avg_time, info

# ---------------------------------------------------------------------
# Deterministic feedback refresher (simplified: no meta, optional verbose)
# ---------------------------------------------------------------------
def refresh_lr_calculation_in_memory(
    memory,
    lr_exp,
    *,
    intercept_display_sf: int = 2,
    factor_display_sf: int = 2,
    tick_per_refresh: float = 2,
    verbose: bool = False,
):
    """
    Deterministic rehearsal after feedback:
      - Refresh META + first N digits for intercept and each coefficient.
      - Charges time per successful refresh (tick_per_refresh).
    """
    # ---- Intercept: META then digits ----
    if memory.refresh("num:lr:intercept:meta"):
        memory.tick(tick_per_refresh)
        if verbose:
            print("[refresh] refreshed intercept META")
    for pos in range(1, intercept_display_sf + 1):
        if memory.refresh(f"num:lr:intercept:d{pos}"):
            memory.tick(tick_per_refresh)
            if verbose:
                print(f"[refresh] refreshed intercept digit {pos}")

    # ---- Coefficients: META then digits ----
    for feat_key in lr_exp.coefficients:
        if memory.refresh(f"num:lr:coef:{feat_key}:meta"):
            memory.tick(tick_per_refresh)
            if verbose:
                print(f"[refresh] refreshed coef[{feat_key}] META")
        for pos in range(1, factor_display_sf + 1):
            if memory.refresh(f"num:lr:coef:{feat_key}:d{pos}"):
                memory.tick(tick_per_refresh)
                if verbose:
                    print(f"[refresh] refreshed coef[{feat_key}] digit {pos}")


    memory.tick(20)

import numpy as np

# Counterfactual method 1
# w/ XAI: choose probabilistically based on range*factor (i.e., max-min * factor)
# Divide the final contribution(which is shown to 2sf) by the factor, if figure out that cannot change the first feature by required amount,
# then depending on the effort move onto the next feature.
# Amount moved is also similar to the decision tree method.
from typing import Dict, Tuple, List, Any, Optional
import numpy as np, math

# ---- Minimal helpers (drop these if you already have them) ----
# def round_to_sf(x: float, sf: int = 2) -> float:
#     if x == 0: return 0.0
#     k = int(math.floor(math.log10(abs(x))))
#     f = 10 ** (sf - 1 - k)
#     return round(x * f) / f

# def _base_index_from_key(key: str) -> int:
#     # "x3" -> 3, "x3=1" -> 3
#     base = key.split('=')[0]
#     return int(base[1:])

def _sf_halfwidth(v: float, sf: int = 2) -> float:
    """Half width of the rounding bin for v at 'sf' significant figures."""
    a = abs(v)
    if a == 0:
        # choose a tiny neighborhood when v == 0 (keeps code robust)
        return 0.5 * 10 ** (-sf)
    k = math.floor(math.log10(a))
    unit = 10 ** (k - (sf - 1))  # e.g., sf=2 => 10^(k-1)
    return 0.5 * unit

def _slider_step(bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    r = max(hi - lo, 1e-12)
    return 10 ** (math.floor(math.log10(r)) - 1)

def _snap_to_step(x: float, bounds: Tuple[float, float], step: float) -> float:
    lo, hi = bounds
    x = min(max(x, lo), hi)
    # Snap to nearest step from 'lo' (consistent slider behavior)
    return lo + round((x - lo) / step) * step

def _mental_division_time(a: float, b: float, compute_sf: int = 2, ability: float = 1.0) -> float:
    """Very simple time proxy if you don't want to hook your ACT-R timing:
    scales with difficulty and requested significant figures."""
    a, b = float(a), float(b)
    mag = 1.0 + math.log10(1.0 + abs(a)) + math.log10(1.0 + abs(b))
    return (compute_sf / max(ability, 1e-6)) * 0.25 * mag  # seconds-ish

# ----------------------------------------------------------------
def cf_lr_calc(
    feature_vector: np.ndarray,
    lr_exp: Any,
    bounds: Dict[str, Tuple[float, float]],
    *,
    n_runs: int = 200,
    value_display_sf: int = 2,
    factor_display_sf: int = 2,
    compute_sf: int = 2,
    W0_ANS: float = 0.2,          # Weber-like noise for division
    T_READ_NUM: float = 2.0,      # base "read" time unit
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Monte-Carlo simulate counterfactual edits:
      - Participant picks ONE feature with prob ∝ |coef|.
      - Rejects feature if required move exceeds its range.
      - If no feature feasible → fallback: choose uniformly from all features.
      - Applies noisy delta (division noise, slider step, 2-s.f. ambiguity).
    Always returns probabilities that sum to 1 across features.
    """
    rng = rng or np.random.default_rng()

    # --- Build rounded LR terms as in read path ---
    x = np.asarray(feature_vector, dtype=float)
    terms, c_rounded = {}, {}
    z = float(round_to_sf(lr_exp.intercept, value_display_sf))
    total_time_const = T_READ_NUM

    for key, c_true in lr_exp.coefficients.items():
        c = round_to_sf(c_true, factor_display_sf)
        c_rounded[key] = c
        if '=' in key:
            base, cat_idx = key.split('=')
            col = int(base[1:])
            x_used = 1.0 if int(x[col]) == int(cat_idx) else 0.0
            total_time_const += T_READ_NUM
        else:
            col = int(key[1:])
            x_used = round_to_sf(x[col], value_display_sf)
            total_time_const += T_READ_NUM
        t = float(c) * float(x_used)
        terms[key] = t
        z += t

    feature_keys = list(lr_exp.coefficients.keys())
    out = {k: dict(p_selected=0.0, mean_delta=0.0, mean_time=0.0) for k in feature_keys}

    raw_w = np.array([abs(c_rounded[k]) for k in feature_keys], dtype=float)
    if np.allclose(raw_w, 0.0):
        raw_w[:] = 1.0
    base_probs = raw_w / raw_w.sum()

    if verbose:
        print(f"[CF] Initial z={z:.3f}, intercept+terms")
        for k in feature_keys:
            print(f"    {k}: coef={c_rounded[k]:.3f}, term={terms[k]:.3f}")

    # --- Monte Carlo ---
    for run in range(n_runs):
        z_now = float(z)
        pool = feature_keys.copy()
        probs = base_probs.copy()

        used_key, applied_delta, total_time = None, 0.0, 0.0

        while pool:
            mask = np.array([k in pool for k in feature_keys], dtype=bool)
            p = probs[mask]; p /= p.sum()
            idx_all = np.arange(len(feature_keys))[mask]
            key = feature_keys[rng.choice(idx_all, p=p)]
            c = c_rounded[key]

            time_here = T_READ_NUM

            # --- Categorical features always feasible ---
            if '=' in key:
                base, cat_idx = key.split('=')
                col = int(base[1:])
                current_is_cat = int(x[col]) == int(cat_idx)
                applied = (0.0 if current_is_cat else +1.0)
                applied_delta = applied
                used_key = key
                total_time += time_here
                break

            # --- Numeric features: check feasibility ---
            col = int(key[1:])
            lo, hi = bounds.get(key, (-np.inf, np.inf))
            step = _slider_step((lo, hi))
            if abs(c) < 1e-12:
                pool.remove(key); continue
            delta_true = -z_now / c
            max_move = hi - x[col] if delta_true >= 0 else lo - x[col]
            if not np.isfinite(max_move) or abs(max_move) < abs(delta_true):
                pool.remove(key); continue  # infeasible

            # Apply noisy division
            noisy_delta = delta_true + rng.normal(0.0, W0_ANS * abs(delta_true))
            time_here += _mental_division_time(z_now, c, compute_sf=compute_sf)
            x_target = x[col] + noisy_delta
            x_target = _snap_to_step(x_target, (lo, hi), step)

            disp = round_to_sf(x_target, 2)
            half = _sf_halfwidth(disp, 2)
            x_landed = rng.uniform(disp - half, disp + half)
            x_landed = _snap_to_step(x_landed, (lo, hi), step)

            applied_delta = x_landed - x[col]
            used_key = key
            total_time += time_here
            break

        # --- If no feature selected → fallback uniform pick ---
        if used_key is None:
            used_key = rng.choice(feature_keys)
            applied_delta = 0.0  # no effective move
            total_time += T_READ_NUM

        # Record stats
        s = out[used_key]
        s["p_selected"] += 1.0
        s["mean_delta"] += applied_delta
        s["mean_time"] += total_time_const + total_time

    # --- Normalize ---
    for k, s in out.items():
        used = s["p_selected"]
        if used > 0:
            s["mean_delta"] /= used
            s["mean_time"] /= used
        s["p_selected"] /= float(n_runs)

    if verbose:
        total_p = sum(s["p_selected"] for s in out.values())
        print(f"[CF] ∑ p_selected = {total_p:.3f}")

    return out



# ===== lr_memory_module.py =====
import math
import numpy as np
from typing import Optional, Sequence, Tuple, List, Dict, Any
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

def ddm_prob_rt(
    evidence: float,
    *,
    a: float = 1.5,     # boundary separation
    s: float = 1.0,     # diffusion SD
    Tnd: float = 0.30,  # non-decision time
    gain: float = 1.0   # global sensitivity (maps evidence to drift)
) -> tuple[float, float, float]:
    """
    Evidence-based DDM kernel.
    Inputs:
      evidence: trial-level evidence (e.g., digit difference for DT, 
                normalized sum for LR)
      a, s, Tnd: diffusion parameters (shared across tasks)
      gain: global sensitivity factor

    Returns:
      (p_upper, E_RT, drift)
        p_upper : probability of choosing 'upper' (evidence > 0)
        E_RT    : expected response time including Tnd
        drift   : effective drift velocity
    """
    v = gain * evidence

    if abs(v) < 1e-12:
        p_up = 0.5
        E_dec = (a*a) / (s*s)
    else:
        k = (2*a*v) / (s**2)
        p_up = 1.0 / (1.0 + math.exp(-k))
        E_dec = (a / v) * math.tanh((a*v) / (s**2))

    return p_up, Tnd + E_dec, v


def evidence_lr_divnorm(terms, mode="l2", eps=1e-6):
    """
    Compute LR evidence as normalized sum of terms.
    mode: 'l1', 'l2', or 'max' normalization.
    """
    num = float(sum(terms))
    if mode == "l1":
        denom = eps + sum(abs(t) for t in terms)
    elif mode == "l2":
        denom = eps + math.sqrt(sum(t*t for t in terms))
    elif mode == "max":
        denom = eps + max((abs(t) for t in terms), default=0.0)
    else:
        raise ValueError("mode must be 'l1'|'l2'|'max'")
    return num / denom



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
    factor_display_sf: int = 2
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
            max_sf=factor_display_sf
        )



# ---------------------------------------------------------------------
# MAIN: lr_calculation (single entry point, LR-only)
# ---------------------------------------------------------------------

# --- Reusable helpers ---------------------------------------------------------

def tick(memory, dt: float, total_time_box: List[float]) -> None:
    """Advance internal memory time and accumulate wall time."""
    memory.tick(dt)
    total_time_box[0] += float(dt)

def sample_number_from_profile(profile: Dict[str, Any]) -> float:
    """
    Draw a single number from a profile produced by build_number_profile.
    Expects:
      profile["meta"]   = [((sign, p10) or None, p), ...]
      profile["digits"] = [[(digit or None, p), ...], ...]
    """
    m_vals, m_probs = zip(*profile["meta"])
    meta_choice = random.choices(m_vals, weights=m_probs, k=1)[0]
    if meta_choice is None:
        return 0.0
    sign, p10 = meta_choice

    digits = []
    for opts in profile["digits"]:
        d_vals, d_probs = zip(*opts)
        d_choice = random.choices(d_vals, weights=d_probs, k=1)[0]
        if d_choice is None:
            break
        digits.append(int(d_choice))
    if not digits:
        return 0.0
    return float(digits_to_value(sign, p10, digits, len(digits)))

def get_x_used(
    key: str,
    x: np.ndarray,
    compute_sf: int
) -> float:
    """Value channel: 1.0 for categorical match; rounded numeric otherwise."""
    if "=" in key:
        base, cat_idx = key.split("=")
        col_idx = int(base[1:])
        return 1.0 if int(x[col_idx]) == int(cat_idx) else 0.0
    col_idx = int(key[1:])
    return float(round_to_sf(x[col_idx], compute_sf))

# --- Main function ------------------------------------------------------------

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
    active_indices: Optional[List[int]] = None,
    # Monte Carlo controls (retrieve mode)
    n_mc: int = 64,
    topk_k: int = 3,
    refresh_prob: float = 1.0,
    verbose: bool = False,
    factor_display_sf: int = 2,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Returns: (probs(np.array([p0,p1])), total_time, info(dict))

    Assumes the following exist in your codebase:
      - build_number_profile(...)
      - round_to_sf, digits_to_value, _base_index_from_key(...)
      - evidence_lr_divnorm(...), ddm_prob_rt(...)
    """
    x = np.asarray(feature_vector, dtype=float)
    idx_set = set(active_indices) if active_indices is not None else None

    # Filter coef list by active feature indices (if provided)
    if idx_set is not None:
        coef_items = [
            (key, coef_true)
            for key, coef_true in lr_exp.coefficients.items()
            if _base_index_from_key(key) in idx_set
        ]
    else:
        coef_items = list(lr_exp.coefficients.items())

    total_time_box = [0.0]

    # ------------------------ READ (deterministic) ----------------------------
    if mode == "read":
        if verbose: print("[lr_calculation] mode=read")
        terms: List[float] = []
        ops_count = 0

        # Intercept factor (read once)
        tick(memory, T_enc, total_time_box)
        c0 = float(round_to_sf(float(lr_exp.intercept), 1))
        terms.append(c0)

        # Coefficients: read factor & value; pay T_op only if both nonzero
        for key, coef_true in coef_items:
            tick(memory, T_enc, total_time_box)  # read coef
            c = float(round_to_sf(float(coef_true), factor_display_sf))

            tick(memory, T_enc, total_time_box)  # read value
            x_used = get_x_used(key, x, compute_sf)

            if c != 0.0 and x_used != 0.0:
                ops_count += 1
                tick(memory, T_op, total_time_box)
            terms.append(c * x_used)

        # Decision (DDM)
        evidence = evidence_lr_divnorm(terms, mode=ddm_norm)
        p_up, rt_dec, v_val = ddm_prob_rt(
            evidence, a=ddm_a, s=ddm_s, Tnd=ddm_Tnd, gain=1.0
        )
        tick(memory, rt_dec, total_time_box)

        probs = np.array([1.0 - p_up, p_up], dtype=float)
        info = {
            "mode": "read",
            "terms": terms,
            "sum": float(sum(terms)),
            "ops_count": ops_count,
            "evidence": evidence,
            "ddm": {"a": ddm_a, "s": ddm_s, "Tnd": ddm_Tnd, "gain": 1.0,
                    "p_up": p_up, "rt_dec": rt_dec, "v": v_val},
            "T_enc": T_enc, "T_op": T_op,
        }
        return probs, total_time_box[0], info

    # ----------------------- RETRIEVE (Monte Carlo) --------------------------
    if verbose:
        print(f"[lr_calculation] mode=retrieve, n_mc={n_mc}, topk_k={topk_k}")

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

    mc_probs_p1: List[float] = []
    mc_times: List[float] = []

    for _ in range(int(max(1, n_mc))):
        this_time = 0.0
        terms: List[float] = []

        # Intercept retrieval
        c0 = sample_number_from_profile(inter_profile)
        this_time += inter_profile["expected_rt"]
        terms.append(c0)

        # Coefficients retrieval
        for key, _coef_true in coef_items:
            prof = coef_profiles[key]
            c = sample_number_from_profile(prof)
            this_time += prof["expected_rt"]
            this_time += T_enc  # read value
            x_used = get_x_used(key, x, compute_sf)
            if c != 0.0 and x_used != 0.0:
                this_time += T_op
            terms.append(c * x_used)

        # Decision (DDM)
        evidence = evidence_lr_divnorm(terms, mode=ddm_norm)
        p_up, rt_dec, _v = ddm_prob_rt(
            evidence, a=ddm_a, s=ddm_s, Tnd=ddm_Tnd, gain=1.0
        )
        this_time += rt_dec

        mc_probs_p1.append(float(p_up))
        mc_times.append(float(this_time))

    # --- after the MC loop in RETRIEVE branch ---
    p1 = float(np.mean(mc_probs_p1)) if mc_probs_p1 else 0.5
    avg_time = float(np.mean(mc_times)) if mc_times else 0.0
    probs = np.array([1.0 - p1, p1], dtype=float)

    # === RESTORED ORIGINAL info dict ===
    info = {
        "mode": "retrieve",
        "n_mc": int(n_mc),
        "topk_k": int(topk_k),
        "compute_sf": int(compute_sf),
        "avg_p_up": p1,
        "avg_time": avg_time,
        "ddm": {"a": ddm_a, "s": ddm_s, "Tnd": ddm_Tnd, "norm": ddm_norm},
        # "explain_rows": []  # (left empty in MC retrieve; add back if you use it later)
    }

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
    active_indices: Optional[list] = None,
):
    """
    Deterministic rehearsal after feedback:
      - Refresh META + first N digits for intercept and selected coefficients.
      - Charges time per successful refresh (tick_per_refresh).
    """
    # ---- Intercept: always included ----
    if memory.refresh("num:lr:intercept:meta"):
        memory.tick(tick_per_refresh)
        if verbose:
            print("[refresh] refreshed intercept META")
    for pos in range(1, intercept_display_sf + 1):
        if memory.refresh(f"num:lr:intercept:d{pos}"):
            memory.tick(tick_per_refresh)
            if verbose:
                print(f"[refresh] refreshed intercept digit {pos}")

    # ---- Coefficients: filter by active_indices if provided ----
    idx_set = set(active_indices) if (active_indices is not None) else None

    for feat_key in lr_exp.coefficients:
        base_idx = _base_index_from_key(feat_key)
        if idx_set is not None and base_idx not in idx_set:
            continue  # skip this coefficient

        if memory.refresh(f"num:lr:coef:{feat_key}:meta"):
            memory.tick(tick_per_refresh)
            if verbose:
                print(f"[refresh] refreshed coef[{feat_key}] META")
        for pos in range(1, factor_display_sf + 1):
            if memory.refresh(f"num:lr:coef:{feat_key}:d{pos}"):
                memory.tick(tick_per_refresh)
                if verbose:
                    print(f"[refresh] refreshed coef[{feat_key}] digit {pos}")

    # Optional baseline tick
    memory.tick(20)

# Counterfactual method 1
# w/ XAI: choose probabilistically based on range*factor (i.e., max-min * factor)
# Divide the final contribution(which is shown to 2sf) by the factor, if figure out that cannot change the first feature by required amount,
# then depending on the effort move onto the next feature.
# Amount moved is also similar to the decision tree method.
# ----------------------------------------------------------------
from typing import Dict, Tuple, List, Any, Optional
import numpy as np
import math

# Assumed available:
# - round_to_sf(x: float, sf: int) -> float
# - _base_index_from_key(key: str) -> int      # "x3"->3, "x3=1"->3

def _slider_step(bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    r = max(hi - lo, 1e-12)
    return 10 ** (math.floor(math.log10(r)) - 1)

def _snap_to_step(x: float, bounds: Tuple[float, float], step: float) -> float:
    lo, hi = bounds
    x = min(max(x, lo), hi)
    return lo + round((x - lo) / step) * step

def cf_lr_calculation(
    feature_vector: np.ndarray,
    lr_exp: Any,
    bounds: Dict[str, Tuple[float, float]],
    *,
    # kept for interface symmetry / timing
    compute_sf: int = 2,
    T_enc: float = 2.0,
    T_op: float  = 3.0,
    memory: Optional[Any] = None,
    store_prob: bool = True,
    factor_display_sf: int = 2,
    value_display_sf: int = 2,
    active_indices: Optional[List[int]] = None,
    verbose: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Analytic (no-MC) counterfactual distribution:
      - z_true = intercept + Σ coef_true * value_true  (no rounding)
      - z_disp = round_to_sf(z_true, 2)
      - For each feature i:
          • numeric: Δx_i = - z_disp / coef_i (feasible if within bounds)
          • categorical: feasible; Δ=1 if flip needed else 0
          • weight_i = range_i * |coef_i|  (categorical range=1)
      - p_selected(i) = weight_i / Σ weights over feasible; if none feasible -> uniform over all.

    Returns { key: {p_selected, mean_delta, mean_time} }
    """
    x = np.asarray(feature_vector, dtype=float)

    # Filter coefficients
    if active_indices is not None:
        idx_set = set(active_indices)
        coef_items = [(key, c) for key, c in lr_exp.coefficients.items()
                      if _base_index_from_key(key) in idx_set]
    else:
        coef_items = list(lr_exp.coefficients.items())

    feature_keys = [k for k, _ in coef_items]
    out = {k: dict(p_selected=0.0, mean_delta=0.0, mean_time=0.0) for k in feature_keys}

    # ---- Build z_true (no rounding) & base read/multiply time ----
    z_true = float(lr_exp.intercept)
    base_time_const = 0.0

    for key, c_true in coef_items:
        # base_time_const += T_enc  # read coef
        c = float(c_true)
        # base_time_const += T_enc  # read value

        if "=" in key:
            base, cat_idx = key.split("=")
            col = int(base[1:])
            x_used = 1.0 if int(x[col]) == int(cat_idx) else 0.0
        else:
            col = int(key[1:])
            x_used = float(x[col])

        # if c != 0.0 and x_used != 0.0:
        #     base_time_const += T_op  # multiply

        z_true += c * x_used
    
    base_time_const = T_enc * 2 # read displayed sum and value

    # Only the final sum is rounded (what the user "sees")
    z_disp = float(round_to_sf(z_true, 2))

    # Helpers for weights and feasibility
    def _range_of(key: str) -> float:
        if "=" in key:
            return 1.0
        lo, hi = bounds.get(key, (0.0, 0.0))
        return max(hi - lo, 0.0)

    # Precompute per-feature deterministic deltas, feasibility, weights, times
    feas_weights = {}
    for key, c_true in coef_items:
        c = float(c_true)
        extra_time = 0.0
        delta = 0.0
        feasible = True

        if "=" in key:
            base, cat_idx = key.split("=")
            col = int(base[1:])
            already = (int(x[col]) == int(cat_idx))
            delta = 0.0 if already else 1.0  # 0/1 flip indicator
            if not already:
                extra_time += T_op
        else:
            # numeric
            if abs(c) < 1e-12:
                feasible = False
            else:
                col = int(key[1:])
                lo, hi = bounds.get(key, (-np.inf, np.inf))
                step = _slider_step((lo, hi))
                x_target = x[col] + (- z_disp / c)

                range = hi - lo
                lo_relaxed = lo - range/2 if (lo - 0 < 1e8) else lo
                hi_relaxed = hi + range/2 if (hi - 1 < 1e8) else hi

                if not (np.isfinite(x_target) and lo_relaxed <= x_target <= hi_relaxed):
                    feasible = False
                else:
                    x_target = min(max(x_target, lo), hi)
                    x_landed = _snap_to_step(x_target, (lo, hi), step)
                    delta = float(x_landed - x[col])
                    if delta != 0.0:
                        extra_time += T_op

        # print("Increasing by ", delta, "on feature", key, "with coef", c, "to remove sum", z_disp)

        weight = abs(c) if feasible else 0.0

        out[key]["mean_delta"] = delta
        # print("Base time: ", base_time_const, "Extra time: ", extra_time)
        out[key]["mean_time"] = base_time_const + extra_time
        feas_weights[key] = weight

    # Normalize over feasible weights; if none -> uniform over all features
    total_w = sum(feas_weights.values())
    if total_w > 0.0:
        for k in feature_keys:
            out[k]["p_selected"] = feas_weights[k] / total_w
    else:
        # fallback: uniform distribution (mirrors the MC fallback behavior)
        u = 1.0 / max(1, len(feature_keys))
        for k in feature_keys:
            out[k]["p_selected"] = u

    if verbose:
        feasible_keys = [k for k, w in feas_weights.items() if w > 0]
        print(f"[CF] z_true={z_true:.6f}, z_disp(2sf)={z_disp:.6f}, "
              f"feasible={feasible_keys or 'none'}")

    # === PROBABILISTIC SAVE: create NEW chunk per feature; no updates/means ===
    # === SINGLE COMBO CHUNK PER DIRECTION (update + probabilistic refresh) ===
    if store_prob and memory is not None and len(out) > 0:
        mem_core = getattr(memory, "dm", memory)  # CombinedMemory or DM
        mem_time = getattr(mem_core, "time", 0.0)

        # Build per-direction slices from the current analytic distribution
        feats_inc, feats_dec = {}, {}
        for feat_key, v in out.items():
            p = float(max(0.0, min(1.0, v.get("p_selected", 0.0))))
            if p <= 0.0:
                continue
            delta = float(v.get("mean_delta", 0.0))
            tcost = float(v.get("mean_time", 0.0))

            # Categorical: no sign; include in BOTH directions
            is_categorical = ("=" in feat_key)
            if is_categorical:
                feats_inc[feat_key] = dict(p=p, delta=delta, time=tcost, feature_type="categorical")
                feats_dec[feat_key] = dict(p=p, delta=delta, time=tcost, feature_type="categorical")
            else:
                # Numeric: route by sign
                if delta > 0:
                    feats_inc[feat_key] = dict(p=p, delta=delta, time=tcost, feature_type="numeric")
                elif delta < 0:
                    feats_dec[feat_key] = dict(p=p, delta=delta, time=tcost, feature_type="numeric")
                # delta == 0 -> ignore (no directional preference)

        def _update_combo(direction: str, feats: Dict[str, Dict[str, float]]):
            if not feats:
                return
            # Raw mass = sum of raw p_selected for included features (cap at 1 for refresh prob)
            raw_mass = float(sum(f["p"] for f in feats.values()))
            refresh_p = max(0.0, min(1.0, raw_mass))

            chunk_name = f"lr_change_combo:{direction}"
            ch = memory.get_chunk(chunk_name) if hasattr(memory, "get_chunk") else None

            if ch is None:
                # Create new combo with initial stats (normalized per-direction p)
                total_p = float(sum(f["p"] for f in feats.values())) or 1.0
                p_norm = {k: f["p"] / total_p for k, f in feats.items()}
                slots = {
                    "type": "lr_change_combo",
                    "direction": direction,                   # "increase" | "decrease"
                    "features": list(feats.keys()),           # canonical order is fine
                    "p_select": p_norm,                       # dict[feat] -> prob (sums to 1 over included feats)
                    "delta":   {k: float(f["delta"]) for k, f in feats.items()},
                    "time":    {k: float(f["time"])  for k, f in feats.items()},
                    "mass":    float(total_p),                # running weight accumulator (raw mass)
                    "n_updates": 1,
                    # convenience cache for expected time of this combo
                    "expected_time": float(sum(p_norm[k] * feats[k]["time"] for k in feats)),
                }
                ch = memory.add_chunk(chunk_name, slots, update_retrieval=False)
            else:
                # Merge with weighted running means (by raw_mass): update probs, deltas, times
                s = ch.slots
                w_old = float(s.get("mass", 0.0))
                w_new = w_old + raw_mass if (w_old + raw_mass) > 0 else raw_mass + 1e-12

                # Normalize current feats within direction
                total_p = float(sum(f["p"] for f in feats.values())) or 1.0
                p_norm_new = {k: f["p"] / total_p for k, f in feats.items()}

                # Union of features: carry forward old ones (decay) and add/update new ones
                old_feats = set(s.get("features", []))
                new_feats = set(feats.keys())
                all_feats = sorted(old_feats | new_feats)

                # Make sure dicts exist
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

                    # Weighted running means by raw mass
                    p_upd[k] = (p_k_old * w_old + p_k_new * raw_mass) / w_new
                    d_upd[k] = (d_k_old * w_old + d_k_new * raw_mass) / w_new
                    t_upd[k] = (t_k_old * w_old + t_k_new * raw_mass) / w_new

                # Renormalize p_upd across the union to keep it a distribution (if you prefer)
                norm = sum(p_upd.values()) or 1.0
                for k in p_upd:
                    p_upd[k] /= norm

                s["direction"] = direction
                s["features"] = list(all_feats)
                s["p_select"] = {k: float(p_upd[k]) for k in all_feats}
                s["delta"]    = {k: float(d_upd[k]) for k in all_feats}
                s["time"]     = {k: float(t_upd[k]) for k in all_feats}
                s["mass"]     = float(w_new)
                s["n_updates"] = int(s.get("n_updates", 0)) + 1
                s["expected_time"] = float(sum(p_upd[k] * t_upd[k] for k in all_feats))

            # Probabilistic refresh for the combo
            if ch is not None and hasattr(ch, "add_prob_refresh"):
                ch.add_prob_refresh(mem_time, refresh_p)

        # Update both directions from this pass
        _update_combo("increase", feats_inc)
        _update_combo("decrease", feats_dec)
    # --- expected time under the selection distribution (no format change) ---
    expected_time = sum(v.get("p_selected", 0.0) * v.get("mean_time", 0.0) for v in out.values())
    out["expected_time"] = float(expected_time)

    return out

from .memory import CombinedMemory
from typing import Any, Dict, Optional

def recall_change_lr(
    memory: Any,
    *,
    k: int = 6,
    refresh_prob: float = 1.0,
    normalize: bool = True,   # kept for fallback path
    verbose: bool = False,
    preferred_direction: Optional[str] = None,  # "increase" | "decrease" | None
    T_enc: Optional[float] = None,              # NEW: encoding/read time
) -> Dict[str, Dict[str, float]]:
    """
    Returns a dict keyed by feature:
      {
        "<feature_key>": {"p_selected": float, "mean_delta": float, "mean_time": float},
        ...
        "expected_time": float
      }

    Timing policy:
      mean_time for every feature = retrieval_rt (for this recall) + T_enc.
      Any time stored in chunks is ignored.
    """
    if T_enc is None:
        T_enc = float(getattr(memory, "T_enc", 0.0))

    # Helper: renormalize a dict of probabilities (safely)
    def _renorm_prob_map(p_map: Dict[str, float]) -> Dict[str, float]:
        Z = float(sum(max(0.0, v) for v in p_map.values()))
        if Z <= 1e-12:
            # fall back to uniform over keys with nonzero entries
            keys = [k for k, v in p_map.items() if v > 0.0] or list(p_map.keys())
            if not keys:
                return {}
            uni = 1.0 / len(keys)
            return {k: (uni if k in keys else 0.0) for k in p_map.keys()}
        return {k: (max(0.0, v) / Z) for k, v in p_map.items()}

    # 0) Try combo retrieval first if direction specified
    if preferred_direction in ("increase", "decrease"):
        combo_request = {"type": "lr_change_combo", "direction": preferred_direction}
        if isinstance(memory, CombinedMemory):
            res_c = memory.topk_retrievals_with_prob_refresh(
                request=combo_request, k=1, refresh_prob=refresh_prob, add_refresh=True, verbose=verbose
            )
            retrieval_rt = float(res_c.get("retrieval_time", res_c.get("rt", 0.0)))
            top_c = res_c.get("top_k", [])
            if top_c:
                ch, p = top_c[0]
                s = getattr(ch, "slots", {}) or {}
                feats = list(s.get("features", []))
                p_map = _renorm_prob_map(dict(s.get("p_select", {})))
                d_map = dict(s.get("delta", {}))

                mean_time_const = float(retrieval_rt + T_enc)

                out: Dict[str, Dict[str, float]] = {}
                for feat in feats:
                    out[feat] = {
                        "p_selected": float(p_map.get(feat, 0.0)),
                        "mean_delta": float(d_map.get(feat, 0.0)),
                        "mean_time":  mean_time_const,  # <-- retrieval_rt + T_enc
                    }

                expected_time = sum(v["p_selected"] * v["mean_time"] for v in out.values())
                out["expected_time"] = float(expected_time)

                if verbose:
                    print(f"[CF-RECALL/LR combo] dir={preferred_direction}, "
                          f"retrieval_rt={retrieval_rt:.4f}, T_enc={T_enc:.4f}, "
                          f"expected_time={expected_time:.4f}")
                return out
        else:
            # bare DM fallback: exact-name lookup (no latency info available)
            ch = memory.get_chunk(f"lr_change_combo:{preferred_direction}") if hasattr(memory, "get_chunk") else None
            if ch:
                s = getattr(ch, "slots", {}) or {}
                feats = list(s.get("features", []))
                p_map = _renorm_prob_map(dict(s.get("p_select", {})))
                d_map = dict(s.get("delta", {}))
                retrieval_rt = 0.0
                mean_time_const = float(retrieval_rt + T_enc)

                out: Dict[str, Dict[str, float]] = {}
                for feat in feats:
                    out[feat] = {
                        "p_selected": float(p_map.get(feat, 0.0)),
                        "mean_delta": float(d_map.get(feat, 0.0)),
                        "mean_time":  mean_time_const,
                    }
                expected_time = sum(v["p_selected"] * v["mean_time"] for v in out.values())
                out["expected_time"] = float(expected_time)
                return out

    # 1) Fallback: per-feature retrieval/averaging
    request = {"type": "lr_change"}
    if preferred_direction in ("increase", "decrease"):
        request["direction"] = preferred_direction

    if isinstance(memory, CombinedMemory):
        res = memory.topk_retrievals_with_prob_refresh(
            request=request, k=k, refresh_prob=refresh_prob, add_refresh=True, verbose=verbose
        )
        top_k = res.get("top_k", [])
        p_none = float(res.get("p_none", 0.0))
        retrieval_rt = float(res.get("retrieval_time", res.get("rt", 0.0)))
    else:
        ch, act, rt = memory.retrieve(request, verbose=verbose)
        top_k = [(ch, 1.0)] if ch is not None else []
        p_none = 0.0 if ch is not None else 1.0
        retrieval_rt = float(rt)

    per_feat: Dict[str, Dict[str, float]] = {}
    per_feat_kind: Dict[str, str] = {}

    for ch, p in top_k:
        if ch is None or p <= 0.0:
            continue
        s = getattr(ch, "slots", {}) or {}
        feat = str(s.get("feature", "")) or ""
        if not feat:
            continue

        feature_type = s.get("feature_type", s.get("kind", "?"))
        delta = float(s.get("delta", 0.0))

        # Respect requested direction for numeric deltas
        if feature_type == "numeric" and preferred_direction in ("increase", "decrease"):
            current_sign = 1 if delta > 0 else (-1 if delta < 0 else 0)
            want_sign = 1 if preferred_direction == "increase" else -1
            if current_sign != 0 and current_sign != want_sign:
                delta = -delta

        acc = per_feat.setdefault(feat, {"p_sum": 0.0, "delta_wsum": 0.0})
        acc["p_sum"]      += float(p)
        acc["delta_wsum"] += float(p) * delta
        per_feat_kind.setdefault(feat, feature_type)

    if not per_feat:
        if verbose:
            print(f"[CF-RECALL/LR] no chunks retrieved (p_none={p_none:.3f}).")
        return {"expected_time": float(retrieval_rt + T_enc)}

    total_p = sum(v["p_sum"] for v in per_feat.values())
    norm_denom = total_p if (normalize and total_p > 0.0) else 1.0

    mean_time_const = float(retrieval_rt + T_enc)

    out: Dict[str, Dict[str, float]] = {}
    for feat, v in per_feat.items():
        p_feat = v["p_sum"]
        mean_delta = (v["delta_wsum"] / p_feat) if p_feat > 1e-12 else 0.0

        out[feat] = {
            "p_selected": (p_feat / norm_denom if norm_denom > 0 else 0.0),
            "mean_delta": float(mean_delta),
            "mean_time":  mean_time_const,  # <-- retrieval_rt + T_enc
        }

    expected_time = sum(v["p_selected"] * v["mean_time"] for v in out.values())
    out["expected_time"] = float(expected_time)

    if verbose:
        print(f"[CF-RECALL/LR fallback] retrieval_rt={retrieval_rt:.4f}, "
              f"T_enc={T_enc:.4f}, expected_time={expected_time:.4f}")

    return out




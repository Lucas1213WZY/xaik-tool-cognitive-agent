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

from .lr_memory import ddm_prob_rt_ratio
def lr_heuristic(
    feature_vector,
    memory,
    lr_exp,
    *,
    num_samples: int = 40,
    K_top: int = 3,                 # top-k to consider per retrieval
    T_READ_NUM: float = 2.0,        # per numeric/categorical read
    T_INTUITIVE_OP: float = 0.5,    # per internal +/*
    # DDM params (tune a, s, Tnd; norm as in your helper)
    ddm_a: float = 1.5,
    ddm_s: float = 1.0,
    ddm_Tnd: float = 0.30,
    ddm_norm: str = "l2",
    active_indices: list = None,
    verbose: bool = False,
):
    import numpy as np, math
    rng = np.random.default_rng()
    x = np.asarray(feature_vector, float)
    active_set = set(active_indices) if active_indices is not None else None

    read_cost = 0.0
    retrieval_cost = 0.0

    # --- Intercept retrieval (distribution) ---
    r_int = memory.topk_retrievals_with_prob_refresh(
        {"type":"intercept_prob"}, k=K_top, refresh_prob=1.0, add_refresh=True
    )
    retrieval_cost += float(r_int["expected_rt"])

    # --- Coef metadata + one-time read/equality costs like before ---
    feat_meta = []
    for key in lr_exp.coefficients.keys():
        base_idx = int(key.split('=')[0][1:])
        if active_set and base_idx not in active_set:
            continue

        # retrieve distribution for coef params
        r_coef = memory.topk_retrievals_with_prob_refresh(
            {"type":"coef_prob","feature_key":key}, k=K_top, refresh_prob=1.0, add_refresh=True
        )
        retrieval_cost += float(r_coef["expected_rt"])

        # read value once (deterministic read cost)
        if '=' in key:
            base, cat = key.split('=')
            col = int(base[1:])
            read_cost += T_READ_NUM
            val = 1.0 if int(x[col]) == int(cat) else 0.0
            is_numeric = False
        else:
            col = int(key[1:])
            read_cost += T_READ_NUM
            val = float(x[col])
            is_numeric = True

        feat_meta.append((key, val, is_numeric, r_coef))

    # if verbose:
    #     print(f"[LR_HEUR] intercept retrieved: {[(ch.name,p) for ch, p in r_int['top_k']]}, expected RT {r_int['expected_rt']:.2f}s")
    #     for (key, val, is_numeric, r_coef) in feat_meta:
    #         retrieved_str = [f"{ch.name}:{p:.2f}" for ch, p in r_coef['top_k']]
    #         print(f"  feature {key} (val={val}, {'num' if is_numeric else 'cat'}), "
    #             f"retrieved: {retrieved_str}, expected RT {r_coef['expected_rt']:.2f}s")


    # --- Monte Carlo over terms with mis-retrievals ---
    z_terms = []   # per-sample list of lists (terms)
    for _ in range(num_samples):
        terms = []

        # Intercept sample
        ch = _draw_from_topk(r_int, rng)
        if ch is None:
            mu, var = 0.0, 0.01   # small prior variance if nothing retrieved
        else:
            mu  = float(ch.slots.get("mu", 0.0))
            # lean chunks store only "var"; fall back to "sigma" if present
            var = float(ch.slots["var"])
        sigma = math.sqrt(max(var, 1e-12))
        intercept_sample = rng.normal(mu, sigma)
        terms.append(intercept_sample)

        # Coef * value samples
        for key, val, _, r_coef in feat_meta:
            ch = _draw_from_topk(r_coef, rng)
            if ch is None:
                coef_mu, coef_var = 0.0, 0.0
            else:
                coef_mu  = float(ch.slots.get("mu", 0.0))
                if "var" in ch.slots:
                    coef_var = float(ch.slots["var"])
                else:
                    coef_sigma_legacy = float(ch.slots.get("sigma", 1.0))
                    coef_var = coef_sigma_legacy * coef_sigma_legacy

            if coef_var > 0.0:
                coef_sample = rng.normal(coef_mu, math.sqrt(coef_var))
            else:
                coef_sample = coef_mu

            terms.append(coef_sample * val)

        z_terms.append(terms)

    z_terms = np.asarray(z_terms, float)

    # if verbose:
    #     print(f"[LR_HEUR] Sampled total z's (first 20 samples):\n{np.sum(z_terms, axis=1)[:20]}")

    # DDM on each sample's terms, then average probabilities & times
    p1_s, rt_s, v_s = [], [], []
    for terms in z_terms:
        p_up, E_RT, v_ratio, _ = ddm_prob_rt_ratio(
            terms, a=ddm_a, s=ddm_s, Tnd=ddm_Tnd, norm=ddm_norm
        )
        p1_s.append(p_up); rt_s.append(E_RT); v_s.append(v_ratio)

    if verbose:
        print(f"DDM A:{ddm_a}, S:{ddm_s}, Tnd:{ddm_Tnd}, norm:{ddm_norm}")
        print(f"v_ratios DMM (20 samples): {np.array(v_s[:20])}")
        print(f"Response Time DMM (20 samples): {np.array(rt_s[:20])}")

    p1 = float(np.mean(p1_s))
    probs = np.array([1.0 - p1, p1], float)

    # Intuitive ops: count features actually considered (exclude intercept)
    intuitive_ops = len(feat_meta)
    computation_cost = T_INTUITIVE_OP * max(0, intuitive_ops)

    total_time = retrieval_cost + read_cost + computation_cost + float(np.mean(rt_s))
    memory.tick(total_time)

    info = {
        "decision": {
            "p1": p1,
            "v_ratio_mean": float(np.mean(v_s)),
        },
        "timing": {
            "retrieval_rt_sum": float(retrieval_cost),
            "read_time_sum": float(read_cost),
            "ddm_rt_mean": float(np.mean(rt_s)),
            "total_time": total_time,
        },
        "chunks": {
            "intercept": {
                "chosen_name": (r_int["top_k"][0][0].name if r_int["top_k"] else None),
            },
            "features": [
                {
                    "key": key,
                    "value": float(val),
                    "chosen_name": (r_coef["top_k"][0][0].name if r_coef["top_k"] else None),
                    "is_numeric": bool(is_numeric),
                }
                for (key, val, is_numeric, r_coef) in feat_meta
            ]
        }
    }
    if verbose:
        print(f"[LR_HEUR] p1={p1:.4f}, total_time={total_time:.2f}s (retrieval {retrieval_cost:.2f}s,\n \
              read {read_cost:.2f}s, intuitive_ops {intuitive_ops}*{T_INTUITIVE_OP:.2f}s, ddm_rt {np.mean(rt_s):.2f}s)")

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


# def cf_lr_heuristic(
#     feature_vector: np.ndarray,   # NORMALIZED 0..1
#     lr_exp: Any,
#     memory: Any,
#     bounds: Dict[str, Tuple[float, float]],   # per key, e.g., "a3": (lo, hi)
#     *,
#     n_runs: int = 200,
#     value_display_sf: int = 2,
#     factor_display_sf: int = 2,
#     compute_sf: int = 2,
#     W0_ANS: float = 0.2,          # division-like noise for delta solving
#     T_READ_NUM: float = 2.0,
#     rng: Optional[np.random.Generator] = None,
#     verbose: bool = False,
# ) -> Dict[str, Dict[str, float]]:
#     """
#     Heuristic-memory CF LR edits with NORMALIZED internal reasoning:
#       - Build expected internal logit z_exp using normalized x (0..1).
#       - Choose ONE feature per run with prob ∝ p_hit_coef * p_hit_val * |coef_mu * x_mean_norm|.
#       - For numeric: compute Δx_norm = -z_exp / (p_hit_coef * p_hit_val * coef_mu),
#         check feasibility in [0,1], then CONVERT to ORIGINAL units and apply:
#             - slider step snap
#             - 2 s.f. display landing ambiguity
#         mean_delta is returned in ORIGINAL units.
#       - For categorical: toggle if currently 0, else Δ=0 (delta returned as 1.0 / 0.0).
#       - mean_time includes expected retrieval latencies + read/act costs.
#     """
#     import numpy as np, math

#     rng = rng or np.random.default_rng()
#     x_norm = np.asarray(feature_vector, dtype=float)
#     k = len(x_norm)

#     # ---------- helpers ----------
#     def _denorm(xn, lo, hi):
#         # xn in [0,1]; if bad bounds, treat as pass-through
#         if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
#             return float(xn)
#         return float(lo + xn * (hi - lo))

#     def _norm_to_orig_delta(xn, dxn, lo, hi):
#         # Δorig = Δnorm * (hi - lo), robust to bad bounds
#         if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
#             return float(dxn)
#         return float(dxn * (hi - lo))

#     # Precompute original x for later landing/snapping (only used after Δnorm is computed)
#     x_orig = np.empty_like(x_norm, dtype=float)
#     for i in range(k):
#         lo, hi = bounds.get(f"a{i}", (float("-inf"), float("inf")))
#         x_orig[i] = _denorm(x_norm[i], lo, hi)

#     # ---------- DM params ----------
#     F, expo, theta, s_act = _dm_params(memory)

#     # ---------- Intercept (probabilistic chunk, normalized internal) ----------
#     intercept_prior_mu, intercept_prior_sigma = 0.0, 0.1
#     ch_int = memory.get_chunk("LR_intercept_prob")
#     if ch_int is not None:
#         intercept_mu    = float(ch_int.slots.get("mu", 0.0))
#         intercept_sigma = float(ch_int.slots.get("sigma", 0.0))
#         p_hit_intercept = float(memory.retrieval_success_prob({"type": "intercept_prob"}))
#     else:
#         intercept_mu, intercept_sigma = intercept_prior_mu, intercept_prior_sigma
#         p_hit_intercept = 0.0

#     rt_int = _expected_rt_from_phit(p_hit_intercept, F, expo, theta, s_act)
#     z_exp = p_hit_intercept * intercept_mu + (1.0 - p_hit_intercept) * intercept_prior_mu

#     # One-time constant time per run (read intercept + retrieval)
#     total_time_const = T_READ_NUM + rt_int

#     # ---------- Coefficients meta ----------
#     coef_meta = []
#     for key, _ in lr_exp.coefficients.items():
#         cname   = _coef_chunk_name(key)
#         ch_coef = memory.get_chunk(cname)
#         if ch_coef is not None:
#             coef_mu    = float(ch_coef.slots.get("mu", 0.0))
#             coef_sigma = float(ch_coef.slots.get("sigma", 0.0))  # not used here
#             p_hit_coef = float(memory.retrieval_success_prob({"type": "coef_prob", "feature_key": key}))
#         else:
#             coef_mu = coef_sigma = 0.0
#             p_hit_coef = 0.0

#         ch_valp = memory.get_chunk(f"value_prob_{key}")
#         p_hit_val = float(ch_valp.slots["p"]) if ch_valp is not None else 1.0

#         if '=' in key:
#             base, cat_idx = key.split('=')
#             col = int(base[1:])
#             # internal check: use normalized/binary view for category
#             x_mean_norm = 1.0 if int(x_norm[col]) == int(cat_idx) else 0.0
#             total_time_const += (T_READ_NUM + T_READ_NUM)  # read + equality
#             is_numeric = False
#             lo = hi = None  # not used
#         else:
#             col = int(key[1:])
#             # INTERNAL reasoning uses normalized x (rounded if you want UI-like read)
#             x_mean_norm = round_to_sf(x_norm[col], value_display_sf)
#             total_time_const += T_READ_NUM
#             is_numeric = True
#             # original bounds for later conversion
#             lo, hi = bounds.get(key, bounds.get(f"a{col}", (float("-inf"), float("inf"))))

#         rt_coef = _expected_rt_from_phit(p_hit_coef, F, expo, theta, s_act)
#         total_time_const += rt_coef

#         coef_meta.append({
#             "key": key,
#             "col": col,
#             "is_numeric": is_numeric,
#             "coef_mu": coef_mu,
#             "p_hit_coef": p_hit_coef,
#             "x_mean_norm": float(x_mean_norm),
#             "p_hit_val": float(p_hit_val),
#             "lo": lo, "hi": hi,        # original units bounds
#         })

#         # INTERNAL expected contribution (normalized x)
#         z_exp += (p_hit_coef * p_hit_val * coef_mu * x_mean_norm)

#     # ---------- selection weights (INTERNAL, normalized x) ----------
#     feature_keys = [m["key"] for m in coef_meta]
#     weights = np.array([
#         abs(m["p_hit_coef"] * m["p_hit_val"] * m["coef_mu"] * m["x_mean_norm"])
#         for m in coef_meta
#     ], dtype=float)
#     if not np.isfinite(weights).all():
#         weights[~np.isfinite(weights)] = 0.0
#     if np.allclose(weights, 0.0):
#         weights[:] = 1.0

#     # Output accumulators; mean_delta in ORIGINAL units
#     out = {k: dict(p_selected=0.0, mean_delta=0.0, mean_time=0.0) for k in feature_keys}

#     # ---------- Monte Carlo ----------
#     for _ in range(n_runs):
#         pool = list(range(len(coef_meta)))
#         used_idx = None
#         applied_delta_orig = 0.0
#         time_here = 0.0

#         while pool:
#             mask = np.zeros(len(coef_meta), dtype=bool); mask[pool] = True
#             p = weights.copy(); p = p * mask
#             idx = (int(rng.choice(pool)) if p.sum() <= 0
#                    else int(rng.choice(len(coef_meta), p=p / p.sum())))

#             m = coef_meta[idx]
#             key, col = m["key"], m["col"]
#             coef_mu  = m["coef_mu"]
#             denom_norm = m["p_hit_coef"] * m["p_hit_val"] * coef_mu  # INTERNAL, normalized space

#             # categorical: toggle if 0 -> 1, else 0 (delta reported as 1.0/0.0)
#             if not m["is_numeric"]:
#                 x_used = float(m["x_mean_norm"])
#                 applied_delta_orig = 1.0 if x_used < 0.5 else 0.0
#                 time_here += T_READ_NUM
#                 used_idx = idx
#                 break

#             # numeric: compute Δx in NORMALIZED space, then convert to ORIGINAL and apply “other method”
#             if abs(denom_norm) < 1e-12:
#                 pool.remove(idx); continue

#             delta_norm = - z_exp / denom_norm
#             x_target_norm = x_norm[col] + delta_norm

#             # Feasibility check in normalized space
#             if not (0.0 <= x_target_norm <= 1.0) or not np.isfinite(x_target_norm):
#                 pool.remove(idx); continue

#             lo, hi = m["lo"], m["hi"]
#             if lo is None or hi is None:
#                 lo, hi = bounds.get(f"a{col}", (float("-inf"), float("inf")))

#             # Convert to ORIGINAL units and apply slider step + 2 s.f. ambiguity
#             # Division timing (use normalized quantities for mental division) + small UI read cost
#             time_here += _mental_division_time(z_exp, denom_norm, compute_sf=compute_sf) + T_READ_NUM

#             x_target_orig = _denorm(x_target_norm, lo, hi)
#             step = _slider_step((lo, hi))
#             snapped = _snap_to_step(x_target_orig, (lo, hi), step)
#             if not (lo <= snapped <= hi):
#                 pool.remove(idx); continue

#             disp = round_to_sf(snapped, 2)
#             half = _sf_halfwidth(disp, 2)
#             x_landed_orig = rng.uniform(disp - half, disp + half)
#             x_landed_orig = _snap_to_step(x_landed_orig, (lo, hi), step)

#             applied_delta_orig = float(x_landed_orig - x_orig[col])
#             used_idx = idx
#             break

#         # fallback: uniform pick with zero move
#         if used_idx is None:
#             used_idx = int(rng.choice(len(coef_meta)))
#             applied_delta_orig = 0.0
#             time_here += T_READ_NUM

#         key_used = coef_meta[used_idx]["key"]
#         s = out[key_used]
#         s["p_selected"] += 1.0
#         s["mean_delta"] += applied_delta_orig    # ORIGINAL units
#         s["mean_time"]  += (total_time_const + time_here)

#     # ---------- normalize ----------
#     for k2, s in out.items():
#         used = s["p_selected"]
#         if used > 0:
#             s["mean_delta"] /= used
#             s["mean_time"]  /= used
#         s["p_selected"] /= float(n_runs)

#     if verbose:
#         total_p = sum(s["p_selected"] for s in out.values())
#         print(f"[CF/heuristic-memory (norm-internal)] Σ p_selected = {total_p:.3f}")

#     return out

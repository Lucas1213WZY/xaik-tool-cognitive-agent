from string import digits
import numpy as np
from .lr_memory import round_to_sf, ddm_prob_rt
from .memory import remember_number_to_sf, build_number_profile, digits_to_value

# ==============================
# 1) Store DT in memory (simple)
# ==============================
def add_dt_to_memory(memory, dt_exp, *, thresh_sf: int = 2):
    """
    Store a decision tree in memory.

    For each internal node nid with feature feat_key:
      - FEATURE:     name="Node_{nid}_feature"
                     slots={"type":"feature","node":nid,"feat_key":feat_key,"depth":d}
      - THRESH PTR:  name="Node_{nid}_thr_ptr"  (numeric only)
                     slots={"type":"thr_ptr","node":nid,"feat_key":feat_key,"thr_key": f"thr:{nid}:{feat_key}","depth":d}
      - THRESH NUM:  remember_number_to_sf(memory, key=f"thr:{nid}:{feat_key}", value=threshold, max_sf=thresh_sf)

    Also writes:
      - NODE TYPE:   "Node_{nid}_type"  {"type":"node_type","node":nid,"is_leaf":bool,"depth":d}
      - CHILD PTRS:  "Node_{nid}_left"/"Node_{nid}_right" -> child nid
      - LEAF CLASS:  "Node_{nid}_class" {"type":"class_label","node":nid,"value":label,"depth":d}
    """
    nodes = {n["node"]: n for n in dt_exp.tree_structure}

    def _walk(nid: int, depth: int = 0):
        n = nodes[nid]
        memory.add_chunk(f"Node_{nid}_type",
                         {"type":"node_type","node":nid,"is_leaf":bool(n["is_leaf"]),"depth":depth})

        if n["is_leaf"]:
            maj = int(np.argmax(n["value"]))
            label = (dt_exp.class_labels[maj]
                     if getattr(dt_exp, "class_labels", None) and maj < len(dt_exp.class_labels)
                     else f"class {maj}")
            memory.add_chunk(f"Node_{nid}_class",
                             {"type":"class_label","node":nid,"value":label,"depth":depth})
            return

        feat_key = n["feature"]
        memory.add_chunk(f"Node_{nid}_feature",
                         {"type":"feature","node":nid,"feat_key":feat_key,"depth":depth})

        # Two-step threshold: pointer + stored number (numeric only)
        if "=" not in feat_key:
            thr_key = f"thr:{nid}:{feat_key}"
            memory.add_chunk(f"Node_{nid}_thr_ptr",
                             {"type":"thr_ptr","node":nid,"feat_key":feat_key,"thr_key":thr_key,"depth":depth})
            remember_number_to_sf(memory, key=thr_key, value=float(n["threshold"]), max_sf=thresh_sf)

        # Child pointers
        memory.add_chunk(f"Node_{nid}_left",
                         {"type":"child_ptr","node":nid,"which":"left","value":n["left"],"depth":depth})
        memory.add_chunk(f"Node_{nid}_right",
                         {"type":"child_ptr","node":nid,"which":"right","value":n["right"],"depth":depth})

        _walk(n["left"], depth+1)
        _walk(n["right"], depth+1)

    _walk(0)

def evidence_firstdiff(val: float, thr: float, sf: int = 2) -> float:
    """
    Calculate evidence for val <= thr using sign/exponent/digit rules.
    Evidence is capped at ±10. If all digits are equal (at given sf),
    evidence = 0.
    """
    import math

    def decompose(v: float, sf: int):
        if v == 0.0:
            return (1, 0, [0] * sf)
        sign = -1 if v < 0 else 1
        v_abs = abs(v)
        p10 = int(math.floor(math.log10(v_abs)))
        scale = 10 ** (sf - 1 - p10)
        m = int(round(v_abs * scale))
        if m >= 10**sf:  # handle carry overflow
            m //= 10
            p10 += 1
        digits = [int(d) for d in f"{m:0{sf}d}"]
        return sign, p10, digits

    s_v, p_v, ds_v = decompose(val, sf)
    s_t, p_t, ds_t = decompose(thr, sf)

    # Sign difference → cap
    if s_v != s_t:
        return 10.0 if s_t > s_v else -10.0

    # Exponent difference → cap
    if p_v != p_t:
        return 10.0 if p_t > p_v else -10.0

    # Digit comparison
    for dv, dt in zip(ds_v, ds_t):
        if dv != dt:
            diff = float(dt - dv)
            return max(-10.0, min(10.0, diff))

    # Tie at all compared digits
    return 0.0

# ---------- helpers ----------
# def _round_sf(v): return float(round_to_sf(float(v), compute_sf))

def _is_cat(k: str) -> bool:
    return ("=" in k)

def _feat_base(k: str) -> str:
    return k.split("=")[0] if "=" in k else k

def dt_traverse(
    feature_vector,
    memory, dt_exp,
    *,
    mode: str = "retrieve",          # "retrieve" or "read"
    compute_sf: int = 2,
    T_enc: float = 2.0,
    # DDM params
    ddm_a: float = 1.5,
    ddm_s: float = 1.0,
    ddm_Tnd: float = 0.30,
    ddm_norm: str = "l2",
    # MC / retrieval planning
    n_mc: int = 64,
    topk_k: int = 3,
    refresh_prob_cap: float = 1.0,   # cap per post-hoc refresh (retrieve-mode only)
    verbose: bool = False,
):
    """
    Returns: (probs(np.array[K]), expected_time, info(dict))

    Simplifications:
      • Only numeric <= and categorical equality (categorical is PERFECT; no DDM errors).
      • mode="read": uses literal tree thresholds/feature labels, but still MC to get a
        distribution via numeric DDM randomness; no memory retrieval or number profiles.
      • mode="retrieve": uses memory top-k for features/threshold pointers and number profiles.
      • No memory.tick() within the MC; we accumulate expected time and tick ONCE at end.
    """
    import random, numpy as np

    rng = random.Random()
    x = np.asarray(feature_vector, float)
    nodes = {n["node"]: n for n in dt_exp.tree_structure}
    num_classes = len(next(n for n in nodes.values() if n["is_leaf"])["value"])
    UNIFORM = np.full(num_classes, 1.0 / num_classes, float)

    if verbose:
        print("\n============================================")
        print(f"[DT_TRAVERSE] mode={mode}, compute_sf={compute_sf}, "
              f"T_enc={T_enc}, ddm_a={ddm_a}, ddm_s={ddm_s}, n_mc={n_mc}")

    # ---------- helpers ----------
    def _round_sf(v): return float(round_to_sf(float(v), compute_sf))

    def _topk_no_refresh(request: dict):
        return memory.topk_retrievals_with_prob_refresh(
            request, k=topk_k, refresh_prob=0.0, add_refresh=False, verbose=False
        )

    def _draw_topk_filtered(ret, rng_, used_chunks):
        """Filter out already-used chunks; include None with its p_none."""
        choices, probs = [], []
        for ch, p in ret["top_k"]:
            if ch is None or ch not in used_chunks:
                choices.append(ch)
                probs.append(float(p))
        choices.append(None)
        probs.append(float(ret["p_none"]))
        tot = sum(probs)
        if tot <= 0:
            return None
        return rng_.choices(choices, weights=probs, k=1)[0]

    def _pick_one(options, rng):
        """Weighted draw over indices; returns (idx, item) or (None, None) if empty/zero mass."""
        if not options:
            return None, None
        weights = [float(o["prob"]) for o in options]
        tot = sum(weights)
        if tot <= 0:
            return None, None
        idx = rng.choices(range(len(options)), weights=weights, k=1)[0]
        return idx, options[idx]

    # ---------- planning (retrieve mode only) ----------
    feature_topk = {}
    thrptr_topk  = {}
    num_profiles = {}

    if mode == "retrieve":
        for n in dt_exp.tree_structure:
            nid = n["node"]
            if n["is_leaf"]:
                continue
            feature_topk[nid] = _topk_no_refresh({"type": "feature", "node": nid})
            if "=" not in n["feature"]:
                thrptr_topk[nid] = _topk_no_refresh({"type": "thr_ptr", "node": nid, "feat_key": n["feature"]})
                for ch, _ in thrptr_topk[nid]["top_k"]:
                    if ch is None:
                        continue
                    thr_key = ch.slots.get("thr_key")
                    if thr_key and thr_key not in num_profiles:
                        num_profiles[thr_key] = build_number_profile(
                            memory, key=thr_key, sf_req=compute_sf,
                            k=topk_k, refresh_prob=0.0, verbose=False
                        )

    # ---------- Monte Carlo runs ----------
    probs_acc = np.zeros(num_classes, float)
    time_acc  = 0.0

    read_time = 0.0
    retrieve_time = 0.0
    decision_time = 0.0

    feature_sel_counts = {}
    thr_key_counts     = {}

    S_runs = max(1, n_mc)

    for _ in range(S_runs):
        node_id = 0
        run_time = 0.0

        # No-reuse trackers for this traversal/path
        used_feature_chunks = set()   # set of chunk objects (feature labels)
        used_thrptr_chunks  = set()   # set of chunk objects (threshold pointers)
        used_thr_keys       = set()   # set of strings (e.g., "thr:nid:feat")

        used_num_meta_chunks  = set()  # str chunk names (or ids)
        used_num_digit_chunks = set()  # str chunk names (any digit pos)

        while True:
            node = nodes[node_id]

            if node["is_leaf"]:
                maj = int(np.argmax(node["value"]))
                probs_acc[maj] += 1.0
                break

            node_feat = node["feature"]
            node_is_cat = _is_cat(node_feat)

            # --- FEATURE KEY ---
            if mode == "read":
                run_time += T_enc
                read_time += T_enc
                feat_key = node_feat           # literal node feature in read-mode
            else:
                retF = feature_topk[node_id]
                run_time += float(retF["expected_rt"])
                retrieve_time += float(retF["expected_rt"])
                chF = _draw_topk_filtered(retF, rng, used_feature_chunks)
                if chF is None:
                    probs_acc += UNIFORM
                    break

                used_feature_chunks.add(chF)
                feature_sel_counts[chF] = feature_sel_counts.get(chF, 0) + 1
                feat_key = chF.slots.get("feat_key", node_feat)

                # ==== VALIDATE retrieved feature against the node ====
                if node_is_cat:
                    if not _is_cat(feat_key) or (feat_key != node_feat):
                        probs_acc += UNIFORM
                        break
                else:
                    if _is_cat(feat_key) or (_feat_base(feat_key) != node_feat):
                        probs_acc += UNIFORM
                        break

            # --- THRESHOLD (numeric only) ---
            thr_val = None
            if not node_is_cat:
                if mode == "read":
                    run_time += T_enc
                    read_time += T_enc
                    thr_val = _round_sf(nodes[node_id]["threshold"])
                else:
                    retP = thrptr_topk.get(node_id)
                    if retP is None:
                        probs_acc += UNIFORM
                        break
                    run_time += float(retP["expected_rt"])
                    retrieve_time += float(retP["expected_rt"])
                    chP = _draw_topk_filtered(retP, rng, used_thrptr_chunks)
                    if chP is None:
                        probs_acc += UNIFORM
                        break
                    used_thrptr_chunks.add(chP)

                    thr_key = chP.slots.get("thr_key")
                    if not thr_key or thr_key not in num_profiles or thr_key in used_thr_keys:
                        probs_acc += UNIFORM
                        break
                    used_thr_keys.add(thr_key)
                    thr_key_counts[thr_key] = thr_key_counts.get(thr_key, 0) + 1

                    prof = num_profiles[thr_key]
                    # --- META sampling (no reuse) ---
                    meta_opts = [o for o in prof["meta_with_chunks"]
                                 if (o["chunk_name"] is None) or (o["chunk_name"] not in used_num_meta_chunks)]
                    if not meta_opts:
                        probs_acc += UNIFORM
                        break

                    _, chosen_meta = _pick_one(meta_opts, rng)
                    if chosen_meta is None or chosen_meta["value"] is None:
                        probs_acc += UNIFORM
                        break

                    if chosen_meta["chunk_name"] is not None:
                        used_num_meta_chunks.add(chosen_meta["chunk_name"])

                    sign, p10 = chosen_meta["value"]

                    # --- DIGIT sampling across positions (no reuse) ---
                    digits = []
                    for pos in range(1, compute_sf + 1):
                        opts = [o for o in prof["digits_with_chunks"][pos - 1]
                                if (o["chunk_name"] is None) or (o["chunk_name"] not in used_num_digit_chunks)]
                        if not opts:
                            break

                        _, pick = _pick_one(opts, rng)
                        if pick is None:
                            break

                        if pick["chunk_name"] is not None:
                            used_num_digit_chunks.add(pick["chunk_name"])

                        if pick["value"] is None:
                            break

                        digits.append(int(pick["value"]))

                    thr_val = digits_to_value(sign, p10, digits, len(digits)) if digits else 0.0
                    thr_val = _round_sf(thr_val)
                    run_time += float(prof["expected_rt"])  # number build + encode threshold
                    retrieve_time += float(prof["expected_rt"])

            # --- READ STIMULUS VALUE ---
            if node_is_cat:
                base, cat_idx = node_feat.split("=")
                att = int(base[1:])
                run_time += T_enc
                read_time += T_enc
                is_member = (int(x[att]) == int(cat_idx))
            else:
                att = int(node_feat[1:])
                run_time += T_enc
                read_time += T_enc
                val = float(x[att])

            # --- DECISION AT NODE ---
            if node_is_cat:
                go_left = bool(is_member)     # perfect for categorical
                rt_dec = 0.0
            else:
                e = evidence_firstdiff(val, thr_val, sf=compute_sf)
                if val == thr_val:
                    e = 0
                elif val < thr_val:
                    e = 1
                else:
                    e = -1
                p_up, E_RT, _ = ddm_prob_rt(e, a=ddm_a, s=ddm_s, Tnd=ddm_Tnd, gain=1.0)
                go_left = (rng.random() < p_up)
                rt_dec = E_RT
            decision_time += rt_dec
            run_time += rt_dec

            node_id = node["left"] if go_left else node["right"]

        time_acc += run_time

    # ---------- Aggregate ----------
    probs = probs_acc / float(S_runs)
    expected_time = float(time_acc) / float(S_runs) if S_runs > 0 else 0.0

    read_time /= float(S_runs) if S_runs > 0 else 0.0
    retrieve_time /= float(S_runs) if S_runs > 0 else 0.0
    decision_time /= float(S_runs) if S_runs > 0 else 0.0

    if verbose:
        print(f"Time Breakdown (averaged over {S_runs} runs):")
        print(f"  → read_time: {read_time:.3f}s")
        print(f"  → retrieve_time: {retrieve_time:.3f}s")
        print(f"  → decision_time: {decision_time:.3f}s")
        print(f"  → total expected_time: {expected_time:.3f}s")

    # Post-hoc refresh (retrieve mode only)
    if mode == "retrieve" and feature_sel_counts:
        now = memory.dm.time if hasattr(memory, "dm") else memory.time
        for ch, c in feature_sel_counts.items():
            pr = min(refresh_prob_cap, float(c) / float(S_runs))
            if pr > 0.0:
                ch.add_prob_refresh(now, pr)
    if mode == "retrieve" and thr_key_counts:
        for thr_key, c in thr_key_counts.items():
            pr = min(refresh_prob_cap, float(c) / float(S_runs))
            if pr > 0.0:
                build_number_profile(
                    memory, key=thr_key, sf_req=compute_sf,
                    k=topk_k, refresh_prob=pr, verbose=False
                )

    # Single final tick with expected time
    memory.tick(expected_time)

    info = {
        "mode": mode,
        "n_mc": int(S_runs),
        "compute_sf": int(compute_sf),
        "ddm": {"a": ddm_a, "s": ddm_s, "Tnd": ddm_Tnd, "norm": ddm_norm},
        "T_enc": T_enc,
        "refresh_counts": {
            "feature": {getattr(ch, "name", ""): int(c) for ch, c in feature_sel_counts.items()},
            "thr_key": dict(thr_key_counts),
        }
    }
    return probs, expected_time, info

def refresh_dt_path_in_memory(
    memory, dt_exp, feature_vector, *, thresh_sf: int = 2
):
    """
    Deterministically traverse the DT using dt_exp thresholds and the instance values,
    and refresh the chunks that *would* be used along that path (no probabilities).
    Uses chunk.update_retrieval(now) for a direct, non-probabilistic refresh.
    """
    import numpy as np

    x = np.asarray(feature_vector, float)
    nodes = {n["node"]: n for n in dt_exp.tree_structure}
    now = memory.dm.time if hasattr(memory, "dm") else memory.time

    nid = 0
    while True:
        n = nodes[nid]

        # Leaf: refresh its class chunk and stop
        if n["is_leaf"]:
            leaf_nm = f"Node_{nid}_class"
            ch = memory.get_chunk(leaf_nm)
            if ch: ch.update_retrieval(now)
            break

        # Feature chunk (used at this node)
        feat_key = n["feature"]
        feat_nm = f"Node_{nid}_feature"
        chf = memory.get_chunk(feat_nm)
        if chf: chf.update_retrieval(now)

        # Decide branch using literal values from dt_exp / instance
        if "=" in feat_key:
            base, cat_idx = feat_key.split("=")
            att = int(base[1:])
            is_member = (int(x[att]) == int(cat_idx))
            # Child pointer chunk actually used
            child_nm = f"Node_{nid}_{'left' if is_member else 'right'}"
            ch_child = memory.get_chunk(child_nm)
            if ch_child: ch_child.update_retrieval(now)
            # Move to child
            nid = n["left"] if is_member else n["right"]

        else:
            # Numeric threshold path
            # Refresh the threshold pointer chunk
            thr_ptr_nm = f"Node_{nid}_thr_ptr"
            chp = memory.get_chunk(thr_ptr_nm)
            if chp: chp.update_retrieval(now)

            # Also refresh the number memory that was stored by remember_number_to_sf
            thr_key = f"thr:{nid}:{feat_key}"
            meta_nm = f"num:{thr_key}:meta"
            chm = memory.get_chunk(meta_nm)
            if chm: chm.update_retrieval(now)

            # Refresh first `thresh_sf` digit chunks (if present)
            for pos in range(1, thresh_sf + 1):
                d_nm = f"num:{thr_key}:d{pos}"
                cd = memory.get_chunk(d_nm)
                if cd: cd.update_retrieval(now)

            # Compare and follow the used child; refresh that pointer
            att = int(feat_key[1:])
            go_left = (float(x[att]) <= float(n["threshold"]))
            child_nm = f"Node_{nid}_{'left' if go_left else 'right'}"
            ch_child = memory.get_chunk(child_nm)
            if ch_child: ch_child.update_retrieval(now)
            nid = n["left"] if go_left else n["right"]

from typing import Any, Dict, Tuple, Optional, Callable
import numpy as np

# Reuse your helpers
from .lr_memory import _base_index_from_key, _slider_step, _snap_to_step, round_to_sf
def cf_change_path_dt(
    feature_vector: np.ndarray,
    dt_exp: Any,
    bounds: Dict[str, Tuple[float, float]],
    *,
    mode: str,                              # "read" (with XAI) or "retrieve" (without XAI)
    chosen_depth: Optional[int] = None,     # Laplace center over decision depths
    # display / rounding
    value_display_sf: int = 2,
    compute_sf: int = 2,
    # timing (aligned with dt_traverse)
    T_enc: float = 2.0,
    # DDM params (numeric decisions)
    ddm_a: float = 1.5,
    ddm_s: float = 1.0,
    ddm_Tnd: float = 0.30,
    ddm_norm: str = "l2",
    # retrieve-mode configs
    memory: Any = None,
    n_mc: int = 64,
    topk_k: int = 3,
    refresh_prob_cap: float = 1.0,          # used only in retrieve-mode
    # depth selection smoothing
    tau: float = 1.0,                       # smaller => sharper around center
    depth_eps: float = 1e-9,                # tiny floor
    # rng
    rng: Optional[np.random.Generator] = None,
    return_depth_info: bool = False,
    verbose: bool = False,
):
    """
    With XAI ("read"):   Expected selection over depths via Laplace weights (no depth sampling).
    Without XAI ("retrieve"): Monte Carlo; each run samples depth via Laplace, traverses stochastically,
                              then attempts a minimal flip at the chosen depth.

    Returns:
      out: dict { "a0": {"p_selected", "mean_delta", "mean_time"}, ... }
      If return_depth_info=True: (out, {"depth_probs": list[float], "chosen_depth": int})
    """
    import numpy as np, math, random

    rng = rng or np.random.default_rng()
    rnd = random.Random(rng.integers(0, 2**31 - 1))

    x = np.asarray(feature_vector, dtype=float)
    nodes = {n["node"]: n for n in dt_exp.tree_structure}
    k = len(x)

    # ---------- small helpers ----------
    def _is_cat_key(key: str) -> bool:
        return "=" in key

    def _feat_key_for_node(n) -> str:
        return n["feature"]

    def _thr_for_node(n) -> float:
        return float(n["threshold"])

    def _left(n) -> int:
        return int(n["left"])

    def _right(n) -> int:
        return int(n["right"])

    def _base_index_from_key(key: str) -> int:
        base = key.split('=')[0]  # "aN" from "aN" or "aN=K"
        return int(base[1:])

    def _bounds_for_feat_key(fk: str) -> Tuple[float, float]:
        if _is_cat_key(fk):
            return (-np.inf, np.inf)
        return bounds.get(fk, bounds.get(f"a{_base_index_from_key(fk)}", (-np.inf, np.inf)))

    def _slider_step(bounds_: Tuple[float, float]) -> float:
        lo, hi = bounds_
        r = max(hi - lo, 1e-12)
        return 10 ** (math.floor(math.log10(r)) - 1)

    def _snap_to_step(xv: float, bounds_: Tuple[float, float], step: float) -> float:
        lo, hi = bounds_
        xv = min(max(xv, lo), hi)
        return lo + round((xv - lo) / step) * step

    def _round_sf(v: float) -> float:
        return float(round_to_sf(float(v), compute_sf))

    def _literal_path_for_x(x_arr: np.ndarray) -> list:
        """Deterministic literal route (uses true node thresholds & x<=thr for branch)."""
        path = []
        nid = 0
        while True:
            n = nodes[nid]
            path.append(n)
            if n["is_leaf"]:
                break
            fk = _feat_key_for_node(n)
            if _is_cat_key(fk):
                base, cat_idx = fk.split("=")
                col = _base_index_from_key(base)
                go_left = int(x_arr[col]) == int(cat_idx)
            else:
                col = _base_index_from_key(fk)
                thr = _thr_for_node(n)
                go_left = float(x_arr[col]) <= float(thr)
            nid = _left(n) if go_left else _right(n)
        return path

    def _depth_weights(m: int, center: int) -> np.ndarray:
        idx = np.arange(m)
        w = np.exp(-np.abs(idx - center) / max(tau, 1e-6)) + float(depth_eps)
        return (w / w.sum()).astype(float)

    def _flip_at_node(x_arr: np.ndarray, n, verbose=False) -> Tuple[Optional[str], float, float]:
        """
        Minimal flip across the test at node n.
        Returns (feat_key_used, delta_applied, extra_time).
        """
        fk = _feat_key_for_node(n)
        if _is_cat_key(fk):
            base, cat_idx = fk.split("=")
            col = _base_index_from_key(base)
            cur = int(x_arr[col])
            x_new = int(cat_idx) if cur != int(cat_idx) else (int(cat_idx) ^ 1)
            delta = float(x_new - cur)
            x_arr[col] = x_new
            return fk, delta, T_enc  # one encode/change
        # numeric
        col = _base_index_from_key(fk)
        thr = _thr_for_node(n)
        lo, hi = _bounds_for_feat_key(fk)
        step = _slider_step((lo, hi))
        cur = float(x_arr[col])
        target = thr + step if cur <= thr else thr - step
        target = _snap_to_step(target, (lo, hi), step)
        if verbose:
            print(f"Feature {fk}, Current value: {cur}, Threshold: {thr}, Target: {target}, Step: {step}")
        # target = _round_sf(target)
        if not (lo <= target <= hi) or np.isclose(target, cur):
            return None, 0.0, T_enc
        delta = float(target - cur)
        x_arr[col] = float(target)
        return fk, delta, 2.0 * T_enc  # read + operate

    # ---------- READ MODE (with XAI): expected distribution over depths ----------
    if mode == "read":
        # literal path defines decision nodes / indices
        path = _literal_path_for_x(x.copy())
        dec_nodes = [n for n in path if not n["is_leaf"]]
        if not dec_nodes:
            out = {f"a{i}": {"p_selected": 0.0, "mean_delta": 0.0, "mean_time": 0.0} for i in range(k)}
            depth_info = {"depth_probs": [], "chosen_depth": None}
            return (out, depth_info) if return_depth_info else out

        m = len(dec_nodes)
        center = max(0, min(chosen_depth if chosen_depth is not None else m - 1, m - 1))
        w = _depth_weights(m, center)  # Laplace over indices

        # Precompute cumulative expected time to reach each depth index using DDM RT (numeric)
        # We use the literal route to know which nodes are encountered up to index i,
        # but still add expected DDM decision time at numeric nodes.
        cum_time = np.zeros(m, float)
        running = 0.0
        for i, n in enumerate(dec_nodes):
            fk = _feat_key_for_node(n)
            if _is_cat_key(fk):
                # encode feature + stimulus + 0 decision RT
                running += T_enc + T_enc
            else:
                # encode feature + encode threshold + encode stimulus + expected decision RT
                col = _base_index_from_key(fk)
                thr_val = _round_sf(_thr_for_node(n))
                val = float(x[col])
                # simple firstdiff evidence sign as in dt_traverse
                if val == thr_val:
                    e = 0
                elif val < thr_val:
                    e = 1
                else:
                    e = -1
                _, E_RT, _ = ddm_prob_rt(e, a=ddm_a, s=ddm_s, Tnd=ddm_Tnd, gain=1.0)
                running += T_enc + T_enc + T_enc + float(E_RT)
            cum_time[i] = running

        # Aggregate expected selection outcomes over depths
        out = {f"a{i}": {"p_selected": 0.0, "mean_delta": 0.0, "mean_time": 0.0} for i in range(k)}
        for i, n in enumerate(dec_nodes):
            x_tmp = x.copy()
            fk, dlt, t_flip = _flip_at_node(x_tmp, n)
            if fk is None:
                continue
            col = _base_index_from_key(fk.split("=")[0] if "=" in fk else fk)
            out_key = f"a{col}"
            out[out_key]["p_selected"] += float(w[i])
            out[out_key]["mean_delta"] += float(w[i]) * float(dlt)
            out[out_key]["mean_time"]  += float(w[i]) * float(cum_time[i] + t_flip)

            # print("Node threshold: ", n["threshold"], "Feature: ", n["feature"], "Current value: ", x[_base_index_from_key(n["feature"])], "Delta: ", dlt)


        # normalize mean_time/delta by probability mass per feature (so they are conditional means)
        for key, vals in out.items():
            p = vals["p_selected"]
            if p > 0:
                vals["mean_delta"] = vals["mean_delta"] / p
                vals["mean_time"]  = vals["mean_time"]  / p

        if verbose:
            print(f"[CF_DT][read] center={center}, m={m}, expected total time={sum(v['p_selected']*v['mean_time'] for v in out.values()):.3f}s")

        depth_info = {"depth_probs": w.tolist(), "chosen_depth": int(center)}
        expected_time = sum(v["p_selected"] * v["mean_time"] for v in out.values())
        out["expected_time"] = float(expected_time)
        return out

    # ---------- RETRIEVE MODE (without XAI): Monte-Carlo with Laplace depth each run ----------
    if mode != "retrieve":
        raise ValueError('mode must be "read" or "retrieve"')

    if memory is None:
        raise ValueError("mode='retrieve' requires a `memory` object.")

    # literal path only to define max possible decision-depth cardinality for Laplace
    lit_path = _literal_path_for_x(x.copy())
    lit_dec_nodes = [n for n in lit_path if not n["is_leaf"]]
    if not lit_dec_nodes:
        out = {f"a{i}": {"p_selected": 0.0, "mean_delta": 0.0, "mean_time": 0.0} for i in range(k)}
        out["expected_time"] = 0.0
        return out

    m = len(lit_dec_nodes)
    center = max(0, min(chosen_depth if chosen_depth is not None else m - 1, m - 1))
    # print("Center: ", center)
    base_w = _depth_weights(m, center)

    # ===== Pre-plan retrieval menus (no refresh, no time advance) =====
    def _topk_no_refresh(req: dict):
        return memory.topk_retrievals_with_prob_refresh(
            req, k=topk_k, refresh_prob=0.0, add_refresh=False, verbose=False
        )

    feature_topk = {}
    thrptr_topk  = {}
    num_profiles = {}

    for n in dt_exp.tree_structure:
        if n["is_leaf"]:
            continue
        nid = n["node"]
        feature_topk[nid] = _topk_no_refresh({"type": "feature", "node": nid})

        if not _is_cat_key(n["feature"]):
            thrptr_topk[nid] = _topk_no_refresh({"type": "thr_ptr", "node": nid, "feat_key": n["feature"]})
            # pre-build number profiles for any threshold-pointer choices
            for ch, _p in thrptr_topk[nid].get("top_k", []):
                if ch is None:
                    continue
                thr_key = ch.slots.get("thr_key")
                if thr_key and thr_key not in num_profiles:
                    num_profiles[thr_key] = build_number_profile(
                        memory, key=thr_key, sf_req=compute_sf, k=topk_k, refresh_prob=0.0, verbose=False
                    )

    # ===== Helpers for MC traversal using pre-planned menus =====
    def _draw_topk_filtered(ret, used_set):
        """Sample from pre-planned top-k but avoid reusing chunks; include None via p_none."""
        choices, probs = [], []
        for ch, p in ret.get("top_k", []):
            if ch is None or ch not in used_set:
                choices.append(ch)
                probs.append(float(p))
        choices.append(None)
        probs.append(float(ret.get("p_none", 0.0)))
        tot = sum(probs)
        if tot <= 0:
            return None
        return rnd.choices(choices, weights=probs, k=1)[0]

    def _pick_one(options):
        if not options:
            return None, None
        w = [float(o["prob"]) for o in options]
        tot = sum(w)
        if tot <= 0:
            return None, None
        idx = rnd.choices(range(len(options)), weights=w, k=1)[0]
        return idx, options[idx]

    def _traverse_once_to_depth_retrieve_planned(x_arr: np.ndarray, target_idx: int):
        """
        Uses pre-planned feature_topk, thrptr_topk, and num_profiles.
        Returns:
          path_nodes: list of nodes visited
          run_time: float
          usage: dict (for refresh accounting)
          thr_seq: list of threshold values per decision node in order (None for categorical)
        """
        path = []
        node_id = 0
        run_time = 0.0
        d = 0

        used_feature_chunks = set()
        used_thrptr_chunks  = set()
        used_thr_keys       = set()
        used_num_meta_chunks  = set()
        used_num_digit_chunks = set()

        feat_chunk_counts = {}
        thr_key_local     = {}
        thr_seq           = []  # aligned with decision nodes encountered on this path

        while True:
            n = nodes[node_id]
            path.append(n)
            if n["is_leaf"]:
                break
            if d == target_idx:
                break

            node_feat = n["feature"]
            is_cat = _is_cat_key(node_feat)

            # --- feature retrieval ---
            retF = feature_topk[n["node"]]
            run_time += float(retF.get("expected_rt", 0.0))
            chF = _draw_topk_filtered(retF, used_feature_chunks)
            if chF is None:
                break
            used_feature_chunks.add(chF)
            feat_chunk_counts[chF] = feat_chunk_counts.get(chF, 0) + 1
            feat_key = chF.slots.get("feat_key", node_feat)

            # validate against node
            if is_cat:
                if feat_key != node_feat:
                    break
            else:
                if _is_cat_key(feat_key) or (feat_key.split('=')[0] != node_feat):
                    break

            # --- threshold retrieval (numeric only) ---
            thr_val = None
            if not is_cat:
                retP = thrptr_topk.get(n["node"])
                if retP is None:
                    break
                run_time += float(retP.get("expected_rt", 0.0))
                chP = _draw_topk_filtered(retP, used_thrptr_chunks)
                if chP is None:
                    break
                used_thrptr_chunks.add(chP)

                thr_key = chP.slots.get("thr_key")
                if not thr_key or (thr_key not in num_profiles) or (thr_key in used_thr_keys):
                    break
                used_thr_keys.add(thr_key)
                thr_key_local[thr_key] = thr_key_local.get(thr_key, 0) + 1

                prof = num_profiles[thr_key]

                # meta
                meta_opts = [o for o in prof["meta_with_chunks"]
                             if (o["chunk_name"] is None) or (o["chunk_name"] not in used_num_meta_chunks)]
                if not meta_opts:
                    break
                _, chosen_meta = _pick_one(meta_opts)
                if chosen_meta is None or chosen_meta["value"] is None:
                    break
                if chosen_meta["chunk_name"] is not None:
                    used_num_meta_chunks.add(chosen_meta["chunk_name"])
                sign, p10 = chosen_meta["value"]

                # digits
                digits = []
                for pos in range(1, compute_sf + 1):
                    opts = [o for o in prof["digits_with_chunks"][pos - 1]
                            if (o["chunk_name"] is None) or (o["chunk_name"] not in used_num_digit_chunks)]
                    if not opts:
                        break
                    _, pick = _pick_one(opts)
                    if pick is None or pick["value"] is None:
                        break
                    if pick["chunk_name"] is not None:
                        used_num_digit_chunks.add(pick["chunk_name"])
                    digits.append(int(pick["value"]))
                thr_val = digits_to_value(sign, p10, digits, len(digits)) if digits else 0.0
                thr_val = _round_sf(thr_val)
                run_time += float(prof["expected_rt"])

            # read stimulus & DDM decision
            if is_cat:
                base, cat_idx = node_feat.split("=")
                att = int(base[1:])
                run_time += T_enc
                is_member = (int(x_arr[att]) == int(cat_idx))
                go_left = bool(is_member)
                thr_seq.append(None)  # categorical
            else:
                att = int(node_feat[1:])
                run_time += T_enc
                val = float(x_arr[att])
                if val == thr_val:
                    e = 0
                elif val < thr_val:
                    e = 1
                else:
                    e = -1
                p_up, E_RT, _ = ddm_prob_rt(e, a=ddm_a, s=ddm_s, Tnd=ddm_Tnd, gain=1.0)
                go_left = (rnd.random() < p_up)
                run_time += float(E_RT)
                thr_seq.append(float(thr_val))

            node_id = _left(n) if go_left else _right(n)
            d += 1

        usage = {"feature_chunks": feat_chunk_counts, "thr_keys": thr_key_local}
        return path, run_time, usage, thr_seq


    # ====== Monte-Carlo using preplanned menus ======
    select_mass = {f"a{i}": 0.0 for i in range(k)}
    delta_sum   = {f"a{i}": 0.0 for i in range(k)}
    time_sum    = {f"a{i}": 0.0 for i in range(k)}

    total_time_runs = 0.0

    # refresh ledgers
    feature_sel_counts = {}
    thr_key_counts     = {}

    # threshold accumulation for features we actually flipped at (for saving)
    thr_acc_sum  = {}  # feature -> sum(threshold used at picked node)
    thr_acc_cnt  = {}  # feature -> count

    N_runs = max(1, int(n_mc))
    N = 0
    for _ in range(N_runs):
        depth_idx = int(rng.choice(np.arange(m), p=base_w))
        path_stoch, run_time, usage, thr_seq = _traverse_once_to_depth_retrieve_planned(x.copy(), depth_idx)
        total_time_runs += float(run_time)

        # aggregate for refresh
        for ch, c in usage["feature_chunks"].items():
            feature_sel_counts[ch] = feature_sel_counts.get(ch, 0) + int(c)
        for tk, c in usage["thr_keys"].items():
            thr_key_counts[tk] = thr_key_counts.get(tk, 0) + int(c)

        dec_nodes = [n for n in path_stoch if not n["is_leaf"]]
        if not dec_nodes:
            continue
        picked_idx = min(depth_idx, len(dec_nodes) - 1)

        # attempt flip at chosen depth
        x_tmp = x.copy()
        fk, dlt, t_flip = _flip_at_node(x_tmp, dec_nodes[picked_idx])
        run_time += float(t_flip)

        if fk is None:
            continue

        col = _base_index_from_key(fk.split("=")[0] if "=" in fk else fk)
        fkey = f"a{col}"
        select_mass[fkey] += 1.0
        delta_sum[fkey]   += float(dlt)
        time_sum[fkey]    += float(run_time)

        # record threshold used at this decision depth if numeric
        thr_here = thr_seq[picked_idx] if (picked_idx < len(thr_seq)) else None
        if thr_here is not None:
            thr_acc_sum[fkey] = thr_acc_sum.get(fkey, 0.0) + float(thr_here)
            thr_acc_cnt[fkey] = thr_acc_cnt.get(fkey, 0) + 1

        N += 1
    # produce output: probabilities and conditional means
    out = {f"a{i}": {"p_selected": 0.0, "mean_delta": 0.0, "mean_time": 0.0} for i in range(k)}
    # N = float(N_runs)
    for key in out.keys():
        p = select_mass[key] / N
        out[key]["p_selected"] = p
        if select_mass[key] > 0:
            out[key]["mean_delta"] = delta_sum[key] / select_mass[key]
            out[key]["mean_time"]  = time_sum[key]  / select_mass[key]

    # ===== Post-hoc refresh (retrieve-mode only) =====
    now = getattr(getattr(memory, "dm", None), "time", getattr(memory, "time", 0.0))
    if feature_sel_counts:
        for ch, c in feature_sel_counts.items():
            pr = min(float(refresh_prob_cap), float(c) / float(N))
            if pr > 0.0 and hasattr(ch, "add_prob_refresh"):
                ch.add_prob_refresh(now, pr)
    if thr_key_counts:
        for thr_key, c in thr_key_counts.items():
            pr = min(float(refresh_prob_cap), float(c) / float(N))
            if pr > 0.0:
                build_number_profile(
                    memory, key=thr_key, sf_req=compute_sf,
                    k=topk_k, refresh_prob=pr, verbose=False
                )

    # ===== Tick memory by expected time =====
    expected_time = 0
    for v in out.values():
        expected_time += float(v["p_selected"] * v["mean_time"])
    if hasattr(memory, "tick"):
        memory.tick(expected_time)

    if verbose:
        print(f"[CF_DT][retrieve] center={center}, m={m}, n_mc={int(N)}, "
              f"expected_time={expected_time:.3f}s, selected_any={sum(1 for v in select_mass.values() if v>0)}")

    # === SAVE DT COMBO (direction-agnostic): type="dt_change_combo" ===
    out["expected_time"] = float(expected_time)

    if memory is not None and len(out) > 0:
        mem_core = getattr(memory, "dm", memory)
        mem_time = getattr(mem_core, "time", 0.0)

        feats = [f for f, v in out.items() if f != "expected_time" and v.get("p_selected", 0.0) > 0.0]
        if feats:
            # renormalize p over kept features
            Z = sum(out[f]["p_selected"] for f in feats) or 1.0
            p_norm = {f: out[f]["p_selected"] / Z for f in feats}
            t_map  = {f: float(out[f]["mean_time"]) for f in feats}

            # thresholds (use MC-averaged thresholds if we flipped there; else fallback to literal tree)
            thr_map = {}
            for f in feats:
                if thr_acc_cnt.get(f, 0) > 0:
                    thr_map[f] = float(thr_acc_sum[f] / thr_acc_cnt[f])
                else:
                    # fallback: first node in the tree using this feature key
                    node_thr = None
                    base_key = f  # "a{i}"
                    for n in dt_exp.tree_structure:
                        if not n["is_leaf"] and n["feature"] == base_key:
                            node_thr = float(n["threshold"])
                            break
                    thr_map[f] = float(node_thr) if node_thr is not None else 0.0

            exp_time = float(sum(p_norm[f] * t_map[f] for f in feats))
            raw_mass = float(sum(out[f]["p_selected"] for f in feats))
            refresh_p = max(0.0, min(1.0, raw_mass))

            chunk_name = "dt_change_combo"
            ch = memory.get_chunk(chunk_name) if hasattr(memory, "get_chunk") else None

            if ch is None:
                slots = {
                    "type": "dt_change_combo",
                    "features": feats,
                    "p_select": {f: float(p_norm[f]) for f in feats},
                    "threshold": {f: float(thr_map[f]) for f in feats},
                    "time": {f: float(t_map[f]) for f in feats},
                    "mass": float(raw_mass),
                    "n_updates": 1,
                    "expected_time": exp_time,
                }
                ch = memory.add_chunk(chunk_name, slots, update_retrieval=False)
            else:
                # merge with mass-weighted running means across union
                s = ch.slots
                w_old = float(s.get("mass", 0.0))
                w_new = w_old + raw_mass if (w_old + raw_mass) > 0 else raw_mass + 1e-12

                old_feats = set(s.get("features", []))
                all_feats = sorted(old_feats | set(feats))

                p_old  = dict(s.get("p_select", {}))
                t_old  = dict(s.get("time", {}))
                th_old = dict(s.get("threshold", {}))

                p_new  = {f: p_norm.get(f, 0.0) for f in all_feats}
                t_new  = {f: float(t_map.get(f, t_old.get(f, 0.0))) for f in all_feats}
                th_new = {f: float(thr_map.get(f, th_old.get(f, 0.0))) for f in all_feats}

                p_upd, t_upd, th_upd = {}, {}, {}
                for f in all_feats:
                    p_upd[f]  = (p_old.get(f, 0.0) * w_old + p_new[f] * raw_mass) / w_new
                    t_upd[f]  = (t_old.get(f, 0.0) * w_old + t_new[f] * raw_mass) / w_new
                    th_upd[f] = (th_old.get(f, 0.0) * w_old + th_new[f] * raw_mass) / w_new

                Z2 = sum(p_upd.values()) or 1.0
                for f in p_upd:
                    p_upd[f] /= Z2

                s["features"]  = list(all_feats)
                s["p_select"]  = {f: float(p_upd[f]) for f in all_feats}
                s["threshold"] = {f: float(th_upd[f]) for f in all_feats}
                s["time"]      = {f: float(t_upd[f]) for f in all_feats}
                s["mass"]      = float(w_new)
                s["n_updates"] = int(s.get("n_updates", 0)) + 1
                s["expected_time"] = float(sum(p_upd[f] * t_upd[f] for f in all_feats))

            if ch is not None and hasattr(ch, "add_prob_refresh"):
                ch.add_prob_refresh(mem_time, refresh_p)

    return out


from typing import Any, Dict, Optional
from .memory import CombinedMemory
from typing import Any, Dict, Optional, Tuple
import math
import numpy as np
def recall_change_dt(
    feature_vector: np.ndarray,
    memory: Any,
    bounds: Dict[str, Tuple[float, float]],
    *,
    compute_sf: int = 2,
    k: int = 2,                 # merge up to top-k combo chunks
    refresh_prob: float = 1.0,
    T_enc: Optional[float] = 2.0,   # NEW: encoding/read time for the value
    verbose: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Returns:
      {
        "<feature_key>": {"p_selected": float, "mean_delta": float, "mean_time": float},
        ...
        "expected_time": float
      }

    Reads combo chunks saved by cf_change_path_dt with:
      type="dt_change_combo",
      fields: features, p_select, threshold, mass, expected_time  # (time in chunk is IGNORED)

    For each numeric feature 'aN', computes the minimal signed delta needed to cross the
    stored threshold from the CURRENT feature_vector, with snapping to the slider step
    and respecting (lo, hi) bounds. Categorical tests are ignored.
    Mean time per feature = retrieval_time (of combo recall) + one T_enc.
    """

    x = np.asarray(feature_vector, dtype=float)

    # --- helpers ---
    def _slider_step(bnds: Tuple[float, float]) -> float:
        lo, hi = bnds
        r = max(hi - lo, 1e-12)
        return 10 ** (math.floor(math.log10(r)) - 1)

    def _snap_to_step(v: float, bnds: Tuple[float, float], step: float) -> float:
        lo, hi = bnds
        v = min(max(v, lo), hi)
        return lo + round((v - lo) / step) * step

    def _base_index_from_key(f: str) -> int:
        # f is like "a3"
        return int(f[1:])

    # --- retrieve up to k dt_change_combo chunks and capture retrieval latency ---
    def _retrieve_dt_combos():
        rt = 0.0
        if hasattr(memory, "topk_retrievals_with_prob_refresh"):
            res = memory.topk_retrievals_with_prob_refresh(
                request={"type": "dt_change_combo"},
                k=k, refresh_prob=refresh_prob, add_refresh=True, verbose=verbose
            )
            # Try common fields for retrieval latency
            rt = float(res.get("retrieval_time", res.get("rt", 0.0)))
            chunks = [ch for ch, _p in res.get("top_k", []) if ch is not None]
            return chunks, rt
        # bare DM fallback (no latency info)
        chunks = [ch for ch in getattr(memory, "chunks", [])
                  if getattr(ch, "slots", {}).get("type") == "dt_change_combo"]
        return chunks, rt

    chunks, retrieval_rt = _retrieve_dt_combos()
    if not chunks:
        if verbose:
            print("[recall_change_dt] no dt_change_combo chunks found.")
        return {"expected_time": float(retrieval_rt + T_enc)}

    # --- mass-weighted mixture across chunks, then renormalize across features ---
    per_feat = {}     # feat -> accumulators over (mass * p_select)
    total_mass = 0.0

    for ch in chunks:
        s = getattr(ch, "slots", {}) or {}
        feats   = list(s.get("features", []))            # ["a0","a1",...]
        p_map   = dict(s.get("p_select", {}))            # feature -> prob within chunk
        thr_map = dict(s.get("threshold", {}))           # feature -> threshold (float, numeric only)
        mass    = float(s.get("mass", 1.0))              # chunk mass
        total_mass += mass

        for f in feats:
            p_f   = float(p_map.get(f, 0.0))
            thr_f = float(thr_map.get(f, float("nan")))  # NaN for non-numeric
            w     = mass * p_f

            acc = per_feat.setdefault(f, {
                "p_wsum": 0.0, "thr_wsum": 0.0, "thr_wsum_count": 0.0
            })
            acc["p_wsum"] += w
            if math.isfinite(thr_f):
                acc["thr_wsum"]       += w * thr_f
                acc["thr_wsum_count"] += w

    if not per_feat:
        if verbose:
            print("[recall_change_dt] combo chunks had no features.")
        return {"expected_time": float(retrieval_rt + T_enc)}

    # --- renormalize p across features to form a distribution ---
    Z = sum(v["p_wsum"] for v in per_feat.values()) or 1.0

    # mean_time is identical for all features under this rule:
    mean_time_const = float(retrieval_rt + T_enc)

    out: Dict[str, Dict[str, float]] = {}
    for f, v in per_feat.items():
        p_feat = v["p_wsum"] / Z

        # Compute mean_delta from CURRENT x against the mixed threshold (numeric only)
        mean_delta = 0.0
        if v["thr_wsum_count"] > 1e-12:
            thr_mean = v["thr_wsum"] / v["thr_wsum_count"]

            col = _base_index_from_key(f)
            cur = float(x[col])

            lo, hi = bounds.get(f, bounds.get(f"a{col}", (-float("inf"), float("inf"))))
            step = _slider_step((lo, hi))

            # Minimal signed move to cross the threshold with a 1-step nudge
            if cur <= thr_mean:
                target = thr_mean + step
            else:
                target = thr_mean - step
            target = _snap_to_step(target, (lo, hi), step)

            # If target collapsed to cur due to bounds/snap, push one more step.
            if target == cur:
                target = _snap_to_step(target + (step if cur <= thr_mean else -step), (lo, hi), step)

            mean_delta = float(target - cur)

        out[f] = {
            "p_selected": float(p_feat),
            "mean_delta": float(mean_delta),
            "mean_time":  mean_time_const,   # <-- retrieval_rt + T_enc
        }

        if verbose:
            if v["thr_wsum_count"] > 0:
                thr_mean_dbg = v["thr_wsum"] / v["thr_wsum_count"]
                print(f"[recall_change_dt] {f}: p={p_feat:.3f}, thr≈{thr_mean_dbg:.4g}, "
                      f"Δ={mean_delta:.4g}, t={mean_time_const:.3f}")
            else:
                print(f"[recall_change_dt] {f}: p={p_feat:.3f}, (non-numeric/no thr), "
                      f"Δ={mean_delta:.4g}, t={mean_time_const:.3f}")

    # expected time under the per-feature selection distribution
    expected_time = sum(v["p_selected"] * v["mean_time"] for v in out.values())
    out["expected_time"] = float(expected_time)

    if verbose:
        print(f"[recall_change_dt] merged {len(chunks)} combo chunk(s), "
              f"retrieval_rt={retrieval_rt:.4f}, T_enc={T_enc:.4f}, expected_time={expected_time:.4f}")

    return out
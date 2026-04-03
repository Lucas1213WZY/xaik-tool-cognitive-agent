from string import digits
import numpy as np
from .lr_memory import round_to_sf, ddm_prob_rt_ratio
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
    import math, random, numpy as np

    rng = random.Random()
    x = np.asarray(feature_vector, float)
    nodes = {n["node"]: n for n in dt_exp.tree_structure}
    num_classes = len(next(n for n in nodes.values() if n["is_leaf"])["value"])
    UNIFORM = np.full(num_classes, 1.0 / num_classes, float)

    if verbose:
        print(f"[DT_TRAVERSE] mode={mode}, compute_sf={compute_sf}, "
              f"T_enc={T_enc}, ddm_a={ddm_a}, ddm_s={ddm_s}, n_mc={n_mc}")

    # ---------- helpers ----------
    def _round_sf(v): return float(round_to_sf(float(v), compute_sf))

    def _ddm_numeric_lte(val, thr):
        # evidence terms [thr, -val] => "left" if thr >= val
        return ddm_prob_rt_ratio([thr, -val], a=ddm_a, s=ddm_s, Tnd=ddm_Tnd, norm=ddm_norm)

    def _topk_no_refresh(request: dict):
        return memory.topk_retrievals_with_prob_refresh(
            request, k=topk_k, refresh_prob=0.0, add_refresh=False, verbose=False
        )

    # ---------- planning (retrieve mode only) ----------
    feature_topk = {}
    thrptr_topk  = {}
    num_profiles = {}

    if mode == "retrieve":
        for n in dt_exp.tree_structure:
            nid = n["node"]
            if n["is_leaf"]:
                continue
            # Feature retrieval distribution
            feature_topk[nid] = _topk_no_refresh({"type": "feature", "node": nid})
            # Threshold pointer + number profile only for numeric features
            if "=" not in n["feature"]:
                thrptr_topk[nid] = _topk_no_refresh({"type": "thr_ptr", "node": nid, "feat_key": n["feature"]})
                for ch, _ in thrptr_topk[nid]["top_k"]:
                    if ch is None:
                        continue
                    thr_key = ch.slots.get("thr_key")
                    if thr_key and thr_key not in num_profiles:
                        # build once; no refresh during planning
                        num_profiles[thr_key] = build_number_profile(
                            memory, key=thr_key, sf_req=compute_sf,
                            k=topk_k, refresh_prob=0.0, verbose=False
                        )

    if verbose and mode == "retrieve":
        print(f"[DT_TRAVERSE] Planning complete: feature_topk={feature_topk},\n "
              f"thrptr_topk={thrptr_topk},\n num_profiles={num_profiles}")
        
        print(f"  → Number profiles built: {list(num_profiles.keys())}")
    
    

    def _draw_topk_filtered(ret, rng_, used_chunks):
        """Filter out already-used chunks; include None with its p_none."""
        choices = []
        probs   = []
        for ch, p in ret["top_k"]:
            if ch is None or ch not in used_chunks:
                choices.append(ch)
                probs.append(float(p))
        # always include None as a possible outcome
        choices.append(None)
        probs.append(float(ret["p_none"]))
        tot = sum(probs)
        if tot <= 0:
            return None
        return rng_.choices(choices, weights=probs, k=1)[0]

    # ---------- Monte Carlo runs ----------
    probs_acc = np.zeros(num_classes, float)
    time_acc  = 0.0
    feature_sel_counts = {}
    thr_key_counts     = {}

    leaf_nodes_reached = []

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
                leaf_nodes_reached.append(node_id)
                break

            # --- FEATURE KEY ---
            if mode == "read":
                # literal feature; pay encoding once here
                run_time += T_enc
                feat_key = node["feature"]
                chF = None  # no chunk used
            else:
                # retrieval distribution for feature label (filter out used ones)
                retF = feature_topk[node_id]
                run_time += float(retF["expected_rt"])  # expected retrieval time
                chF = _draw_topk_filtered(retF, rng, used_feature_chunks)
                if chF is None:
                    # retrieval failure -> uniform mass over classes
                    probs_acc += UNIFORM
                    leaf_nodes_reached.append(f"feature_fail_node{node_id}")
                    break
                # record usage and book-keep counts
                used_feature_chunks.add(chF)
                feature_sel_counts[chF] = feature_sel_counts.get(chF, 0) + 1
                feat_key = chF.slots.get("feat_key", node["feature"])


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


            # --- THRESHOLD (numeric only) ---
            chP = None
            thr_val = None
            if "=" not in feat_key:
                if mode == "read":
                    # read literal numeric threshold; round to sf
                    run_time += T_enc
                    thr_val = _round_sf(nodes[node_id]["threshold"])
                else:
                    # retrieve threshold pointer then number profile (filter out used)
                    retP = thrptr_topk[node_id]
                    run_time += float(retP["expected_rt"])
                    chP = _draw_topk_filtered(retP, rng, used_thrptr_chunks)
                    if chP is None:
                        probs_acc += UNIFORM
                        leaf_nodes_reached.append(f"thrptr_fail_node{node_id}")
                        break
                    used_thrptr_chunks.add(chP)

                    thr_key = chP.slots.get("thr_key")
                    if not thr_key or thr_key not in num_profiles or thr_key in used_thr_keys:
                        probs_acc += UNIFORM
                        leaf_nodes_reached.append(f"thr_key_fail_node{node_id}")
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

                    pick_idx, chosen_meta = _pick_one(meta_opts, rng)
                    if chosen_meta is None or chosen_meta["value"] is None:
                        # failed to retrieve meta or got None -> fallback
                        probs_acc += UNIFORM
                        break

                    # record chosen meta chunk to prevent reuse in this path
                    if chosen_meta["chunk_name"] is not None:
                        used_num_meta_chunks.add(chosen_meta["chunk_name"])

                    sign, p10 = chosen_meta["value"]

                    # --- DIGIT sampling across positions (no reuse) ---
                    digits = []
                    for pos in range(1, compute_sf + 1):
                        opts = [o for o in prof["digits_with_chunks"][pos - 1]
                                if (o["chunk_name"] is None) or (o["chunk_name"] not in used_num_digit_chunks)]
                        if not opts:
                            # no available digit chunk at this position -> stop number build early
                            break

                        pick_idx, pick = _pick_one(opts, rng)
                        if pick is None:
                            # defensively stop; we'll use what we have so far
                            break

                        # mark digit chunk as used (if it exists)
                        if pick["chunk_name"] is not None:
                            used_num_digit_chunks.add(pick["chunk_name"])

                        # if retrieval yields None for this digit, stop building further digits
                        if pick["value"] is None:
                            break

                        digits.append(int(pick["value"]))

                    # build the numeric threshold value from whatever digits we managed
                    thr_val = digits_to_value(sign, p10, digits, len(digits)) if digits else 0.0
                    thr_val = _round_sf(thr_val)
                    run_time += float(prof["expected_rt"]) + T_enc  # number build + encode threshold

            # --- READ STIMULUS VALUE ---
            if "=" in feat_key:
                base, cat_idx = feat_key.split("=")
                att = int(base[1:])
                run_time += T_enc
                is_member = (int(x[att]) == int(cat_idx))
            else:
                att = int(feat_key[1:])
                run_time += T_enc
                val = float(x[att])

            # --- DECISION AT NODE ---
            if "=" in feat_key:
                # Categorical branch is PERFECT: equal -> left, else -> right
                go_left = bool(is_member)
                rt_dec = 0.0  # no decision time cost beyond encodes (or add tiny constant if you want)
            else:
                # Numeric: use DDM; randomness creates distribution in both modes
                p_up, E_RT, _, _ = _ddm_numeric_lte(val, thr_val)
                go_left = (rng.random() < p_up)  # left if val <= thr
                rt_dec = E_RT
            run_time += rt_dec

            node_id = node["left"] if go_left else node["right"]

        time_acc += run_time

    # ---------- Aggregate ----------
    probs = probs_acc / float(S_runs)
    expected_time = float(time_acc) / float(S_runs) if S_runs > 0 else 0.0

    if verbose:
        print(f"[DT_TRAVERSE] Finished {S_runs} MC runs")
        print(f"  → probs={probs}, expected_time={expected_time:.3f}s")
        print(f"  → leaf nodes reached: {leaf_nodes_reached}")

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


# import numpy as np
# from typing import Any, Dict, Tuple, Optional

# def cf_change_path_xai(
#     feature_vector: np.ndarray,
#     dt_exp: Any,
#     bounds: Dict[str, Tuple[float, float]],
#     *,
#     n_runs: int = 200,
#     value_display_sf: int = 2,
#     T_READ_NUM: float = 2.0,
#     rng: Optional[np.random.Generator] = None,
#     verbose: bool = False,
# ) -> Dict[str, Dict[str, float]]:
#     rng = rng or np.random.default_rng()
#     x = np.asarray(feature_vector, dtype=float)
#     k = len(x)

#     # --- Original path ---
#     full_path = _path_for_x(dt_exp, x, value_display_sf=value_display_sf)
#     dec_nodes = [n for n in full_path if not n["is_leaf"]]
#     if len(dec_nodes) == 0:
#         return {f"a{i}": {"p_selected": 0.0, "mean_delta": 0.0, "mean_time": 0.0} for i in range(k)}
#     orig_leaf = full_path[-1]
#     orig_class = int(np.argmax(orig_leaf["value"]))

#     # --- Recipes per node ---
#     node_info = []
#     for n in dec_nodes:
#         fk, delta_req, step, is_cat = _delta_to_cross(n, x, bounds, value_display_sf)
#         lo, hi = bounds.get(fk, (-np.inf, np.inf))
#         feasible = True
#         if not is_cat:
#             thr = n["threshold"]
#             if thr < lo or thr > hi:
#                 feasible = False
#         node_info.append((fk, delta_req, step, is_cat, feasible, n))

#     out = {f"a{i}": dict(p_selected=0.0, mean_delta=0.0, mean_time=0.0) for i in range(k)}
#     base_inspect_time = len(dec_nodes) * T_READ_NUM

#     # For quick node lookup
#     nodes = {n["node"]: n for n in dt_exp.tree_structure}

#     # --- Initial-choice weights: deepest node is 2x any other ---
#     m = len(dec_nodes)
#     weights = np.ones(m, dtype=float)
#     if m >= 1:
#         weights[-1] = 2.0  # final (deepest) node
#     weights /= weights.sum()

#     # --- Monte Carlo ---
#     for _ in range(n_runs):
#         idx = int(rng.choice(m, p=weights))
#         applied_delta, time_here, fk_used = 0.0, 0.0, None

#         while idx >= 0:
#             fk, delta_req, step, is_cat, feasible, node = node_info[idx]

#             if not feasible:
#                 time_here += T_READ_NUM
#                 idx -= 1
#                 continue

#             # Force the opposite branch at this node
#             col = None
#             x_prime = x.copy()
#             if is_cat:
#                 base_key, cat_idx = fk.split("=")
#                 col = int(base_key[1:])
#                 if int(x[col]) == int(cat_idx):
#                     x_prime[col] = -999  # force "not this category"
#                 else:
#                     x_prime[col] = int(cat_idx)
#                 applied_delta = 1.0
#                 time_here += T_READ_NUM
#             else:
#                 col = int(fk[1:])
#                 thr = node["threshold"]
#                 if x[col] <= thr:
#                     x_prime[col] = thr + 1e-6
#                 else:
#                     x_prime[col] = thr - 1e-6
#                 applied_delta = float(x_prime[col] - x[col])
#                 time_here += 2 * T_READ_NUM

#             # --- 50%: accept without checking; 50%: check & maybe escalate ---
#             if rng.random() < 0.5:
#                 fk_used = fk
#                 break
#             else:
#                 y_path = _path_for_x(dt_exp, x_prime, value_display_sf=value_display_sf)
#                 new_leaf = y_path[-1]
#                 new_class = int(np.argmax(new_leaf["value"]))
#                 time_here += len([n for n in y_path if not n["is_leaf"]]) * T_READ_NUM

#                 if new_class != orig_class or idx == 0:
#                     fk_used = fk
#                     break
#                 else:
#                     idx -= 1  # escalate upward

#         if fk_used is None:
#             fk_used, applied_delta = "a0", 0.0
#             time_here += T_READ_NUM

#         s = out[fk_used]
#         s["p_selected"] += 1.0
#         s["mean_delta"] += applied_delta
#         s["mean_time"]  += base_inspect_time + time_here

#     # --- Normalize ---
#     for fk, s in out.items():
#         used = s["p_selected"]
#         if used > 0:
#             s["mean_delta"] /= used
#             s["mean_time"]  /= used
#         s["p_selected"] /= float(n_runs)

#     # Renorm guard
#     total_p = sum(s["p_selected"] for s in out.values())
#     if not np.isclose(total_p, 1.0):
#         for fk in out:
#             s = out[fk]
#             s["p_selected"] /= total_p if total_p > 0 else 1.0

#     return out

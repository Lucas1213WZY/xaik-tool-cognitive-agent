import math
import random
from collections import deque

class Chunk:
    def __init__(self, name, slots):
        self.name = name
        self.slots = slots
        self.retrieval_times = []
        # NEW: list of probabilistic refreshes as (time, prob)
        # These are "possible boosts" that will be included in expectation.
        self.prob_refreshes = []  # [(t_star, p), ...]

    def add_prob_refresh(self, time, p):
        """Record a probabilistic refresh event that occurred at `time` with probability `p`."""
        if p <= 0.0:
            return
        # clamp tiny overs/unders
        p = max(0.0, min(1.0, float(p)))
        self.prob_refreshes.append((float(time), p))

    def update_retrieval(self, time):
        self.retrieval_times.append(time)

    def base_level_activation(self, current_time, decay=0.5):
        """
        Standard BLL plus MEAN-FIELD expected contribution of probabilistic refreshes:
        E[ sum_j (t - t_j)^-d  +  sum_m z_m * (t - t*_m)^-d ] 
        = sum_j (t - t_j)^-d + sum_m p_m * (t - t*_m)^-d
        """
        if not self.retrieval_times and not self.prob_refreshes:
            return float("-inf")

        # Certain (realized) retrievals strictly before current_time
        certain = [t for t in self.retrieval_times if t < current_time]
        # Probabilistic refreshes strictly before current_time
        prob = [(t_star, p) for (t_star, p) in self.prob_refreshes if t_star < current_time]

        if not certain and not prob:
            return float("-inf")

        eps = 1e-10
        s_certain = sum((current_time - t + eps) ** -decay for t in certain)

        # Mean-field expected contribution from probabilistic refreshes
        s_prob = sum(p * (current_time - t_star + eps) ** -decay for (t_star, p) in prob)

        total = s_certain + s_prob
        if total <= 0.0:
            return float("-inf")

        return math.log(total)

    def similarity_to(self, request, mismatch_penalty=-1.0, must_match=('type','kind')):
        for k in must_match:
            if k in request and self.slots.get(k) != request[k]:
                return float("-inf")
        sim = 0.0
        for key, val in request.items():
            if key not in self.slots or self.slots[key] != val:
                sim += mismatch_penalty
        return sim

    def activation(self, request, current_time, decay=0.5, mismatch_penalty=-1.0, 
                   assoc_strength=0.0, max_assoc_strength=2.0, fans=None, verbose=False):
        base = self.base_level_activation(current_time, decay)
        sim = self.similarity_to(request, mismatch_penalty)

        if fans and request:
            num_cues = len(request)
            w = 1.0 / num_cues if num_cues > 0 else 0.0
            for key, val in request.items():
                if key in self.slots and self.slots[key] == val:
                    fan = fans.get(val, 1)
                    if fan > 0:
                        s_ji = max_assoc_strength - math.log(fan + 1e-10)
                        assoc_strength += w * s_ji

        assoc_strength = max(0, assoc_strength)
        return base + sim + assoc_strength

    def __repr__(self):
        return (f"{self.name}: {self.slots}\n")
                # f"Refreshed times: {self.retrieval_times}\n"
                # f"Prob. refreshes: {self.prob_refreshes}")


# =====================
# DeclarativeMemory (unchanged except helper methods below)
# =====================

class DeclarativeMemory:
    def __init__(self, decay=0.5, mismatch_penalty=-1.0, retrieval_threshold=0.0,
                 max_assoc_strength=2.0, activation_noise=0.0, latency_factor=0.1, latency_exponent=1.0):
        self.chunks = []
        self.time = 0
        self.decay = decay
        self.mismatch_penalty = mismatch_penalty
        self.retrieval_threshold = retrieval_threshold
        self.max_assoc_strength = max_assoc_strength
        self.activation_noise = activation_noise  # "s" scale for logistic noise
        self.latency_factor = latency_factor
        self.latency_exponent = latency_exponent

    def add_chunk(self, name, slots, update_retrieval=True):
        chunk = Chunk(name, slots)
        self.chunks.append(chunk)
        if update_retrieval:
            chunk.update_retrieval(self.time)
        return chunk

    def tick(self, dt=1):
        self.time += dt

    def retrieve(self, request, verbose=False):
        best_chunk = None
        best_activation = float("-inf")
        
        request_values = set(request.values())
        fans = {}
        for val in request_values:
            fans[val] = sum(1 for chunk in self.chunks if any(slot_val == val for slot_val in chunk.slots.values()))
        
        for chunk in self.chunks:
            act = chunk.activation(request, self.time, self.decay, self.mismatch_penalty,
                                   assoc_strength=1.0, max_assoc_strength=self.max_assoc_strength, fans=fans, verbose=verbose)

            if self.activation_noise > 0:
                eta = random.random()
                if eta == 0 or eta == 1:
                    eta = 1e-10 if eta == 0 else 1 - 1e-10
                noise = self.activation_noise * math.log((1 - eta) / eta)
                act += noise
            
            if act >= self.retrieval_threshold and act > best_activation:
                best_activation = act
                best_chunk = chunk

        activation_for_latency = best_activation if best_chunk else self.retrieval_threshold
        retrieval_time = self.latency_factor * math.exp(-self.latency_exponent * activation_for_latency)

        if best_chunk:
            best_chunk.update_retrieval(self.time)

        return best_chunk, best_activation, retrieval_time

    def __repr__(self):
        return "\n".join(f"{chunk} | Activation: {chunk.activation({}, self.time, self.decay, self.mismatch_penalty):.2f}" for chunk in self.chunks)
    
    def refresh(self, chunk_name):
        for chunk in self.chunks:
            if chunk.name == chunk_name:
                chunk.update_retrieval(self.time)
                return chunk
        return None

    def get_chunk(self, chunk_name):
        for chunk in self.chunks:
            if chunk.name == chunk_name:
                return chunk
        return None

    # === existing helpers ===
    def _all_activations_for_request(self, request: dict):
        request_values = set(request.values())
        fans = {}
        for val in request_values:
            fans[val] = sum(
                1 for ch in self.chunks if any(slot_val == val for slot_val in ch.slots.values())
            )

        out = []
        for ch in self.chunks:
            act = ch.activation(
                request=request,
                current_time=self.time,
                decay=self.decay,
                mismatch_penalty=self.mismatch_penalty,
                assoc_strength=1.0,
                max_assoc_strength=self.max_assoc_strength,
                fans=fans,
                verbose=False
            )
            out.append((ch, act))
        out.sort(key=lambda t: t[1], reverse=True)
        return out

    def retrieval_success_prob(self, request: dict, verbose: bool = False) -> float:
        acts = self._all_activations_for_request(request)
        top_act = acts[0][1] if acts else float("-inf")

        theta = float(self.retrieval_threshold)
        s = float(self.activation_noise)

        if s <= 0.0:
            p = 1.0 if top_act >= theta else 0.0
            if verbose:
                print(f"[retrieval_success_prob] top_act={top_act:.3f}, "
                      f"theta={theta:.3f}, s={s:.3f} → p={p:.3f}")
            return p

        z = (top_act - theta) / s
        p = 1.0 / (1.0 + math.exp(-z))
        if verbose:
            print(f"[retrieval_success_prob] top_act={top_act:.3f}, "
                  f"theta={theta:.3f}, s={s:.3f}, z={(top_act - theta):.3f}/{s:.3f}={z:.3f} "
                  f"→ sigmoid(z)={p:.3f}")
        return p


# =====================
# WM and CombinedMemory (unchanged logic; NEW method added below)
# =====================

def _matches_request_by_slots(chunk, request: dict) -> bool:
    s = getattr(chunk, "slots", {}) or {}
    for k, v in request.items():
        if s.get(k) != v:
            return False
    return True

class WorkingMemoryQueue:
    def __init__(self, capacity=4):
        self.capacity = capacity
        self._dq = deque()

    def get_by_slots(self, request: dict):
        for chunk in reversed(self._dq):
            if _matches_request_by_slots(chunk, request):
                return chunk
        return None

    def put_chunk(self, chunk):
        name = getattr(chunk, "name", None)
        self._dq = deque([c for c in self._dq if getattr(c, "name", None) != name])
        self._dq.append(chunk)
        while len(self._dq) > self.capacity:
            self._dq.popleft()

    def clear(self):
        self._dq.clear()


class CombinedMemory:
    def __init__(self, declarative_memory, wm_capacity=4):
        self.dm = declarative_memory
        self.wm = WorkingMemoryQueue(capacity=wm_capacity)

    def add_chunk(self, name, slots, *, update_retrieval=True):
        return self.dm.add_chunk(name, slots, update_retrieval=update_retrieval)

    def tick(self, dt=1):
        self.dm.tick(dt)

    def retrieve(self, request: dict, verbose: bool=False, allow_dm: bool=True):
        chunk = self.wm.get_by_slots(request)
        if chunk is not None:
            if verbose:
                print(f"WM HIT {tuple(sorted(request.items()))} -> {chunk.slots} (rt=0.00)")
            return chunk, None, 0.0

        if not allow_dm:
            if verbose:
                print(f"WM MISS {tuple(sorted(request.items()))} (no DM fallback)")
            return None, None, 0.0

        chunk, activation, rt = self.dm.retrieve(request, verbose=verbose)
        self.dm.tick(rt)

        if chunk is not None and _matches_request_by_slots(chunk, request):
            self.wm.put_chunk(chunk)
        else:
            if verbose and chunk is not None:
                print(f"DM RETURN MISMATCH for {tuple(sorted(request.items()))} "
                      f"-> {chunk.slots}; not caching")

        return chunk, activation, rt

    def refresh(self, chunk_name):
        return self.dm.refresh(chunk_name)

    def get_chunk(self, chunk_name):
        return self.dm.get_chunk(chunk_name)

    @property
    def chunks(self):
        return self.dm.chunks

    def __repr__(self):
        return repr(self.dm)

    def retrieval_success_prob(self, request: dict, verbose: bool = False) -> float:
        if self.wm.get_by_slots(request) is not None:
            return 1.0
        return float(self.dm.retrieval_success_prob(request, verbose=verbose))

    # ============================================================
    # NEW: Top-k retrieval distribution + expected RT + prob. refresh logging
    # ============================================================

    def topk_retrievals_with_prob_refresh(
        self,
        request: dict,
        k: int = 3,
        refresh_prob: float = 1.0,
        add_refresh: bool = True,
        verbose: bool = False
    ):
        """
        Compute a retrieval *distribution* over chunks (softmax with 'none'),
        return top-k (chunk, probability) plus p_none and expected RT,
        and (optionally) record probabilistic refreshes proportional to p_i.

        Softmax with 'none':
          logits_i = (A_i - theta) / s
          p_i = exp(logits_i) / (1 + sum_j exp(logits_j))
          p_none = 1 / (1 + sum_j exp(logits_j))
        If s<=0: deterministic fallback against threshold.

        Expected RT:
          E[RT] = sum_i p_i * F * exp(-f * A_i) + p_none * F * exp(-f * theta)

        If add_refresh:
          For each chunk i, add a probabilistic refresh at current dm.time with prob = refresh_prob * p_i.
          (No WM updates or time advance here; this is a planning/estimation pass.)
        """
        # 0) WM exact match? Then it's trivially that chunk with prob=1, rt=0
        wm_chunk = self.wm.get_by_slots(request)
        if wm_chunk is not None:
            if add_refresh and refresh_prob > 0.0:
                wm_chunk.add_prob_refresh(self.dm.time, refresh_prob * 1.0)
            return {
                "top_k": [(wm_chunk, 1.0)],
                "p_none": 0.0,
                "expected_rt": 0.0
            }

        # 1) Gather deterministic activations (no noise) for all chunks
        acts = self.dm._all_activations_for_request(request)  # [(chunk, A)]
        theta = float(self.dm.retrieval_threshold)
        s = float(self.dm.activation_noise)
        F = float(self.dm.latency_factor)
        fexp = float(self.dm.latency_exponent)

        if not acts:
            # Nothing in memory matches; only 'none' is possible
            return {"top_k": [], "p_none": 1.0, "expected_rt": F * math.exp(-fexp * theta)}

        if s <= 0.0:
            # Deterministic: pick the best above threshold if any
            best_chunk, best_A = acts[0]
            if best_A >= theta:
                # deterministic hit
                p_map = {best_chunk: 1.0}
                p_none = 0.0
                expected_rt = F * math.exp(-fexp * best_A)
                if add_refresh and refresh_prob > 0.0:
                    best_chunk.add_prob_refresh(self.dm.time, refresh_prob)
                top_k = [(best_chunk, 1.0)]
                return {"top_k": top_k, "p_none": p_none, "expected_rt": expected_rt}
            else:
                # deterministic failure
                return {"top_k": [], "p_none": 1.0, "expected_rt": F * math.exp(-fexp * theta)}

        # remove chunks with less than -100 activation (very unlikely)
        acts = [(ch, A) for (ch, A) in acts if A >= theta - 100 * s]

        # 2) Softmax with a "none" option at activation = theta
        logits = []
        for ch, A in acts:
            z = (A - theta) / s
            logits.append((ch, A, z))

        max_z = max(z for _, _, z in logits) if logits else 0.0
        # stabilize
        exp_z = [(ch, A, math.exp(z - max_z)) for (ch, A, z) in logits]
        sum_exp = sum(ez for _, _, ez in exp_z)

        # p_none uses the same stabilization: exp(0 - max_z) = exp(-max_z)
        p_none = math.exp(-max_z) / (math.exp(-max_z) + sum_exp)

        # chunk probabilities
        probs = []
        denom = (math.exp(-max_z) + sum_exp)
        for ch, A, ez in exp_z:
            p = ez / denom
            probs.append((ch, A, p))

        # 3) Expected RT
        expected_rt = p_none * F * math.exp(-fexp * theta)
        if math.isnan(expected_rt) or math.isinf(expected_rt):
            print(f"[DEBUG] bad expected_rt init: p_none={p_none}, F={F}, fexp={fexp}, theta={theta}")

        for ch, A, p in probs:
            term = p * F * math.exp(-fexp * A)
            if math.isnan(term) or math.isinf(term):
                print(f"[DEBUG] bad RT term: A={A}, p={p}, F={F}, fexp={fexp}")
            expected_rt += term

        # 4) Add probabilistic refreshes proportional to retrieval probability
        if add_refresh and refresh_prob > 0.0:
            for ch, _, p in probs:
                pr = refresh_prob * p
                if pr > 0.0:
                    ch.add_prob_refresh(self.dm.time, pr)

        # 5) Return top-k most likely chunks
        probs.sort(key=lambda t: t[2], reverse=True)  # by p desc
        top_k = [(ch, p) for (ch, _, p) in probs[:max(0, int(k))]]

        total_p = p_none + sum(p for _, p in top_k) or 1.0
        top_k = [(ch, p / total_p) for ch, p in top_k]
        p_none = p_none / total_p

        if verbose:
            print(f"[topk] p_none={p_none:.3f}, expected_rt={expected_rt:.4f}")
            for ch, p in top_k:
                print(f"  {ch.name}: p={p:.3f}")

        return {"top_k": top_k, "p_none": p_none, "expected_rt": expected_rt}
    
import math
import numpy as np

# ------------------------------
# Utilities
# ------------------------------

def breakdown_number_to_sf(value: float, max_sf: int):
    """Return (sign, scale10, digits[1..max_sf]) for scientific notation:
       value = sign * (d1.d2d3...)*10**scale10, with d1!=0 unless value==0.
    """
    if value == 0 or not math.isfinite(value):
        return 1, 0, [0]*max_sf
    sign = -1 if value < 0 else 1
    ax = abs(value)
    p = math.floor(math.log10(ax))
    mant = ax / (10**p)  # in [1,10)
    s = f"{mant:.{max_sf-1}f}".replace(".", "")
    if len(s) > max_sf:
        s = s[:max_sf]
        p += 1
    digits = [int(c) for c in s[:max_sf]]
    return sign, p, digits

def digits_to_value(sign:int, p:int, digits:list[int], sf_req:int) -> float:
    """Reconstruct numeric value from the first sf_req digits."""
    if sf_req <= 0:
        return 0.0
    s = "".join(str(d) for d in digits[:sf_req])
    mant = int(s) / (10**(sf_req-1))  # e.g., 314 -> 3.14 for sf=3
    return sign * mant * (10**p)

# ------------------------------
# remember_number_to_sf
# ------------------------------

def remember_number_to_sf(memory, key: str, value: float, max_sf: int):
    """
    Store number in memory:
      - META chunk (sign + scale)
      - DIGIT chunks
    """
    sign, scale10, digits = breakdown_number_to_sf(value, max_sf)
    created = []

    # META chunk
    memory.add_chunk(f"num:{key}:meta",
                     {"kind": "nummeta", "key": key, "sign": sign, "p": scale10})
    created.append(f"num:{key}:meta")

    # DIGITS
    for pos, d in enumerate(digits, start=1):
        name = f"num:{key}:d{pos}"
        memory.add_chunk(name,
                         {"kind": "digit", "key": key, "pos": pos, "digit": d})
        created.append(name)

    return created

# ------------------------------
# ACT-R parameters + latency expectation
# ------------------------------

# def _dm_params(mem):
#     dm = getattr(mem, "dm", mem)
#     F     = float(getattr(dm, "latency_factor",      1.0))
#     expo  = float(getattr(dm, "latency_exponent",    1.0))
#     theta = float(getattr(dm, "retrieval_threshold", 0.0))
#     s     = float(getattr(dm, "activation_noise",    0.0))
#     return F, expo, theta, s

# def _expected_rt_from_phit(p_hit, F, expo, theta, s):
#     """Expected latency given probability of hit p_hit."""
#     p = min(max(float(p_hit), 1e-9), 1.0 - 1e-9)
#     if s > 0.0:
#         A_hat = theta + s * math.log(p / (1.0 - p))
#     else:
#         A_hat = float("inf") if p >= 1.0 else float("-inf") if p <= 0.0 else theta
#     rt_hit  = F * math.exp(-expo * (A_hat if math.isfinite(A_hat) else theta))
#     rt_null = F * math.exp(-expo * theta)
#     return p * rt_hit + (1.0 - p) * rt_null

# ------------------------------
# retrieve_number_to_sf (stub → calls build_number_profile)
# ------------------------------

def retrieve_number_to_sf(memory, key: str, sf_req: int, verbose: bool=False):
    """
    Stub: delegates to build_number_profile with probabilistic chain.
    Returns (val, sf_got, chunks, activs, total_rt) for compatibility.
    """
    val, sf_got, p_succ, total_rt, aux = build_number_profile(memory, key, sf_req, verbose=verbose)
    return val, sf_got, [], [], total_rt

# ------------------------------
# build_number_profile → probabilistic chain version
# ------------------------------
def build_number_profile(
    memory, key: str, sf_req: int, *,
    k: int = 3,
    refresh_prob: float = 1.0,
    verbose: bool = False
):
    """
    Return distributions for meta and each digit.
    Each distribution = [(value, prob), ...] including (None, p_none).
    Also computes the expected retrieval time of the full chain.

    Additionally returns chunk-name aware payloads:
      - meta_with_chunks:   [{"value": (sign,p10)|None, "prob": p, "chunk_name": str|None}, ...]
      - digits_with_chunks: [[{"value": d|None,        "prob": p, "chunk_name": str|None}, ...], ...]
    """
    F = float(memory.dm.latency_factor)
    fexp = float(memory.dm.latency_exponent)
    theta = float(memory.dm.retrieval_threshold)

    if verbose:
        print(f"[build_number_profile] key={key}, sf_req={sf_req}, k={k}")
        print(f"  DM params: F={F}, fexp={fexp}, theta={theta}")

    # --- META ---
    meta_dist = memory.topk_retrievals_with_prob_refresh(
        {"kind": "nummeta", "key": key}, k=k,
        refresh_prob=0.0, add_refresh=False, verbose=verbose
    )
    meta_options = []          # legacy: list of (value, prob)
    meta_with_chunks = []      # new: list of {value, prob, chunk_name}

    p_none_meta = float(meta_dist["p_none"])
    p_meta = 1.0 - p_none_meta
    rt_meta = float(meta_dist["expected_rt"])

    if verbose:
        print(f"  META: p_none={p_none_meta}, p_meta={p_meta}, rt_meta={rt_meta}")

    if p_none_meta > 0:
        meta_options.append((None, p_none_meta))
        meta_with_chunks.append({"value": None, "prob": p_none_meta, "chunk_name": None})

    for ch, p in meta_dist["top_k"]:
        sign = int(ch.slots.get("sign", 1))
        p10  = int(ch.slots.get("p", 0))
        chunk_name = getattr(ch, "name", None)
        meta_options.append(((sign, p10), float(p)))
        meta_with_chunks.append({"value": (sign, p10), "prob": float(p), "chunk_name": chunk_name})
        if refresh_prob > 0:
            ch.add_prob_refresh(memory.dm.time, refresh_prob * float(p))

    # --- DIGITS ---
    digit_options_all = []     # legacy: list per pos of [(value, prob)]
    digits_with_chunks = []    # new: list per pos of [{value, prob, chunk_name}...]

    C_prev = p_meta
    expected_rt_chain = rt_meta
    for pos in range(1, sf_req + 1):
        d_dist = memory.topk_retrievals_with_prob_refresh(
            {"kind": "digit", "key": key, "pos": pos}, k=k,
            refresh_prob=0.0, add_refresh=False, verbose=verbose
        )
        opts_legacy = []
        opts_with_chunks = []

        p_none = float(d_dist["p_none"])
        p_hit  = 1.0 - p_none
        rt_d   = float(d_dist["expected_rt"])

        if verbose:
            print(f"  DIGIT {pos}: p_none={p_none}, p_hit={p_hit}, rt_d={rt_d}, C_prev={C_prev}")

        if p_none > 0:
            opts_legacy.append((None, p_none))
            opts_with_chunks.append({"value": None, "prob": p_none, "chunk_name": None})

        for ch, p in d_dist["top_k"]:
            dval = int(ch.slots.get("digit", 0))
            chunk_name = getattr(ch, "name", None)
            opts_legacy.append((dval, float(p)))
            opts_with_chunks.append({"value": dval, "prob": float(p), "chunk_name": chunk_name})
            if refresh_prob > 0 and C_prev > 0:
                ch.add_prob_refresh(memory.dm.time, refresh_prob * C_prev * float(p))

        digit_options_all.append(opts_legacy)
        digits_with_chunks.append(opts_with_chunks)

        # Add expected RT contribution scaled by chain prefix
        contrib = C_prev * rt_d
        expected_rt_chain += contrib
        if verbose:
            print(f"    contrib={contrib}, expected_rt_chain={expected_rt_chain}")
        C_prev *= p_hit

    # Final check for NaN/Inf
    if verbose and (expected_rt_chain != expected_rt_chain or  # NaN
                    expected_rt_chain in (float("inf"), float("-inf"))):
        print(f"  WARNING: expected_rt_chain is abnormal! {expected_rt_chain}")
        print(f"  key={key}, meta_dist={meta_dist}")

    return {
        # legacy keys (unchanged)
        "meta": meta_options,
        "digits": digit_options_all,
        "expected_rt": expected_rt_chain,

        # new chunk-aware payloads
        "meta_with_chunks": meta_with_chunks,
        "digits_with_chunks": digits_with_chunks,
    }

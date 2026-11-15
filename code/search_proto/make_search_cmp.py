#!/usr/bin/env python3
"""Build cnet2_search_cmp.py from the pinned vendor base cnet2_search.py.

The transformation is a small set of anchored, exact-match edits; every
anchor must match exactly once or the build stops. Line numbers cited in
the successor header are BASE-source line numbers (the vendor copy minus
its 3-line pack_bridge header; that base text's md5 must equal the
pinned code md5 94f7fb64...).
"""
import difflib
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.expanduser("~/workspace/search-group/proto/vendor/cnet2_search.py")
OUT = os.path.join(HERE, "cnet2_search_cmp.py")
PIN_CODE_MD5 = "94f7fb6438bb7577b141e388f068e122"

base_lines = open(VENDOR, "rb").read().splitlines(keepends=True)[3:]
base = b"".join(base_lines).decode("utf-8")
got = hashlib.md5(base.encode("utf-8")).hexdigest()
assert got == PIN_CODE_MD5, f"base code md5 mismatch: {got} != {PIN_CODE_MD5}"
lines = base.splitlines(keepends=True)

def lineno(anchor, start=0):
    hits = [i for i in range(start, len(lines)) if anchor in lines[i]]
    assert len(hits) == 1, f"anchor {anchor!r}: {len(hits)} hits"
    return hits[0] + 1  # 1-based

# --- locate the judgment segment in base coordinates -------------------
seg_start = lineno("        models = [acts_o]")            # first replaced line
inner_i = lineno("    _inner = agent")
# last non-empty line before _inner is the segment's final `return acts`
seg_end = inner_i - 1
while lines[seg_end - 1].strip() == "":
    seg_end -= 1
assert lines[seg_end - 1].strip() == "return acts", lines[seg_end - 1]
print(f"base segment to replace: lines {seg_start}-{seg_end}")

SEGMENT = '''        # ---- comparator segment (successor: replaces the base batch
        # branch advance + budget gates + MIN_DEPTH/MARGIN judgment) ----
        _pre_cmp = clk() - t0   # SI02 phase: snapshot/baseline/opponent
                                # inference/candidate generation, charged
                                # to the same whole-call envelope
        # SI01 entry: reject already-terminal roots. The authoritative
        # terminal is extinction OR the time boundary (last valid action
        # observation step 298, terminal state step 299).
        while len(pool) < 1:
            pool.append(FastCarbon())
        sim = pool[0]
        sim.load_observation(full)
        if left <= 0 or _terminal(sim) or sim.step_count >= 299:
            stats["cmp_term_root"] = stats.get("cmp_term_root", 0) + 1
            return acts
        # Admission screens (WIRING_MAP 7.4): (1) root legality was
        # enforced at generation time above (legal mask + seen dedup);
        # (2) guard consistency is screened HERE, per candidate, on a
        # guard.copy() -- the base chain only guarantees A0 passed the
        # guard (in _inner); a deviating slot was never re-checked. A
        # candidate the guard copy rewrites (or whose re-check raises, so
        # consistency cannot be proven) is vetoed; A0 is exempt by
        # contract. (3) tactical relevance: K <= 4 -- A0 plus the first
        # <= 3 deviations in generation order (key_units stake-desc).
        keep = list(range(min(len(cand), 4)))
        lg0 = st.get("lg0")
        vetoed = set()
        _gs_t = clk()
        if lg0 is not None and cfg["GUARD"]:
            for i in keep[1:]:
                try:
                    out = guard.copy().apply(x, f.aux, cand_y[i].copy(), lg0,
                                             pol, f.own_uid, st.get("T_now", T))
                except Exception:
                    stats["cmp_guard_err"] = stats.get("cmp_guard_err", 0) + 1
                    vetoed.add(i)
                    continue
                if not np.array_equal(np.asarray(out), cand_y[i]):
                    vetoed.add(i)
        _guard_s = clk() - _gs_t
        # SI02 budget: ONE whole-call envelope from _inner entry (t0).
        # Every phase above has already spent from it; the comparator is
        # bounded by an absolute deadline at t0 + BUDGET. There is NO
        # top-up: an exhausted envelope returns A0 without a comparison.
        rem = cfg["BUDGET"] - (clk() - t0)
        if rem <= 0:
            _tot = clk() - t0
            cmp_stats.record_decision({
                "completed": False, "rounds": 0, "changed": False,
                "depth": None, "leaf_calls": 0, "cost_s": _tot,
                "abort_reason": "deadline",
                "phases": {"pre_cmp": _pre_cmp, "guard_screen": _guard_s,
                           "compare": 0.0, "total": _tot},
            })
            stats["cmp_deadline"] = stats.get("cmp_deadline", 0) + 1
            return acts
        horizon = max(1, min(int(cfg.get("CMP_H", 1)), left))
        # Serial rollout on ONE pooled sim: the comparator consumes each
        # leaf_state (via the leaf call) before the next rollout starts,
        # so reuse is safe. scenario_seed is accepted but not consumed:
        # this layer is J=1 deterministic (opponent root action fixed
        # above, both seats then play greedily) -- the sweep path has no
        # seed axis and none is invented here.
        tch = [st.get("T_now", T), st["ot"]]
        _cur = {"idx": None, "vals": {}}
        _by_id = {id(a): i for i, a in enumerate(cand)}

        def _cmp_rollout(state, action, horizon_, scenario_seed):
            _cur["idx"] = _by_id[id(action)]
            s_ = sim
            s_.load_observation(state)
            s_._tid_counter = [c + 500 for c in s_._tid_counter]
            fm, fop = feat_copy(f), feat_copy(fo)
            # F01: cash carried into the root step; used only if that
            # step ends the game by extinction. A pure time terminal
            # (step 299) is scored on the real terminal state's cash,
            # signalled to the leaf by tcash=None.
            tcash = list(s_.cash)
            if me == 0:
                s_.step(action, acts_o)
            else:
                s_.step(acts_o, action)
            depth = 1
            # SI01: after every joint advance, extinction and the time
            # boundary are judged together; a terminal state is frozen
            # and no further policy call is made on it.
            if _terminal(s_):
                term = True
            elif s_.step_count >= 299:
                term = True
                tcash = None
            else:
                term = False
            hh = 1
            while hh < horizon_ and not term:
                pc_ = list(s_.cash)   # F01: cash carried into this step
                xs = [fm.observe(s_.obs(me)), fop.observe(s_.obs(1 - me))]
                lab = act_b(rpol, xs, [fm.aux, fop.aux], tch)
                am, ao = fm.to_actions(lab[0]), fop.to_actions(lab[1])
                if me == 0:
                    s_.step(am, ao)
                else:
                    s_.step(ao, am)
                depth += 1
                hh += 1
                if _terminal(s_):
                    term = True
                    tcash = pc_       # extinction: F01 pre-step cash
                elif s_.step_count >= 299:
                    term = True
                    tcash = None      # time terminal: real final cash
            return (s_, depth, tcash)

        _adapter = HeuristicLeafAdapter(
            lambda ls: float(leaf(ls[0], me, max(0, left - ls[1]), ls[2])),
            a=1.0, b=0.0)   # raw-margin pass-through: comparisons are
                            # within this one leaf; argmax is invariant
                            # to monotone rescaling (registered caliber)

        def _cmp_leaf(leaf_state):
            v = _adapter(leaf_state)
            _cur["vals"][_cur["idx"]] = v
            return v

        comp = Comparator(budget_s=rem, max_candidates=4)
        cands = [Candidate(index=i, action=cand[i], is_a0=(i == 0)) for i in keep]
        _cmp_t = clk()
        dec = comp.decide(full, cands, _cmp_rollout, _cmp_leaf,
                          horizon=horizon, scenario_seed=int(step),
                          guard_vetoed=vetoed,
                          deadline=time.monotonic() + rem)
        _compare_s = clk() - _cmp_t
        _tot = clk() - t0
        cmp_stats.record_decision({
            "completed": dec.completed_rounds > 0,
            "rounds": dec.completed_rounds,
            "changed": bool(dec.changed),
            "depth": dec.deepest_completed_h if dec.completed_rounds > 0 else None,
            "leaf_calls": _adapter.calls,
            "cost_s": _tot,
            "abort_reason": dec.abort_reason,
            "phases": {"pre_cmp": _pre_cmp, "guard_screen": _guard_s,
                       "compare": _compare_s, "total": _tot},
        })
        stats["branches"] += len(keep)
        stats["depth_sum"] += dec.deepest_completed_h
        if cfg["LOG"]:
            log.append((step, [cmeta[i] for i in keep], dec.chosen_index,
                        dec.completed_rounds, dec.abort_reason))
        if dec.completed_rounds > 0 and dec.changed:
            stats["changed"] += 1
            v0 = _cur["vals"].get(0)
            vj = _cur["vals"].get(dec.chosen_index)
            if v0 is not None and vj is not None:
                stats["gain"] += float(vj - v0)
            return dec.action
        return acts
'''

HEADER = f'''# =====================================================================
# SUCCESSOR SOURCE -- cnet2_search_cmp: comparator successor of
# cnet2_search. Base source: code/cnet2_search.py, code md5 (build-hash
# caliber) = {PIN_CODE_MD5}. Everything not listed below is
# byte-identical to the base.
# Changes vs base (line numbers are BASE-source lines):
#   1. This header block (inserted after the module docstring).
#   2. Import of the comparator layer (cnet2_comparator) after the
#      fastsim import.
#   3. DEFAULTS gains CMP_H=1 (comparator horizon: total depth
#      including the root step).
#   4. make_search_agent creates cmp_stats = StatsCollector() next to
#      pool; the outer agent publishes it as agent.cmp_stats.
#   5. _search candidate generation additionally tracks cand_y (the
#      candidates' label vectors) for the guard-consistency screen.
#   6. _search judgment segment replaced: base lines {seg_start}-{seg_end}
#      (batch branch advance, projection/0.45 budget gates, MIN_DEPTH
#      acceptance gate, leaf argmax with MARGIN, ROBUST hunter second
#      model) -> serial complete-round Comparator.decide (see the
#      segment comment). MIN_DEPTH / 0.45 first-layer gate / MARGIN
#      are NOT carried over; the layer is J=1 deterministic.
#      SI01 caliber inside the segment: already-terminal roots are
#      rejected at entry; after every joint advance, extinction and
#      the step-299 time boundary are judged together and a terminal
#      state is frozen with zero extra policy calls (F01 pre-step
#      cash on extinction; real terminal cash on a pure time-out).
#      SI02 caliber: one whole-call envelope from _inner entry --
#      pre-comparator phases and the guard screen spend from it, the
#      comparator is bounded by an absolute deadline at t0 + BUDGET,
#      an exhausted envelope returns A0 with NO top-up, and telemetry
#      records total plus per-phase time.
#   7. stats parity notes: "branches" now counts compared candidates;
#      "depth_sum" accumulates the completed round's depth (0 on
#      abort); "gain" accumulates leaf-value differences on change
#      (raw-margin scale); _search no longer updates "shallow" /
#      "term" / "branch_steps".
# =====================================================================
'''

def sub_once(text, old, new):
    assert text.count(old) == 1, f"anchor not unique ({text.count(old)}): {old[:80]!r}"
    return text.replace(old, new)

text = base

# (2) comparator import after the fastsim import
text = sub_once(
    text,
    "from fastsim import FastCarbon  # noqa: E402\n",
    "from fastsim import FastCarbon  # noqa: E402\n"
    "from cnet2_comparator import Candidate, Comparator, HeuristicLeafAdapter, StatsCollector  # noqa: E402\n")

# (3) DEFAULTS gains CMP_H
text = sub_once(text, "    H=5, KEY_R=2,", "    H=5, CMP_H=1, KEY_R=2,")

# (4) cmp_stats creation next to pool
text = sub_once(
    text,
    "    pool = []\n",
    "    pool = []\n"
    "    cmp_stats = StatsCollector()   # comparator telemetry; published as agent.cmp_stats\n")

# (5) cand_y tracking in _search candidate generation
text = sub_once(
    text,
    "        cand = [acts]; seen = {tuple(y)}; cmeta = [None]\n",
    "        cand = [acts]; seen = {tuple(y)}; cmeta = [None]; cand_y = [y]   # cand_y: label vectors, for the guard screen\n")
text = sub_once(
    text,
    "            seen.add(k); cand.append(f.to_actions(yy)); cmeta.append(mm)\n",
    "            seen.add(k); cand.append(f.to_actions(yy)); cmeta.append(mm); cand_y.append(yy)\n")

# (6) replace the judgment segment (recompute anchors on current text)
cur = text.splitlines(keepends=True)
s = next(i for i, l in enumerate(cur) if "        models = [acts_o]" in l)
e = next(i for i, l in enumerate(cur) if l.rstrip() == "    _inner = agent")
j = e - 1
while cur[j].strip() == "":
    j -= 1
assert cur[j].rstrip() == "        return acts", cur[j]
text = "".join(cur[:s]) + SEGMENT + "\n" + "".join(cur[e:])

# (7) publish cmp_stats on the outer agent
text = sub_once(
    text,
    "    agent.cfg = cfg\n    return agent",
    "    agent.cfg = cfg\n"
    "    agent.cmp_stats = cmp_stats   # comparator telemetry channel (snapshot() per game)\n"
    "    return agent")

# (1) header block after the module docstring (first closing ''')
end = text.index('"""', text.index('"""') + 3) + 3
text = text[:end] + "\n" + HEADER + text[end:]

open(OUT, "w").write(text)

# --- audit: diff stats base -> successor --------------------------------
diff = list(difflib.unified_diff(base.splitlines(), text.splitlines(), lineterm=""))
added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
print(f"wrote {OUT}: {len(text)} chars, md5 {hashlib.md5(text.encode()).hexdigest()}")
print(f"diff vs base: +{added} -{removed} lines")

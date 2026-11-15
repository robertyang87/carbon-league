"""cnet2_comparator -- shallow tactical-search comparator for the cnet2 policy.

Production module for the comparator search layer (successor to the
cnet2_search sweep/MCTS judgment segment). It bundles three pieces:

  Comparator / Candidate / Decision
      The complete-round candidate comparator. K <= 4 candidates, always
      including A0 (the original policy action, post-guard). All eligible
      candidates are compared in ONE complete round at the same horizon H
      under the same opponent scenario (same scenario_seed). A result is
      committed only after a full round completes; if the deadline hits
      mid-round the partial round is discarded wholesale and A0 is
      returned (completed_rounds=0, changed=False). Guard-vetoed
      candidates (indices in guard_vetoed) are filtered before the round
      and can never be returned; A0 is exempt from veto filtering (it is
      the mandated fallback). Ties prefer A0, then the smaller index.

      Budget contract (SI02): ONE absolute deadline governs the whole
      public call -- snapshot, baseline, opponent inference, candidate
      generation, guard screening, simulation, leaf and return. The
      wiring layer anchors that deadline at the public entry, spends the
      pre-comparator phases from the same envelope, and passes the
      remainder to decide() as the `deadline` argument (an absolute
      time.monotonic() instant). There is NO budget top-up anywhere: if
      the envelope is already exhausted before decide(), the wiring
      returns A0 without calling decide() at all. Called without
      `deadline`, decide() falls back to t0 + budget_s measured from its
      own entry (unit-test caliber only). Budgets are per-call -- nothing
      banks leftover time across steps.

      Terminal contract (SI01, wiring/rollout side): the authoritative
      terminal is extinction OR the time boundary -- the last valid
      action observation is step 298 and the terminal state is step 299.
      The wiring must reject already-terminal roots before decide(), and
      a rollout must, after every joint advance, freeze at extinction or
      step 299 with zero extra policy calls. Settlement keeps the F01
      caliber: cash carried into the extinction step; a plain time-out
      is scored on the real terminal state's cash.

  StatsCollector
      Standalone telemetry. The comparator's caller records every
      decision via record_decision(); consumers read snapshot(). This
      channel is deliberately decoupled from the measured code's internal
      state: the legacy search stats dict accumulated inside the search
      closure and was never published to the agent, which once made a
      live search read as "never ran". A zero count here means the
      channel is disconnected, not that nothing ran -- check wiring first.

  HeuristicLeafAdapter / VNetLeafAdapter
      Leaf adapters exposing the unified leaf interface
      callable(leaf_state) -> float on the U scale (deployment reward
      = 2U - 1, U in [0, 1]).
        - HeuristicLeafAdapter wraps the in-source heuristic leaf via an
          affine map U = a*x + b. The (a, b) parameters must come from
          this line's own calibration; parameters fitted on another
          policy/teacher must not be migrated in. Out-of-range U values
          are returned unclipped, flagged via last_out_of_range and
          counted in out_of_range_count -- silent clipping would dress a
          scale error up as a normal value.
        - VNetLeafAdapter is the full PPO critic leaf. Its identity is
          (checkpoint, state_md5); U = (tanh(Z_vnet) + 1) / 2. Until a
          concrete critic passes identity / behaviour-matching /
          calibration / cost acceptance, no backend may be injected and
          calling it raises NotImplementedError -- a placeholder must
          never quietly stand in for the real leaf.

RolloutFn = callable(state, action, horizon: int, scenario_seed: int)
            -> leaf_state (opaque)
LeafFn    = callable(leaf_state) -> float, U scale in [0, 1]

Pure Python 3, standard library only, no imports from the policy pack.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Set

__all__ = [
    "Candidate",
    "Decision",
    "Comparator",
    "RolloutFn",
    "LeafFn",
    "StatsCollector",
    "HeuristicLeafAdapter",
    "VNetLeafAdapter",
]

RolloutFn = Callable[[Any, Any, int, int], Any]
LeafFn = Callable[[Any], float]


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    index: int
    action: Any
    is_a0: bool


@dataclass
class Decision:
    action: Any
    chosen_index: int
    completed_rounds: int
    deepest_completed_h: int
    completed_comparisons: int
    changed: bool
    elapsed_s: float
    abort_reason: Optional[str]


class Comparator:
    """One-step, single-horizon, complete-round candidate comparator."""

    def __init__(self, budget_s: float, max_candidates: int = 4):
        if budget_s <= 0:
            raise ValueError("budget_s must be positive")
        if max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        self.budget_s = float(budget_s)
        self.max_candidates = int(max_candidates)

    # -- internal helpers -------------------------------------------------

    def _a0_decision(
        self,
        a0: Candidate,
        t0: float,
        abort_reason: Optional[str],
    ) -> Decision:
        """Fallback decision: return A0, no completed round."""
        return Decision(
            action=a0.action,
            chosen_index=a0.index,
            completed_rounds=0,
            deepest_completed_h=0,
            completed_comparisons=0,
            changed=False,
            elapsed_s=time.monotonic() - t0,
            abort_reason=abort_reason,
        )

    # -- main entry -------------------------------------------------------

    def decide(
        self,
        state: Any,
        candidates: List[Candidate],
        rollout: RolloutFn,
        leaf: LeafFn,
        horizon: int,
        scenario_seed: int,
        guard_vetoed: Set[int] = frozenset(),
        deadline: Optional[float] = None,
    ) -> Decision:
        """Run one complete-round comparison.

        deadline: absolute time.monotonic() instant bounding this call,
        anchored by the wiring layer at the public entry (SI02). When
        given, it replaces the budget_s-derived deadline; budget_s is
        then only the caller's statement of the total envelope. When the
        deadline has already passed at entry, the first pre-call check
        aborts to A0 -- decide() never tops the envelope back up.
        """
        t0 = time.monotonic()
        t_end = float(deadline) if deadline is not None else t0 + self.budget_s

        # ---- contract validation -------------------------------------
        if not candidates:
            raise ValueError("candidates must be non-empty")
        if len(candidates) > self.max_candidates:
            raise ValueError(
                f"too many candidates: {len(candidates)} > {self.max_candidates}"
            )
        a0_list = [c for c in candidates if c.is_a0]
        if len(a0_list) != 1:
            raise ValueError("exactly one candidate must have is_a0=True")
        a0 = a0_list[0]

        # ---- veto filtering (A0 exempt: it is the mandated fallback) --
        eligible = [
            c for c in candidates if c.is_a0 or c.index not in guard_vetoed
        ]
        # Evaluate A0 first, then ascending index: deterministic order, and
        # the fallback's own value is known earliest within the round.
        eligible.sort(key=lambda c: (not c.is_a0, c.index))

        # ---- one complete round at horizon H --------------------------
        values = {}  # candidate index -> leaf value, this round only
        for cand in eligible:
            # Pre-call budget check before rollout.
            if time.monotonic() >= t_end:
                return self._a0_decision(a0, t0, "deadline")
            leaf_state = rollout(state, cand.action, horizon, scenario_seed)
            if time.monotonic() > t_end:
                # The single call itself overran the budget; the round is
                # treated as never completed.
                return self._a0_decision(a0, t0, "single_call_overrun")

            # Pre-call budget check before leaf evaluation.
            if time.monotonic() >= t_end:
                return self._a0_decision(a0, t0, "deadline")
            value = leaf(leaf_state)
            if time.monotonic() > t_end:
                return self._a0_decision(a0, t0, "single_call_overrun")

            values[cand.index] = float(value)

        # ---- round complete: commit the best --------------------------
        # Highest leaf value wins; ties prefer A0, then the smaller index.
        best = min(
            eligible,
            key=lambda c: (-values[c.index], not c.is_a0, c.index),
        )
        return Decision(
            action=best.action,
            chosen_index=best.index,
            completed_rounds=1,
            deepest_completed_h=horizon,
            completed_comparisons=len(values),
            changed=not best.is_a0,
            elapsed_s=time.monotonic() - t0,
            abort_reason=None,
        )


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

class StatsCollector:
    """Search-decision telemetry counters.

    Counters:
      calls                  -- decisions entered into the comparator
      completed_comparisons  -- roots that completed at least one full round
      completed_rounds       -- completed comparison rounds (a root may
                                complete more than one)
      changed_count          -- final action != original policy action A0
      depth_hist             -- {depth: count} over completed rounds
      leaf_calls             -- total leaf evaluations consumed
      cost_accum_s           -- accumulated cost (s); callers pass the
                                TOTAL public-call cost per decision (SI02),
                                not just the comparator kernel's time
      abort_reasons          -- {reason: count} (deadline /
                                single_call_overrun / ...)
      phase_accum_s          -- {phase: accumulated seconds} per-phase
                                breakdown (SI02): e.g. pre_cmp /
                                guard_screen / compare / total

    record_decision() decision-dict keys (missing keys take defaults):
      completed (bool)        this root completed >= 1 full round
      rounds (int)            rounds completed at this root
      changed (bool)          final action differs from A0
      depth (int | None)      deepest completed round's depth (depth_hist)
      leaf_calls (int)        leaf evaluations consumed at this root
      cost_s (float)          total public-call cost at this root (s)
      abort_reason (str|None) non-None values are counted in abort_reasons
      phases (dict|None)      {phase name: seconds} for this root,
                              accumulated into phase_accum_s
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.completed_comparisons = 0
        self.completed_rounds = 0
        self.changed_count = 0
        self.depth_hist: dict[int, int] = {}
        self.leaf_calls = 0
        self.cost_accum_s = 0.0
        self.abort_reasons: dict[str, int] = {}
        self.phase_accum_s: dict[str, float] = {}

    def record_decision(self, decision: dict) -> None:
        self.calls += 1
        rounds = int(decision.get("rounds", 0))
        self.completed_rounds += rounds
        if decision.get("completed", rounds > 0):
            self.completed_comparisons += 1
            depth = decision.get("depth")
            if depth is not None:
                d = int(depth)
                self.depth_hist[d] = self.depth_hist.get(d, 0) + 1
        if decision.get("changed", False):
            self.changed_count += 1
        self.leaf_calls += int(decision.get("leaf_calls", 0))
        self.cost_accum_s += float(decision.get("cost_s", 0.0))
        reason = decision.get("abort_reason")
        if reason:
            self.abort_reasons[reason] = self.abort_reasons.get(reason, 0) + 1
        phases = decision.get("phases")
        if phases:
            for name, secs in phases.items():
                self.phase_accum_s[str(name)] = (
                    self.phase_accum_s.get(str(name), 0.0) + float(secs))

    def snapshot(self) -> dict:
        """JSON-serialisable deep-copy snapshot (depth_hist keys as str)."""
        return {
            "calls": self.calls,
            "completed_comparisons": self.completed_comparisons,
            "completed_rounds": self.completed_rounds,
            "changed_count": self.changed_count,
            "depth_hist": {str(k): v for k, v in sorted(self.depth_hist.items())},
            "leaf_calls": self.leaf_calls,
            "cost_accum_s": self.cost_accum_s,
            "abort_reasons": copy.deepcopy(self.abort_reasons),
            "phase_accum_s": dict(self.phase_accum_s),
        }


# ---------------------------------------------------------------------------
# Leaf adapters
# ---------------------------------------------------------------------------

class HeuristicLeafAdapter:
    """Wraps the in-source heuristic leaf, affine-mapped to the U scale.

    heuristic_fn: (leaf_state) -> float on the heuristic leaf's raw scale.
    affine (a, b): U = a * x + b. The default (1, 0) is a pass-through
    placeholder: production parameters must be measured and registered
    through this line's own calibration, never migrated from another
    policy's fitted affine (different teacher/subject -- no identity
    basis for the transfer).

    Out-of-range discipline: a mapped U outside [0, 1] is returned
    unclipped, with last_out_of_range set and out_of_range_count
    incremented for telemetry/acceptance to read.
    """

    def __init__(
        self,
        heuristic_fn: Callable[[object], float],
        a: float = 1.0,
        b: float = 0.0,
    ) -> None:
        self._fn = heuristic_fn
        self.a = a
        self.b = b
        self.last_out_of_range = False
        self.out_of_range_count = 0
        self.calls = 0

    def __call__(self, leaf_state: object) -> float:
        self.calls += 1
        u = self.a * float(self._fn(leaf_state)) + self.b
        self.last_out_of_range = not (0.0 <= u <= 1.0)
        if self.last_out_of_range:
            self.out_of_range_count += 1
        return u  # unclipped; range discipline is the flag/counter's job


class VNetLeafAdapter:
    """Full PPO vnet leaf (skeleton; disabled until acceptance).

    Identity = (checkpoint, state_md5). U = (tanh(Z_vnet) + 1) / 2 with
    Z_vnet the critic's raw output; deployment reward = 2U - 1.

    backend: (leaf_state) -> float returning Z_vnet (pre-tanh). With no
    backend injected, calling raises NotImplementedError: a critic leaf
    version may only be wired in after its identity / behaviour-matching
    / calibration / cost acceptances have all passed.
    """

    def __init__(
        self,
        checkpoint: int,
        state_md5: str,
        backend: Optional[Callable[[object], float]] = None,
    ) -> None:
        self.checkpoint = int(checkpoint)
        self.state_md5 = str(state_md5)
        self._backend = backend
        self.last_out_of_range = False  # tanh output is always in (0,1)
        self.calls = 0

    def identity(self) -> dict:
        return {"checkpoint": self.checkpoint, "state_md5": self.state_md5}

    def __call__(self, leaf_state: object) -> float:
        if self._backend is None:
            raise NotImplementedError(
                "VNetLeafAdapter backend not injected: critic leaf disabled "
                "until its acceptances pass"
            )
        self.calls += 1
        z = float(self._backend(leaf_state))
        return (math.tanh(z) + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Self-test: fake rollout/leaf, no external dependencies.
# ---------------------------------------------------------------------------

def _self_test() -> None:
    # --- Case 1: complete round, best non-A0 candidate wins -------------
    values = {"a0": 0.50, "a1": 0.80, "a2": 0.65}

    def rollout_fast(state, action, horizon, scenario_seed):
        return action  # leaf_state is just the action token here

    def leaf_fast(leaf_state):
        return values[leaf_state]

    cands = [
        Candidate(index=0, action="a0", is_a0=True),
        Candidate(index=1, action="a1", is_a0=False),
        Candidate(index=2, action="a2", is_a0=False),
    ]
    comp = Comparator(budget_s=1.0)
    d = comp.decide(None, cands, rollout_fast, leaf_fast, horizon=1,
                    scenario_seed=42)
    print("case1 complete-round:", d)
    assert d.chosen_index == 1 and d.changed is True
    assert d.completed_rounds == 1 and d.deepest_completed_h == 1
    assert d.completed_comparisons == 3 and d.abort_reason is None

    # --- Case 2: deadline mid-round -> fall back to A0 ------------------
    def rollout_slow(state, action, horizon, scenario_seed):
        time.sleep(0.30)
        return action

    comp2 = Comparator(budget_s=0.05)
    d2 = comp2.decide(None, cands, rollout_slow, leaf_fast, horizon=1,
                      scenario_seed=42)
    print("case2 deadline->A0:", d2)
    assert d2.chosen_index == 0 and d2.changed is False
    assert d2.completed_rounds == 0 and d2.deepest_completed_h == 0
    assert d2.completed_comparisons == 0
    assert d2.abort_reason == "single_call_overrun"

    # --- Case 3: vetoed best candidate is filtered, runner-up wins ------
    comp3 = Comparator(budget_s=1.0)
    d3 = comp3.decide(None, cands, rollout_fast, leaf_fast, horizon=1,
                      scenario_seed=42, guard_vetoed={1})
    print("case3 veto-filter:", d3)
    assert d3.chosen_index == 2 and d3.changed is True
    assert d3.completed_comparisons == 2 and d3.abort_reason is None

    # --- Case 3b: all non-A0 vetoed -> A0 returned, round still counts --
    d3b = comp3.decide(None, cands, rollout_fast, leaf_fast, horizon=1,
                       scenario_seed=42, guard_vetoed={1, 2})
    print("case3b all-vetoed->A0:", d3b)
    assert d3b.chosen_index == 0 and d3b.changed is False
    assert d3b.completed_rounds == 1 and d3b.completed_comparisons == 1

    # --- Case 4: telemetry + leaf adapters -------------------------------
    sc = StatsCollector()
    sc.record_decision({"completed": True, "rounds": 1, "changed": True,
                        "depth": 1, "leaf_calls": 3, "cost_s": 0.01,
                        "abort_reason": None,
                        "phases": {"pre_cmp": 0.004, "compare": 0.006}})
    sc.record_decision({"completed": False, "rounds": 0, "changed": False,
                        "depth": None, "leaf_calls": 1, "cost_s": 0.30,
                        "abort_reason": "single_call_overrun",
                        "phases": {"pre_cmp": 0.29, "compare": 0.01}})
    snap = sc.snapshot()
    print("case4 telemetry:", snap)
    assert snap["calls"] == 2 and snap["completed_comparisons"] == 1
    assert snap["changed_count"] == 1 and snap["leaf_calls"] == 4
    assert snap["abort_reasons"] == {"single_call_overrun": 1}
    assert snap["depth_hist"] == {"1": 1}
    assert abs(snap["phase_accum_s"]["pre_cmp"] - 0.294) < 1e-9
    assert abs(snap["phase_accum_s"]["compare"] - 0.016) < 1e-9

    # --- Case 4b: absolute deadline already passed -> A0, no top-up -----
    comp4 = Comparator(budget_s=5.0)
    d4 = comp4.decide(None, cands, rollout_fast, leaf_fast, horizon=1,
                      scenario_seed=42, deadline=time.monotonic() - 1.0)
    print("case4b past-deadline->A0:", d4)
    assert d4.chosen_index == 0 and d4.completed_rounds == 0
    assert d4.abort_reason == "deadline"
    # ... while a generous absolute deadline lets the round complete even
    # with a small nominal budget_s (deadline replaces it).
    d4b = comp4.decide(None, cands, rollout_fast, leaf_fast, horizon=1,
                       scenario_seed=42, deadline=time.monotonic() + 5.0)
    assert d4b.chosen_index == 1 and d4b.completed_rounds == 1

    ha = HeuristicLeafAdapter(lambda s: 2.0 * s, a=0.25, b=0.0)
    assert ha(1.0) == 0.5 and ha.last_out_of_range is False
    assert ha(3.0) == 1.5 and ha.last_out_of_range is True
    assert ha.out_of_range_count == 1 and ha.calls == 2

    va = VNetLeafAdapter(checkpoint=33, state_md5="deadbeef")
    assert va.identity() == {"checkpoint": 33, "state_md5": "deadbeef"}
    try:
        va(None)
        raise AssertionError("VNetLeafAdapter without backend must raise")
    except NotImplementedError:
        pass
    vb = VNetLeafAdapter(checkpoint=33, state_md5="deadbeef",
                         backend=lambda s: 0.0)
    assert vb(None) == 0.5

    print("ALL SELF-TESTS PASSED")


if __name__ == "__main__":
    _self_test()

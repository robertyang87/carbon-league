"""Shallow tactical-search comparator core for the frozen Echo4 policy pack.

Contract (preregistered, S0b prototype; SI02/SI06/anytime fixes applied):
  - K <= 4 candidates, always including A0 (the original policy action).
  - Candidates arrive already produced by real Prefix re-decode; this module
    only compares them via an injected rollout + leaf pair.
  - All eligible candidates are compared in ONE complete round at the same
    horizon H and under the same opponent scenario (same scenario_seed).
  - A result is committed only after a full round completes. If a round is
    aborted mid-way (deadline / overrun / non-finite leaf / exception /
    C_hi admission), the partial round is discarded wholesale and the
    previous complete round's result is returned; if no round has
    completed yet, A0 is returned (completed_rounds=0, changed=False).
  - Anytime multi-round: decide_multi() takes a horizon sequence
    (e.g. [1, 2, 3]); rounds run in order, each completed round replaces
    the committed result. decide() is the single-horizon special case.
  - The deadline is a single absolute point on the time.monotonic()
    clock: either t0 + budget_s (standalone use) or a caller-supplied
    deadline_abs (production entry, so the budget covers the whole
    public call chain, not just this kernel). Budgets are per-call:
    nothing here banks leftover time across steps.
  - C_hi whole-round admission: if est_candidate_cost_s is a positive
    number, a round is only started when est_candidate_cost_s *
    n_eligible fits in the remaining budget; otherwise the round is
    not started at all (abort_reason="budget_gate") instead of being
    cancelled mid-flight. The estimate is the caller's high-cost bound
    for THIS round's (n_candidates, horizon) shape (bucketed cost
    model in real_env, R2-04); est_candidate_cost_s=None means cold
    start (no sample for any shape yet): admission is waived for that
    round -- explicitly NOT a measured zero cost -- and the in-round
    deadline checks below still apply.
  - Finiteness is checked independently of scale: leaf values may lie
    outside [0, 1] (raw-margin leaves), but any NaN / +-Inf value makes
    the whole round invalid (abort_reason="nonfinite_leaf").
  - Rollout / leaf exceptions abort the current round
    (abort_reason="exception") and fall back to the previous complete
    round / A0. Contract violations (no candidates, not exactly one
    A0, more than max_candidates, empty horizon sequence) still raise
    ValueError -- fail fast, never silently degraded.
  - Guard-vetoed candidates (indices in guard_vetoed) are filtered out
    before the round and can never be returned, even with the highest
    leaf value. A0 is the mandated fallback and is exempt from veto
    filtering.

Shared API (other prototype modules code against these names):
  RolloutFn = callable(state, action, horizon: int, scenario_seed: int)
              -> leaf_state (opaque)
  LeafFn    = callable(leaf_state) -> float
  Comparator(budget_s: float, max_candidates: int = 4)
  Comparator.decide(state, candidates, rollout, leaf, horizon,
                    scenario_seed, guard_vetoed=frozenset(),
                    est_candidate_cost_s=0.0, deadline_abs=None) -> Decision
  Comparator.decide_multi(state, candidates, rollout, leaf, horizons,
                    scenario_seed, guard_vetoed=frozenset(),
                    est_candidate_cost_s=0.0, deadline_abs=None) -> Decision

  Decision.total_elapsed_s / Decision.phase_costs are filled in by the
  production entry (real_env) with whole-call accounting; this kernel
  leaves them None.

Pure Python 3, standard library only, no imports from the policy pack.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

__all__ = ["Candidate", "Decision", "Comparator", "RolloutFn", "LeafFn"]

RolloutFn = Callable[[Any, Any, int, int], Any]
LeafFn = Callable[[Any], float]


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
    # Whole-call accounting, filled by the production entry (real_env);
    # None when produced by this kernel standalone.
    total_elapsed_s: Optional[float] = None
    phase_costs: Optional[Dict[str, float]] = None


class Comparator:
    """Complete-round candidate comparator with anytime horizon rounds."""

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

    # -- main entries -----------------------------------------------------

    def decide(
        self,
        state: Any,
        candidates: List[Candidate],
        rollout: RolloutFn,
        leaf: LeafFn,
        horizon: int,
        scenario_seed: int,
        guard_vetoed: Set[int] = frozenset(),
        *,
        est_candidate_cost_s: float = 0.0,
        deadline_abs: Optional[float] = None,
    ) -> Decision:
        """Single-horizon complete-round comparison (original contract)."""
        return self._decide_horizons(
            state, candidates, rollout, leaf, [horizon], scenario_seed,
            guard_vetoed, est_candidate_cost_s, deadline_abs)

    def decide_multi(
        self,
        state: Any,
        candidates: List[Candidate],
        rollout: RolloutFn,
        leaf: LeafFn,
        horizons: List[int],
        scenario_seed: int,
        guard_vetoed: Set[int] = frozenset(),
        *,
        est_candidate_cost_s: float = 0.0,
        deadline_abs: Optional[float] = None,
    ) -> Decision:
        """Anytime multi-round comparison over a horizon sequence.

        Rounds run in the given order (expected ascending). Each round
        that completes replaces the committed result; a round that
        aborts leaves the previous complete round's result in place
        (abort_reason records why the sequence stopped). If the first
        round aborts, the result is the A0 fallback.
        """
        if not horizons:
            raise ValueError("horizons must be a non-empty sequence")
        horizons = [int(h) for h in horizons]
        if any(h < 1 for h in horizons):
            raise ValueError("horizons must be >= 1")
        return self._decide_horizons(
            state, candidates, rollout, leaf, horizons, scenario_seed,
            guard_vetoed, est_candidate_cost_s, deadline_abs)

    # -- round machinery ----------------------------------------------------

    def _decide_horizons(
        self,
        state: Any,
        candidates: List[Candidate],
        rollout: RolloutFn,
        leaf: LeafFn,
        horizons: List[int],
        scenario_seed: int,
        guard_vetoed: Set[int],
        est_candidate_cost_s: float,
        deadline_abs: Optional[float],
    ) -> Decision:
        t0 = time.monotonic()
        deadline = deadline_abs if deadline_abs is not None else t0 + self.budget_s

        # ---- contract validation (fail fast) -------------------------
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

        completed_rounds = 0
        total_comparisons = 0
        best_overall: Optional[Candidate] = None
        deepest_h = 0
        abort_reason: Optional[str] = None

        for horizon in horizons:
            # ---- C_hi whole-round admission ---------------------------
            # Never start a round whose estimated whole-round cost does
            # not fit in the remaining budget; an un-cancellable call
            # discovered overrunning afterwards cannot refund its time.
            # est=None is the documented cold start (no cost samples at
            # all): admission waived, round still deadline-checked.
            if est_candidate_cost_s is not None and est_candidate_cost_s > 0:
                est_round = est_candidate_cost_s * len(eligible)
                if est_round > deadline - time.monotonic():
                    abort_reason = "budget_gate"
                    break

            # ---- one complete round at this horizon -------------------
            values: Dict[int, float] = {}  # candidate index -> leaf value
            round_abort: Optional[str] = None
            for cand in eligible:
                # Pre-call budget check before rollout.
                if time.monotonic() >= deadline:
                    round_abort = "deadline"
                    break
                try:
                    leaf_state = rollout(state, cand.action, horizon,
                                         scenario_seed)
                except Exception:
                    round_abort = "exception"
                    break
                if time.monotonic() > deadline:
                    # The single call itself overran the budget; the
                    # round is treated as never completed.
                    round_abort = "single_call_overrun"
                    break

                # Pre-call budget check before leaf evaluation.
                if time.monotonic() >= deadline:
                    round_abort = "deadline"
                    break
                try:
                    value = leaf(leaf_state)
                except Exception:
                    round_abort = "exception"
                    break
                if time.monotonic() > deadline:
                    round_abort = "single_call_overrun"
                    break

                values[cand.index] = float(value)

            if round_abort is not None:
                abort_reason = round_abort
                break

            # ---- finiteness gate (independent of scale) ---------------
            # Raw-margin leaves may exceed [0, 1]; NaN / +-Inf may not
            # pass. Any bad value invalidates the whole round.
            if any(not math.isfinite(v) for v in values.values()):
                abort_reason = "nonfinite_leaf"
                break

            # ---- round complete: commit the best ----------------------
            # Highest leaf value wins; ties prefer A0, then smaller index.
            best = min(
                eligible,
                key=lambda c: (-values[c.index], not c.is_a0, c.index),
            )
            completed_rounds += 1
            total_comparisons += len(values)
            best_overall = best
            deepest_h = horizon
            abort_reason = None

        if completed_rounds == 0:
            return self._a0_decision(a0, t0, abort_reason)
        assert best_overall is not None
        return Decision(
            action=best_overall.action,
            chosen_index=best_overall.index,
            completed_rounds=completed_rounds,
            deepest_completed_h=deepest_h,
            completed_comparisons=total_comparisons,
            changed=not best_overall.is_a0,
            elapsed_s=time.monotonic() - t0,
            abort_reason=abort_reason,
        )


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

    print("ALL SELF-TESTS PASSED")


if __name__ == "__main__":
    _self_test()

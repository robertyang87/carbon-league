"""Shallow tactical-search comparator core for the frozen Echo4 policy pack.

Contract (preregistered, S0b prototype):
  - K <= 4 candidates, always including A0 (the original policy action).
  - Candidates arrive already produced by real Prefix re-decode; this module
    only compares them via an injected rollout + leaf pair.
  - All eligible candidates are compared in ONE complete round at the same
    horizon H and under the same opponent scenario (same scenario_seed).
  - A result is committed only after a full round completes. If the deadline
    hits mid-round, the partial round is discarded wholesale and A0 is
    returned (completed_rounds=0, changed=False).
  - The deadline is measured from decide() entry with time.monotonic();
    budget_s is supplied by the caller. Budgets are per-call: nothing here
    banks leftover time across steps.
  - Guard-vetoed candidates (indices in guard_vetoed) are filtered out
    before the round and can never be returned, even with the highest leaf
    value. A0 is the mandated fallback and is exempt from veto filtering.

Shared API (other prototype modules code against these names):
  RolloutFn = callable(state, action, horizon: int, scenario_seed: int)
              -> leaf_state (opaque)
  LeafFn    = callable(leaf_state) -> float, U scale in [0, 1]
  Comparator(budget_s: float, max_candidates: int = 4)
  Comparator.decide(state, candidates, rollout, leaf, horizon,
                    scenario_seed, guard_vetoed=frozenset()) -> Decision

Pure Python 3, standard library only, no imports from the policy pack.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Set

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
    ) -> Decision:
        t0 = time.monotonic()

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
            if time.monotonic() - t0 >= self.budget_s:
                return self._a0_decision(a0, t0, "deadline")
            leaf_state = rollout(state, cand.action, horizon, scenario_seed)
            if time.monotonic() - t0 > self.budget_s:
                # The single call itself overran the budget; the round is
                # treated as never completed.
                return self._a0_decision(a0, t0, "single_call_overrun")

            # Pre-call budget check before leaf evaluation.
            if time.monotonic() - t0 >= self.budget_s:
                return self._a0_decision(a0, t0, "deadline")
            value = leaf(leaf_state)
            if time.monotonic() - t0 > self.budget_s:
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

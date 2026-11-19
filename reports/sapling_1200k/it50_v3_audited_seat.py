"""Replay one failed recording on its fixed map/seat, accounting for native zero-depth fallbacks."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import time

from it50_v3_exam import CPU, MAP, PACK, PRIOR, SHA, scoring

exam = scoring.exam


def freeze_stats(stats):
    return tuple(stats[key] for key in ("searched", "single", "aborted"))


def audit_after_call(before, stats):
    searched, single, aborted = (n - p for n, p in zip(freeze_stats(stats), before))
    if (searched, single) == (1, 0) and aborted in (0, 1):
        return "searched"
    if (searched, single, aborted) == (0, 1, 0):
        return "single"
    if (searched, single, aborted) == (0, 0, 1) and stats["last"][1] == 0:
        return "depth_zero_return_base"
    raise RuntimeError(f"unknown search transition {before!r} -> {freeze_stats(stats)!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", default=MAP, type=int)
    p.add_argument("--seat", default=1, type=int, choices=(0,1))
    args = p.parse_args()
    if args.out.exists() or PRIOR.digest(PACK) != SHA or PRIOR.digest(PRIOR.SOURCE, "md5") != PRIOR.EXPECTED_MD5:
        p.error("fresh output and frozen identities required")
    os.sched_setaffinity(0, {CPU})
    seed, seat = args.seed, args.seat
    random.seed(7919 * seed + 104729 * seat + 31)
    sim = exam.FastCarbon(exam.CFG).reset(seed=seed)
    observer = PRIOR.Observer()
    candidate = PRIOR.shared.module(PACK, "audited_hybrid")
    opponent = PRIOR.shared.module(PRIOR.SOURCE, "audited_master")
    assert opponent._TEMP == 1 and opponent._SEED is None and opponent._SEARCH_OK and opponent._RX
    cfg = {k: getattr(opponent, k) for k in exam.SEARCH_KEYS}
    income = [0.0, 0.0]
    search_calls = Counter()
    elapsed = [[], []]
    errors = [0, 0]
    calls = [0, 0]
    events = []
    moments = {}
    entered = 0
    unclassified = []
    original_search = opponent._search
    def counted_search(*args, **kwargs):
        nonlocal entered
        entered += 1
        before = freeze_stats(opponent._S_STATS)
        result = original_search(*args, **kwargs)
        try:
            reason = audit_after_call(before, opponent._S_STATS)
        except RuntimeError as error:
            reason = "unclassified"
            unclassified.append({"root_t": sim.step_count, "error": str(error)})
        search_calls[reason] += 1
        return result
    opponent._search = counted_search
    started = time.monotonic()
    cpu_start = time.process_time()
    quota_stats = Path("/sys/fs/cgroup/cpu.stat")
    candidate_agent = candidate.agent
    agents = [opponent.agent, opponent.agent]
    agents[seat] = candidate_agent
    report = {"status": "running", "map": seed, "seat": seat, "cpu": CPU, "pack_sha256": SHA,
              "recording": ("Seat 1 repeat. Initial attempt's final result could not be recovered from legacy harness assertion. Same map, same seat, fresh opponent randomness."
                            if (seed, seat) == (MAP, 1) else "Fresh fixed development map and seat"),
              "cgroup_cpu_stat_start": quota_stats.read_text(), "load_average_start": os.getloadavg(),
              "examiner_md5": PRIOR.digest(PRIOR.SOURCE, "md5"), "configuration": cfg,
              "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
              "utc_started": PRIOR.utc(), "examiner_rng": "TEMP=1; SEED=None"}
    PRIOR.shared.atomic_json(args.out, report)
    try:
        for _ in range(exam.EPISODE_STEPS - 1):
            pre = sim.get_state()
            actions = []
            for player, agent in enumerate(agents):
                start = time.monotonic()
                try:
                    action = agent(sim.obs(player), sim.config)
                    ids = set(sim.player_workers[player]) | {bid for bid, _ in sim.bases[player]}
                    if not isinstance(action, dict) or set(action) - ids or any(
                        v not in ("UP","RIGHT","DOWN","LEFT","RECCOLLECTOR","RECPLANTER",None) for v in action.values()
                    ):
                        raise ValueError("illegal action dict")
                    actions.append(action)
                except Exception:
                    errors[player] += 1
                    raise
                finally:
                    calls[player] += 1
                    elapsed[player].append(time.monotonic() - start)
            events.append(actions)
            observer.step(sim, actions)
            post = sim.get_state()
            extinct = [bool(exam._extinct(post, player)) for player in (0, 1)]
            for tid, tree in post["trees"].items():
                income[tree["player"]] += float(post["trees_dict"].get(tid,[None, 0])[1])
            if sim.step_count in (60,100,200,299) or any(extinct):
                moments[sim.step_count] = [PRIOR.snapshot(sim, observer, p, errors, calls) for p in (0, 1)]
                report.update(steps=sim.step_count, search_return_types=dict(search_calls))
                PRIOR.shared.atomic_json(args.out, report)
            if any(extinct):
                break
        scores = [(sim.step_count - exam.EPISODE_STEPS - 1) if extinct[p]
                  else float((pre if any(extinct) else post)["cash"][p]) for p in (0, 1)]
        margin = scores[seat] - scores[1-seat]
        audit_ok = (entered == sum(search_calls.values()) == sim.step_count and search_calls["searched"] > 0
                    and not unclassified and cfg == {k: getattr(opponent, k) for k in exam.SEARCH_KEYS})
        report.update(status="complete", utc_finished=PRIOR.utc(), official_score_by_seat=scores, margin=margin,
                      audit_passed=audit_ok, unclassified_search_calls=unclassified,
                      cgroup_cpu_stat_end=quota_stats.read_text(), load_average_end=os.getloadavg(),
                      process_cpu_seconds=time.process_time()-cpu_start,
                      result=("win" if margin > 0 else "loss" if margin < 0 else "draw"),
                      actual_transitions=sim.step_count, extinct_by_seat=extinct, source_search_invocations=entered,
                      search_return_types=dict(search_calls), raw_search_stats=opponent._S_STATS,
                      own_model_action_attribution=candidate._COUNTS,
                      cumulative_tree_income_by_seat=income, snapshots_by_step_and_seat=moments,
                      call_seconds_by_seat=[exam.small_stats(x) for x in elapsed],
                      agent_errors_by_seat=errors, wall_seconds=time.monotonic()-started,
                      actions_sha256=PRIOR.hashed(events), final_state_sha256=PRIOR.hashed(post))
    except Exception as e:
        report.update(status="failed", error=repr(e), step=sim.step_count, errors=errors,
                      source_search_invocations=entered, raw_counts=freeze_stats(opponent._S_STATS))
        raise
    finally:
        PRIOR.shared.atomic_json(args.out, report)
    print(json.dumps({key:report[key] for key in ("result", "margin", "actual_transitions", "search_return_types")}), flush=True)


if __name__ == "__main__":
    main()

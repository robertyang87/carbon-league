"""Play a complete game and persist the score and actions before judging search coverage."""
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import random
import time

import cash_v3_exam as cash
from it50_v3_audited_seat import audit_after_call, freeze_stats

exam = cash.exam
prior = exam.prior


def play(job):
    agent_file, weight_id, seed, side, cpu, directory, pack_sha = job
    os.sched_setaffinity(0, {cpu})
    output = Path(directory)
    if prior.digest(agent_file) != pack_sha or prior.digest(prior.SOURCE, "md5") != prior.EXPECTED_MD5:
        raise ValueError("artifact identity mismatch")
    random.seed(7919 * seed + 104729 * side + 31)
    started = time.monotonic()
    sim = exam.FastCarbon(exam.CFG).reset(seed=seed)
    observer = prior.Observer()
    candidate = prior.shared.module(agent_file, "candidate_screen")
    opp = prior.shared.module(prior.SOURCE, "master_original")
    configuration = {key: getattr(opp,key) for key in exam.SEARCH_KEYS}
    if not (opp._SEARCH_OK and opp._RX and opp._TEMP == 1 and opp._SEED is None):
        raise ValueError("opponent must use original enabled search and unseeded temperature one")
    from tree_win_gate import GUARDED_V5
    guarded_candidate = pack_sha == GUARDED_V5
    unchanged_action_steps = []
    prior_guard_changes = 0
    agents = [None, None]
    agents[side], agents[1-side] = candidate.agent, opp.agent
    search_events, actions_log, errors, warnings = [], [], [], []
    original_search = opp._search
    def search(*args, **kwargs):
        before = freeze_stats(opp._S_STATS)
        result = original_search(*args, **kwargs)
        try:
            kind = audit_after_call(before, opp._S_STATS)
        except Exception as e:
            warnings.append(repr(e))
            kind = "unknown"
        search_events.append([sim.step_count, kind])
        return result
    opp._search = search
    calls, failures, times = [0,0], [0,0], [[],[]]
    snapshots, income = {}, [0.,0.]
    extinct = [False,False]
    pre = post = sim.get_state()
    started_utc = prior.utc()
    def snap():
        result = {}
        for name, seat in (("candidate",side), ("opponent",1-side)):
            state = prior.snapshot(sim, observer, seat, failures, calls)
            stats = {key: state[key] for key in cash.ECONOMY if key != "cumulative_tree_absorption_income"}
            stats["cumulative_tree_absorption_income"] = income[seat]
            deposits = [e for e in observer.events if e["event"] == "deposit" and e["player"] == seat and e["amount"] > 0]
            stats["distinct_delivery_turns"] = len({e["step"] for e in deposits})
            if abs(sum(e["amount"] for e in deposits) - state["own_positive_delivery_carbon"]) > 1e-6:
                warnings.append(f"deposit accounting mismatch at {sim.step_count}/{seat}")
            result[name] = stats
        return result
    for _ in range(exam.EPISODE_STEPS-1):
        pre = sim.get_state()
        actions = []
        for player, agent in enumerate(agents):
            start = time.monotonic()
            try:
                action = agent(sim.obs(player), sim.config)
                ids = set(sim.player_workers[player]) | {bid for bid,_ in sim.bases[player]}
                if not isinstance(action, dict) or set(action)-ids or any(a not in ("UP","RIGHT","DOWN","LEFT","RECCOLLECTOR","RECPLANTER",None) for a in action.values()):
                    raise ValueError("invalid action")
                if guarded_candidate and player == side:
                    changes = candidate.agent.last_stats["changed_roots"]
                    if changes == prior_guard_changes and any(action.values()):
                        unchanged_action_steps.append(sim.step_count)
                    prior_guard_changes = changes
                actions.append(action)
            except Exception as e:
                failures[player] += 1
                errors.append([sim.step_count, player, repr(e)])
                actions.append({})
            finally:
                calls[player] += 1
                times[player].append(time.monotonic()-start)
        actions_log.append(actions)
        if errors:
            break
        observer.step(sim, actions)
        post = sim.get_state()
        extinct = [bool(exam._extinct(post, p)) for p in (0,1)]
        for tid, tree in post["trees"].items():
            income[tree["player"]] += float(post["trees_dict"].get(tid,[None,0])[1])
        if sim.step_count in (60,100,200,299):
            snapshots[str(sim.step_count)] = snap()
        if any(extinct):
            break
    steps = sim.step_count
    complete = not errors and (any(extinct) or steps == exam.EPISODE_STEPS-1)
    scores = [(steps-exam.EPISODE_STEPS-1) if extinct[p] else float((pre if any(extinct) else post)["cash"][p]) for p in (0,1)] if complete else None
    margin = scores[side]-scores[1-side] if complete else None
    result = None if margin is None else ("win" if margin > 0 else "loss" if margin < 0 else "draw")
    by_type = Counter(kind for _,kind in search_events)
    raw = {"started":started_utc, "finished":prior.utc(), "seed":seed, "candidate_seat":side,
           "pack_sha256":pack_sha, "weight_identity":weight_id, "examiner_md5":prior.EXPECTED_MD5,
           "configuration":configuration, "game_configuration":exam.CFG, "actions":actions_log,
           "search_calls":search_events, "official_score_by_seat":scores, "margin":margin, "result":result,
           "last_pre_cash":pre["cash"], "last_post_cash":post["cash"], "steps":steps,
           "extinct_by_seat":extinct, "errors":errors, "audit_messages":warnings}
    attribution = {}
    if guarded_candidate:
        attribution["candidate_guard_stats"] = dict(getattr(candidate.agent, "last_stats", {}))
        attribution["unchanged_network_action_steps"] = unchanged_action_steps
        raw.update(attribution)
    path = output / f"raw_{seed}_{side}.json.gz"
    temp = path.with_suffix(".tmp")
    temp.write_bytes(gzip.compress(json.dumps(raw,allow_nan=False).encode(),mtime=0))
    os.replace(temp, path)
    if configuration != {key: getattr(opp,key) for key in exam.SEARCH_KEYS}:
        warnings.append("examiner configuration changed")
    if len(search_events) != calls[1-side] or by_type["searched"] == 0 or [step for step,_ in search_events] != list(range(calls[1-side])):
        warnings.append("incomplete source search invocations")
    if not complete:
        warnings.append("game did not complete")
    return dict(**attribution,seed=seed,candidate_seat=side,result=result,margin=margin,official_score_by_seat=scores,
                actual_transitions=steps,termination="agent_error" if errors else ("extinction" if any(extinct) else "official_episode_limit"),
                extinct_by_seat=extinct,agent_calls_by_seat=calls,agent_errors_by_seat=failures,errors=errors,
                scope="direct original search function wrapped for per-call accounting; raw score saved first",
                search=opp._S_STATS, search_return_types=dict(by_type), search_invocations=len(search_events),
                audit_messages=warnings, valid=not warnings, wall_seconds=time.monotonic()-started,
                call_seconds_by_seat=[exam.small_stats(x) for x in times],snapshots=snapshots,final=snap(),
                raw_file=path.name,raw_sha256=prior.digest(path),actions_sha256=prior.hashed(actions_log),
                final_state_sha256=prior.hashed(post), cpu_affinity=sorted(os.sched_getaffinity(0)),
                pack_config=getattr(candidate,"_POLICY",getattr(candidate,"_OPTS",None)))

"""Use the existing Master arena and save D1's exact search and student contribution at each call."""
from pathlib import Path

import tree_audited_game as arena
import tree_pack_arena
from turning_point_ledger import compact, sha, write

_BASE_PLAY = arena.play


def play(job):
    pack, _, seed, seat, _, folder, _ = job
    load = arena.prior.shared.module
    calls = []

    def load_with_trace(path, name):
        module = load(path, name)
        if Path(path) == Path(pack) and name == "candidate_screen":
            run = module.agent
            def record(obs, cfg):
                action = run(obs, cfg)
                st = run.last_stats
                totals = (st["depth_sum"], st["changed"])
                old = calls[-1]["cumulative_rounds_changes"] if calls else (0, 0)
                changed = bool(module._LAST["changed"])
                complete = totals[0] - old[0]
                if (complete not in (0,1) or totals[1]-old[1] != int(changed)
                        or (changed and not complete) or st.get("errors") or st.get("root_errors")):
                    raise ValueError("D1 search/override audit disagrees")
                calls.append(dict(root=obs["step"], base=module._LAST["base"],
                                  executed_sha256=sha(compact(action)), search_changed=changed,
                                  completed_h1=complete, legal_branches=module._LAST["legal_branches"],
                                  unforced_sapling_decisions=module._LAST["selected_unforced_differs_from_expert"],
                                  cumulative_rounds_changes=totals))
                return action
            module.agent = record
        return module

    arena.prior.shared.module = load_with_trace
    try:
        row = _BASE_PLAY(job)
    finally:
        arena.prior.shared.module = load
    if [r["root"] for r in calls] != list(range(row["actual_transitions"])):
        raise ValueError("missing live candidate action trace")
    name = f"d1_search_{seed}_{seat}.json.gz"
    digest = write(Path(folder)/name, dict(seed=seed, seat=seat, pack_sha256=job[-1],
                   raw_sha256=row["raw_sha256"], calls=calls), compressed=True)
    row["candidate_search_audit"] = dict(file=name, sha256=digest,
                  search_changes=sum(c["search_changed"] for c in calls),
                  completed_h1=sum(c["completed_h1"] for c in calls),
                  unforced_sapling_decisions=sum(len(c["unforced_sapling_decisions"]) for c in calls))
    return row


if __name__ == "__main__":
    arena.play = play
    tree_pack_arena.main()

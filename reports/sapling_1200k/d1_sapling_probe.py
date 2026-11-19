"""Check D1's shared-prefix roots against archived hybrid decisions before direct play."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import time

import numpy as np
from d1_sapling_build import HERE, OUTPUT, PARENT
from tree_audited_game import exam, prior
from tree_win_gate import validate_game, verify_raw
from turning_point_ledger import write


def run(directory, limit, offset):
    data = json.loads((directory/"results.json").read_bytes())
    c = prior.shared.module(OUTPUT, "isolated_d1_sapling")
    parent = prior.shared.module(PARENT, "isolated_bare_hybrid")
    if (data["header"]["pack_sha256"] != prior.digest(PARENT) or c.CONF["GUARD"] or c.CONF["CMP_H"] != 1
            or c.CONF["CMP_TAU"] != 30):
        raise ValueError("wrong baseline or comparator")
    leaf_values = []
    kernel = c._S.HeuristicLeafAdapter
    class TraceLeaf(kernel):
        def __call__(self, leaf_state):
            v = super().__call__(leaf_state)
            leaf_values.append(v)
            return v
    c._S.HeuristicLeafAdapter = TraceLeaf
    rows = []
    for g in data["games"][offset:offset+2]:
        validate_game(g)
        verify_raw(directory, g, data["header"]["pack_sha256"], data["header"]["weight_sha256"])
        seed, seat = g["seed"], g["candidate_seat"]
        record = json.loads(gzip.decompress((directory/g["raw_file"]).read_bytes()))
        sim = exam.FastCarbon(record["game_configuration"]).reset(seed=seed)
        counts, timing, examples, improvements = Counter(), [], [], []
        for t, actions in enumerate(record["actions"][:limit]):
            obs = sim.obs(seat)
            reference = parent.agent(obs, sim.config)
            leaf_values.clear()
            start = time.perf_counter()
            # The visible view must be all that reaches the new search.
            selected = c.agent(dict(obs, oracle_carbon=[-12345]*225, full_carbon=[-54321]*225), sim.config)
            timing.append(1000*(time.perf_counter()-start))
            if (reference != actions[seat] or reference != c._LAST["base"] or
                    tuple(sorted(selected.items())) not in c._WITNESSES or c.agent.last_stats.get("errors")
                    or c.agent.last_stats.get("root_errors") or c.agent.last_stats.get("cmp_guard_errors")):
                raise ValueError(f"action identity or comparator error at {seed}/{seat}/t{t}: {c.agent.last_stats.get('last_error')}")
            if len(leaf_values) >= 2:
                improvements.append(float(max(leaf_values)-leaf_values[0]))
            counts["roots"] += 1
            counts["legal_redecoded_alternatives"] += c._LAST["legal_branches"]
            counts["unforced_sapling_contributions_on_selected_prefix"] += len(c._LAST["selected_unforced_differs_from_expert"])
            if selected != reference:
                counts["search_changes"] += 1
                if len(examples) < 4:
                    examples.append(dict(root=t, a0=reference, selected=selected))
            sim.step(*actions)
        st=c.agent.last_stats
        rows.append(dict(seed=seed, seat=seat, **counts, witnesses=examples,
                         max_leaf_advantage=max(improvements, default=None), positive_best_alt=sum(v>1e-7 for v in improvements),
                         compared_h1_rounds=st["depth_sum"], kept_branches=st["branches"],
                         step0_ms=timing[0], warm_ms=dict(mean=float(np.mean(timing[1:])),p50=float(np.percentile(timing[1:],50)),
                                                p95=float(np.percentile(timing[1:],95)),max=max(timing[1:]))))
    if any(r["compared_h1_rounds"] <= 0 or
            r["unforced_sapling_contributions_on_selected_prefix"] <= 0 for r in rows):
        raise ValueError("no observed search or no actual student contribution: "+str(rows))
    return dict(pack_sha256=prior.digest(OUTPUT),parent_sha256=prior.digest(PARENT),
                actuation_observed=any(r.get("search_changes",0) for r in rows),
                scope="Archived parent trajectories. Frozen D1 comparator/heuristic/readable-state reconstruction. Enemy modeled by it50 style 1. Same-prefix trained Sapling contributes 25% on each own root and legal alternative. Search and timing here are diagnostics under shared CPU, no game outcomes.",games=rows)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=HERE/"delivery_parent_dev")
    p.add_argument("--offset", type=int, choices=(0,2),default=0)
    p.add_argument("--limit",type=int,default=299)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    write(args.out,run(args.run,args.limit,args.offset))

"""Measure open-loop actuation on both seats of the seven later Top Five replays."""
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
import gzip
import json
import time

from cash_practice import audit, np
from freight_build import OUTPUT, PARENT


def main(corpus, out):
    manifest = json.loads((corpus / "manifest.json").read_bytes())
    candidate = audit.shared.module(OUTPUT, "freight_practice")
    parent = audit.shared.module(PARENT, "freight_parent")
    rows, changed = [], []
    times = {"candidate": [], "parent": []}
    for game in sorted(manifest["games"], key=lambda g: g["id"]):
        data = (corpus / (game["id"] + ".json.gz")).read_bytes()
        assert sha256(data).hexdigest() == game["sha256"]
        steps = json.loads(gzip.decompress(data))["steps"]
        for seat in (0, 1):
            roots, delta, mid = 0, 0, 0
            for t, frame in enumerate(steps[:-1]):
                obs = dict(step=t, player=seat, players=frame[0]["observation"]["players"],
                           carbon=frame[seat]["observation"]["carbon"])
                if t % 8 != 0 and t != len(steps)-2:
                    for m in (candidate, parent):
                        m._FEAT.observe(obs)
                    continue
                actions = []
                order = [(candidate, "candidate"), (parent, "parent")]
                if (t // 8 + seat) % 2:
                    order.reverse()
                for m, name in order:
                    start = time.perf_counter()
                    actions.append(m.agent(obs))
                    if t:
                        times[name].append((time.perf_counter()-start)*1000)
                roots += 1
                difference = actions[0] != actions[1]
                if difference != bool(candidate._FREIGHT_LAST):
                    raise AssertionError((game["id"], seat, t, difference, candidate._FREIGHT_LAST))
                delta += difference
                mid += difference and t < 270
                if difference:
                    changed.append(dict(id=game["id"], seat=seat, step=t, swaps=candidate._FREIGHT_LAST))
            rows.append(dict(id=game["id"], seat=seat, roots=roots, changed=delta,
                             changed_before_270=mid, counts=dict(candidate._FREIGHT_COUNTS)))
    if not changed:
        raise AssertionError("no observed production actuation")
    result = dict(parent_sha256=audit.digest(PARENT), candidate_sha256=audit.digest(OUTPUT),
                  sampling="All seven later Top Five head-to-head games, each seat, every eighth nonterminal root and last root; intervening feature history is retained.",
                  timing_scope="Single BLAS thread, local shared machine. Wall-clock full calls excluding t=0.",
                  timing_ms={name:dict(p50=float(np.median(v)), p95=float(np.percentile(v,95)),
                                       max=float(np.max(v))) for name,v in times.items()},
                  roots=sum(r["roots"] for r in rows), changed=len(changed),
                  changed_before_270=sum(r["changed_before_270"] for r in rows), rows=rows, changes=changed)
    audit.shared.atomic_json(out, result)
    print(result["roots"],result["changed"],result["changed_before_270"],result["timing_ms"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("corpus", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    main(args.corpus, args.out)

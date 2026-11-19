"""Read back signed target shards and contrast the raw teacher's mode with the submitted command."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
manifest_file = HERE / "manifest.json"
records = json.loads(manifest_file.read_bytes())["records"]
counts = defaultdict(Counter)
maps = defaultdict(set)
role_names = ("collector", "planter", "base")
examples = {}
for r in records:
    path = HERE / r["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == r["sha256"]
    key = f"{r['split']}/{r['domain']}"
    maps[key].add(r["terrain"])
    with np.load(path, allow_pickle=False) as d:
        history = d["prefix_mode"] == 0
        q = d["q"][history]
        legal = d["legal"][history]
        actual = d["src_act"][history]
        kind = d["tok_kind"][history, :12]
        valid = (actual >= 0) & (legal.sum(-1) > 1)
        top = q.argmax(-1)
        live = actual >= 0
        c = counts[key]
        c["historical_roots"] += len(q)
        c["prefix_repaired"] += int(d["prefix_repaired"][history].sum())
        c["forced_single_option"] += int(((legal.sum(-1) == 1) & live).sum())
        c["other_submitted_orders_infeasible"] += int(((~np.take_along_axis(legal,actual.clip(min=0)[...,None],axis=-1)[...,0]) & live).sum())
        for name, select in [("all",valid),("collector",valid & (kind==0)),("planter",valid & (kind==1)),("base",valid & (np.arange(12)>=10))]:
            c[name+"_submitted"] += int(select.sum())
            c[name+"_modal_match"] += int(((top == actual)&select).sum())
        # Deliberately record only fit examples; held decisions do not select a checkpoint or modify inputs.
        if r["domain"] == "original" and r["split"]=="fit":
            for typ, select in [("worker_high_confidence",valid & (np.arange(12)<10)),
                                ("base_high_confidence",valid & (np.arange(12)>=10))]:
                bad = select & (top != actual) & (q.max(-1) > .95)
                if bad.any() and typ not in examples:
                    j,s = np.argwhere(bad)[0]
                    examples[typ] = dict(id=r["id"], source=r["source"], terrain=r["terrain"], seat=r["seat"],
                        step=int(d["step"][history][j]), slot=int(s), q=q[j,s].tolist(), submitted=int(actual[j,s]))
output = {k:{**v,"physical_maps":len(maps[k])} for k,v in sorted(counts.items())}
for d in output.values(): assert sum(d[x+"_submitted"] for x in role_names)==d["all_submitted"]
receipt = dict(manifest_sha256=hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
    split_counts=output, examples=examples, actor_visited_outcomes_are_historical=True,
    denominator="Recorded-prefix roots only, one entry per live unit with at least two student-legal commands. Choice is the mode of the projected teacher prior. These agreements are unit-weighted descriptive counts.")
(HERE/"source_evidence.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")

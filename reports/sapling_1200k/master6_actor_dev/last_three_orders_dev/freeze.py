"""Inventory every complete native game in this frozen checkout with the exact parent parameters."""
from pathlib import Path
import gzip,json,hashlib,subprocess
HERE=Path(__file__).resolve().parent
A=HERE.parent;B=A.parent;ROOT=A.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pack=A/"actor_partner_dev/candidate.py"
p_sha="c5ad3f43959a4812e84e89574ee0c6e69765ebb06a8b3ec93f839b0dde128dd3"
assert sha(pack)==p_sha
snapshot=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
paths=subprocess.check_output(["git","ls-tree","-r","--name-only",snapshot,"reports/sapling_1200k"],text=True).splitlines()
rows=[]
for p in paths:
    f=ROOT/p
    if not f.name.startswith("raw_") or not f.name.endswith(".json.gz"):continue
    raw=json.loads(gzip.decompress(f.read_bytes()))
    if not isinstance(raw,dict) or raw.get("pack_sha256")!=p_sha:continue
    assert raw["examiner_md5"]=="57f9a594cce4f2ec9e3573d8d9d26ca1" and not raw["errors"] and raw["steps"]==len(raw["actions"])==299
    assert raw["configuration"]["_TEMP"]==1 and raw["configuration"]["_SEED"] is None and raw["configuration"]["_S_ENGINE"]
    rows.append(dict(seed=raw["seed"],seat=raw["candidate_seat"],source=p,sha256=sha(f)))
excluded={621704800,621704801,621704802,621704803,621758335}
plan=dict(commit=snapshot,parent_sha256=p_sha,all_archived=len(rows),maps=len(set(r["seed"] for r in rows)),games=rows,
          skip_alternative_seeds=sorted(excluded),scope="All complete original-examiner tapes under reports/sapling_1200k bearing the exact frozen actor; no forced rerolls. For five recent response/fresh maps include nominal reachable cargo summaries only. Exposure list bound before newly generated commands or alternate returns.",
          selector="For each t=296,297,298, try one legal own actor slot override, greedily redo remaining legal prefix and original return checks; reject new own conflicts; compare public-only scenarios: opponent stay, repeat preceding visible movement, return laden cargo by shortest path. Each tested with observed carbon only and own public-history estimate. Score real one-step cash margin plus 0.85 times reachable freight difference only when time remains. Require positive minimum improvement over parent >3 in all scenarios; rank by minimum then mean. Do not treat tree sale or unpaid cargo as score.",
          test="At t=298 compare publicly selected action vs original simultaneous enemy recorded action and exact terminal cash. Enumerated retrospective maximum separately and explicitly not deployable. For earlier roots only report one-step observed-scenario outcomes; no opponent future action tape. Trigger additional same-root native response only if (1) public last-turn selections have strictly positive exact average, change at least 2 independent maps, no criterion changed after scores (2) qualifying early roots on >=2 distinct maps, and at least one where independently measured actual near-loss margin lies within maximum realized last-turn uplift of that map. Otherwise no more compute or fresh match.")
assert len(set((r['seed'],r['seat']) for r in rows))==len(rows)
plan["prior_last_turn_actors_only_comparison_maps"]=list(range(621704800,621704808))
plan["inference_split"]="Among eligible 22 tapes, the four 804..807 maps (8 tapes) had a last-turn two-package comparison before; 7 other maps (14 tapes) did not enter that study. Report these cohorts separately."
out=HERE/"protocol.json"
if not out.exists():out.write_text(json.dumps(plan,indent=2)+'\n')
print(len(rows),plan["maps"],sum(r['seed'] not in excluded for r in rows))

"""GET all v1/v2 public scorecards as of one common cutoff; preserve deltas only in the digest."""
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent
LADDER=HERE.parent
sys.path.insert(0,str(LADDER/"pre_v3_cutoff"))
import cutoff
from cutoff import refresh
import argparse

def run(fresh):
    old=refresh.read(LADDER/"pre_v3_cutoff/summary.json")
    if fresh:
        at=datetime.now(timezone.utc).isoformat()
        # Reuse GET pagination, exact strategy IDs and de-duplication checks.
        cutoff.HERE=HERE
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(cutoff.fetch,cutoff.IDS))
        refresh.store(HERE/"timestamps.json",dict(cutoff_utc=at,completed_utc=datetime.now(timezone.utc).isoformat()))
    cut=refresh.read(HERE/"timestamps.json")
    limit=datetime.fromisoformat(cut["cutoff_utc"]).timestamp()
    results={}
    for name,sid in cutoff.IDS.items():
        refresh.ID=sid
        existing={(c["match"],c["game"]):c for c in old["strategies"][name]["all_games"]}
        pages=refresh.read(HERE/(name+"_pages.json.gz"))
        matches=[m for p in pages for m in p["data"]["matches"]];assert len({m["id"] for m in matches})==len(matches)
        final=[];pending=[]
        for m in matches:
            if m["created"]>limit:continue
            assert sid in (m["a"],m["b"])
            if not m["result"] or not m["finished"] or m["finished"]>limit:
                pending.append({k:m[k] for k in ("id","kind","state","name_a","name_b")});continue
            final.extend(refresh.score(m,g) for g in m["result"])
        keyed={(c["match"],c["game"]):c for c in final}
        assert existing.keys()<=keyed.keys()
        for k,x in existing.items():
            assert all(x[j]==keyed[k][j] for j in ("score","outcome","margin","opponent_id","seat"))
        new=sorted([c for k,c in keyed.items() if k not in existing],key=lambda c:c["finished"])
        results[name]=dict(id=sid,sha256=cutoff.digest((HERE/(name+"_pages.json.gz")).read_bytes()),verified_prior=len(existing),
            counts={kind: dict(Counter(c["outcome"] for c in final if c["kind"]==kind)) for kind in sorted({c["kind"] for c in final})},
            added=new,pending=pending)
        print(name,results[name]["counts"],[(c["opponent"],c["margin"],c["match"]) for c in new],flush=True)
    refresh.store(HERE/"matches.json",dict(**cut,versions=results))
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--fetch",action="store_true");run(parser.parse_args().fetch)

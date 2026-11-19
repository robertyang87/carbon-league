"""Read the added public replays and reconcile tree payments in physical seat order."""
import gzip
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/"all_top5"))
import fetch

def analyze(entry):
    key, item = entry
    card = item["card"]
    cache = fetch.CACHE/"replays"/(key+".json.gz")
    url = fetch.SERVER+"/replay/"+card["id"]+"/"+str(card["game"])
    try:
        raw = cache.read_bytes() if cache.exists() else fetch.Retrieval({},0).get(url)
        env, _, _ = fetch.extract(raw)
        verified = fetch.validate(env,card)
        ledger = {}
        for t, pair in enumerate(env["steps"]):
            ob = pair[0]["observation"]
            for seat, player in enumerate(ob["players"]):
                for tid, (pos, age) in player[3].items():
                    tree = ledger.setdefault(tid,dict(origin=seat,site=pos,spans=[]))
                    spans = tree["spans"]
                    if not spans or spans[-1]["seat"] != seat:
                        spans.append(dict(seat=seat,start=t,end=t,income=0.0))
                    spans[-1]["end"] = t
                    spans[-1]["income"] += ob["trees"][tid][1]
        return dict(id=key,verified=verified,url=url,canonical_sha256=fetch.sha(fetch.canonical(env)),
                    terrain=fetch.sha(bytes(env["steps"][0][0]["observation"]["full_carbon"])),trees=ledger)
    except Exception as exc:
        return dict(id=key,url=url,error=repr(exc))
if __name__ == "__main__":
    cards=json.loads((HERE/"ladder_index_delta.json").read_bytes())["cards"]
    with ThreadPoolExecutor(2) as pool:
        rows=list(pool.map(analyze,sorted(cards.items())))
    (HERE/"ladder_trees.json.gz").write_bytes(gzip.compress(fetch.canonical(rows),mtime=0))
    for r in rows: print(r["id"],len(r.get("trees",{})),r.get("error"))

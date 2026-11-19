"""Continue the prior index-based ladder intake without re-reading the preceding seven games."""
import sys
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import ladder_delta as old
A=old.B
B='595633d75'
paths=[p for p in old.fetch.git('diff','--name-only',A,B,'--','data').decode().splitlines() if old.fetch.sourcefile(p)]
prior={};now={}
for p in paths:
    prior.update(old.cards(A,p));now.update(old.cards(B,p))
known={g["id"] for g in json.loads((HERE.parent/"ladder_delta.json").read_bytes())["games"]}
top=HERE.parents[2]/"all_top5"
for manifest in [top/"manifest.json",*sorted(top.glob("appendix_*/manifest.json"))]:
    known.update(g["id"] for g in json.loads(manifest.read_bytes())["games"])
delta=now.keys()-prior.keys()
with ThreadPoolExecutor(2) as pool:
    games=list(pool.map(old.recover,(now[k] for k in sorted(delta-known))))
(HERE/'ladder_delta.json').write_bytes(old.fetch.canonical(dict(base=A,end=B,indices=paths,already_listed=sorted(known&delta),games=games)))
print(len(games))

"""De-duplicate later edits to other ladder indices against the entire index union at the prior revision."""
from intake import HERE,ladder
import json
from concurrent.futures import ThreadPoolExecutor
start='f1c081756';end='b0eafa4da'
paths=[p for p in ladder.fetch.git('diff','--name-only',start,end,'--','data').decode().splitlines() if ladder.fetch.sourcefile(p)]
current={}
for path in paths:current.update(ladder.cards(end,path))
all_paths=[p for p in ladder.fetch.git('ls-tree','-r','--name-only',start,'data').decode().splitlines() if ladder.fetch.sourcefile(p)]
with ThreadPoolExecutor(4) as pool:
    for old in pool.map(lambda p:ladder.cards(start,p),all_paths):
        for key in old:current.pop(key,None)
with ThreadPoolExecutor(2) as pool: rows=list(pool.map(ladder.recover,current.values()))
(HERE/'additional_indices.json').write_bytes(ladder.fetch.canonical(dict(start=start,end=end,changed=paths,prior_indices=len(all_paths),games=rows)))
print(len(rows))

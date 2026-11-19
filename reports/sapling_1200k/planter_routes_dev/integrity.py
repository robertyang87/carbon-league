"""Cross-check terrain / participant splits and the source identities for credited purchases."""
import gzip
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
CACHE=Path('/workspace/sapling-all-top5-20261119')
def sha(data):return hashlib.sha256(data).hexdigest()
def terrain(g):
    with gzip.open(CACHE/g['path']) as f:
        env=json.load(f)
    return sha(bytes(env['steps'][0][0]['observation']['full_carbon']))
if __name__=='__main__':
    split=json.loads((HERE/'split.json').read_bytes());man=json.loads((HERE.parent/'all_top5/manifest.json').read_bytes())
    ex=json.loads((HERE.parent/'latest5_inventory.json').read_bytes())['games'];games={g['id']:g for g in man['games']}
    extractor=sha((HERE.parent/'all_top5/plant_timing_extract.py').read_bytes());groups=defaultdict(dict);opps={};overlaps=[]
    for s,rows in split.items():
        opps[s]=set()
        for r in rows:
            with gzip.open(CACHE/'plant_work'/(r['id']+'.gz')) as f: ledger=json.load(f)
            if ledger['extractor_sha256']!=extractor:raise ValueError('receipt extractor identity mismatch')
            groups[s][r['id']]=terrain(r)
            for n,i in zip(r['seat_names'],r['seat_strategy_ids']):
                if i is None:opps[s].add(n)
    excluded={}
    for g in ex:
        excluded[g['id']]=terrain(games[g['id']])
    official=json.loads((HERE/'parent/official_replay.json').read_bytes())['runs']['parent']['games'][0]['map_sha256']
    for s,rs in groups.items():
        a=set(rs.values())
        if len(a)!=len(rs) or a&set(excluded.values()) or official in a:raise ValueError('map reused')
        for other,ls in groups.items():
            if other!=s and a&set(ls.values()):raise ValueError('map shared among groups')
    unseen_names=sorted(opps['test_unseen_opponent']-opps['train'])
    if not all(any(n in unseen_names for n in g['seat_names']) for g in split['test_unseen_opponent']):raise ValueError('no unseen opponent')
    d={'maps_sha256_by_split':groups,'excluded_old26':excluded,'exam_terrain':official,'unseen_names':unseen_names,
       'receipt_extractor':extractor,'source_manifest_sha256':sha((HERE.parent/'all_top5/manifest.json').read_bytes())}
    (HERE/'integrity.json').write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')

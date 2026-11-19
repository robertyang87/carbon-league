"""Keep the later Fablev8 score upload separate from the previous four Master5s games."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
TOP=HERE.parent
sys.path.insert(0,str(TOP))
import fetch
from appendix_batch import prepare
from validation_delta import run as validate
from appendix_f7109df6.collect import BATCHES as EARLIER, REV as PREVIOUS
BATCHES=(*EARLIER,'appendix_f7109df6')
REV='1fdacb1b1fd8bd7054006d6be1a054039c463495'
SOURCE='data/fablev8_matches.json'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=('collect','analyze'))
    arg=ap.parse_args()
    manifests=[TOP/b/'manifest.json' for b in BATCHES]
    known={g['id'] for p in manifests for g in json.loads(p.read_bytes())['games']}
    if len(known)!=1406:raise ValueError('four-game prior batch missing')
    def score(rev):
        return {fetch.key_of(r):r for r in fetch.walk_matches(json.loads(fetch.git('show',rev+':'+SOURCE))) if r.get('state')=='done'}
    left,right=score(PREVIOUS),score(REV)
    extras=sorted(right.keys()-left.keys())
    assert len(extras)==2 and not left.keys()-right.keys()
    assert all(fetch.identity(left[k])==fetch.identity(right[k]) for k in left)
    changes=fetch.git('diff','--name-only',PREVIOUS,REV,'--','data').decode().splitlines()
    assert [s for s in changes if fetch.sourcefile(s)]==[SOURCE]
    scope=[]
    for gid in extras:
        r=right[gid];names=[r['name_'+s] for s in r['order']]
        scope.append({'id':gid,'physical_seats':names,'score_by_seat':[r['scores'][s] for s in r['order']],
                      'eligible':bool(set(names)&set(fetch.NAMES)),'previously_archived':gid in known})
    new={r['id'] for r in scope if r['eligible'] and not r['previously_archived']}
    assert new=={'7753acfa1a29461c3954ace4d9d541aa-1'}
    delta={'previous_score_revision':PREVIOUS,'revision':REV,'source':SOURCE,'old_reference_count':len(left),
           'new_reference_count':len(right),'previous_distinct_games':len(known),'new_distinct_games':1,
           'existing_references':0,'non_top_five_refs':1,'union_games':len(known)+1,'added':scope}
    fetch.atomic(HERE/'score_delta.json',(json.dumps(delta,ensure_ascii=False,indent=2)+'\n').encode())
    if arg.stage=='collect':
        prepare(REV,manifests,HERE,expected=(1,0));return
    parents=[(b,hashlib.sha256(fetch.git('show',REV+':'+str((TOP/b/'validation_maps.jsonl.gz').relative_to(fetch.ROOT)))).hexdigest()) for b in EARLIER]
    previous=json.loads((TOP/BATCHES[-1]/'validation.json').read_bytes())
    if previous['revision']!=PREVIOUS or previous['union_distinct']['id']!=1406:raise ValueError('previous evidence changed')
    parents.append((BATCHES[-1],previous['map_registry_sha256']))
    validate(fetch.CACHE,HERE,REV,PREVIOUS,parents)
    def run(script,*args):
        subprocess.run([sys.executable,str(script),*map(str,args)],check=True)
    m=HERE/'manifest.json'
    run(TOP/'appendix_499886f1/cases.py','--manifest',m,'--cache',fetch.CACHE)
    run(TOP/'appendix_0137949d/cash.py','--manifest',m,'--cache',fetch.CACHE)
    idx=TOP/f'context_{HERE.name}_index.tsv.gz'
    run(TOP/'context_catalog.py','--manifest',m,'--output',idx)
    run(TOP/'context_audit.py','--index',idx,'--validation',HERE/'validation_games.jsonl.gz','--out',TOP/f'context_{HERE.name}_receipt.json','--jobs','1')

if __name__=='__main__':
    main()

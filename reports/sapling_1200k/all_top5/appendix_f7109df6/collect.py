"""Seal only the new complete Master5s games at the registered score cut."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
TOP=HERE.parent
sys.path.insert(0,str(TOP))
from appendix_batch import prepare
import fetch
from validation_delta import run as validate

PREVIOUS='41e4177eaaa0ca5c808e28a5a4290b5a88355205'
REV='f7109df6568f44ec5dd88635843585213a792b42'
BATCHES=('.', 'appendix_499886f1','appendix_4bfdabf5','appendix_555c0c56','appendix_bf049043',
         'appendix_bd3c0021','appendix_5484590b','appendix_99919cbf','appendix_0137949d','appendix_41e4177e')
SCORES='data/master5s_matches.json'


def source_rows(revision):
    data=json.loads(fetch.git('show',revision+':'+SCORES))
    return {fetch.key_of(g):g for g in fetch.walk_matches(data) if g.get('state')=='done' and
            {g['name_a'],g['name_b']}&set(fetch.NAMES)}


def census():
    manifests=[TOP/b/'manifest.json' for b in BATCHES]
    old={g['id']:(b,g) for b,p in zip(BATCHES,manifests) for g in json.loads(p.read_bytes())['games']}
    old_index,current=source_rows(PREVIOUS),source_rows(REV)
    assert len(old)==1402 and set(old_index)<=set(current)
    assert all(fetch.identity(old_index[k])==fetch.identity(current[k]) for k in old_index)
    changes=fetch.git('diff','--name-only',PREVIOUS,REV,'--','data').decode().splitlines()
    assert [s for s in changes if fetch.sourcefile(s)]==[SCORES]
    new=sorted(current.keys()-old_index.keys())
    assert len(new)==5
    rows=[]
    for gid in new:
        r=current[gid]
        if gid in old and fetch.identity(r)!=fetch.identity(old[gid][1]['card']):
            raise ValueError('existing score identity changed')
        rows.append({'id':gid,'physical_seats':[r['name_'+s] for s in r['order']],
                     'score_by_seat':[r['scores'][s] for s in r['order']],
                     'previous_batch':old[gid][0] if gid in old else None})
    prior=set(old)
    doc={'previous_score_revision':PREVIOUS,'revision':REV,'source':SCORES,'old_reference_count':len(old_index),
         'new_reference_count':len(current),'previous_distinct_games':len(prior),'new_distinct_games':len(set(new)-prior),
         'existing_references':len(set(new)&prior),'union_games':len(prior|set(new)),'added':rows}
    assert (doc['new_distinct_games'],doc['existing_references'])==(4,1)
    fetch.atomic(HERE/'score_delta.json',(json.dumps(doc,ensure_ascii=False,indent=2)+'\n').encode())
    return manifests,doc


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=('collect','analyze'))
    args=ap.parse_args()
    manifests,delta=census()
    if args.stage=='collect':
        prepare(REV,manifests,HERE,expected=(4,0))
        return
    m=HERE/'manifest.json'
    assert {g['id'] for g in json.loads(m.read_bytes())['games']}=={g['id'] for g in delta['added'] if g['previous_batch'] is None}
    parents=[(b,hashlib.sha256(fetch.git('show',REV+':'+str((TOP/b/'validation_maps.jsonl.gz').relative_to(fetch.ROOT)))).hexdigest()) for b in BATCHES]
    validate(fetch.CACHE,HERE,REV,PREVIOUS,parents)
    def run(src,*args):
        subprocess.run([sys.executable,str(src),*map(str,args)],check=True)
    run(TOP/'appendix_499886f1/cases.py','--manifest',m,'--cache',fetch.CACHE)
    run(TOP/'appendix_0137949d/cash.py','--manifest',m,'--cache',fetch.CACHE)
    idx=TOP/f'context_{HERE.name}_index.tsv.gz'
    receipt=TOP/f'context_{HERE.name}_receipt.json'
    run(TOP/'context_catalog.py','--manifest',m,'--output',idx)
    run(TOP/'context_audit.py','--index',idx,'--validation',HERE/'validation_games.jsonl.gz','--out',receipt,'--jobs','1')


if __name__=='__main__':
    main()

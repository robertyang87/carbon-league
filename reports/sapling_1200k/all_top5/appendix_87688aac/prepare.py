"""Rebuild both new authenticated Top Five replays and their per-seat public decisions."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

HERE=Path(__file__).resolve().parent
TOP=HERE.parent
sys.path.insert(0,str(TOP))
import fetch
from appendix_628e28d0.archive import prepare
from appendix_97de1ebc.collect import inventories
from continue_appendix import check_archive,previous_manifests
from validation_delta import run as validate
from appendix_628e28d0 import horizons
import report

REV='87688aac9aaf5d0bf5c9f59b420617a15a2bf850'
PREV='628e28d06b0610f6ceebdac3cc632057766bddd2'
GAMES={'4a8912fffc316669d63682a50de1f7b1-1','c94ce6f5882f7141736fc65769d45fdf-1'}

def go(cache,warm):
    prior=previous_manifests(TOP/'appendix_628e28d0')
    names=[str(p.parent.relative_to(TOP)) for p in prior]
    delta=inventories(PREV,REV,names)
    if {k for k,v in delta['all_added_references_by_id'].items() if v['class']=='top'}!=GAMES or delta['new_count']!=2 or delta['unscored_new_ids'] or delta['new_mutual']:
        raise ValueError('New physical match identities changed')
    fetch.dump(HERE/'score_delta.json',delta);fetch.CACHE=cache
    manifest=HERE/'manifest.json'
    if manifest.is_file():
        older=fetch.load(manifest);check_archive(older)
        with tarfile.open(HERE/'replays.tar') as archive:
            for game in older['games']:
                fetch.atomic(cache/game['path'],archive.extractfile(game['path']).read())
                keys=('id','revision','source','sha256','canonical_sha256','seed','environment_uuid','transitions','terminal','clean_terminal','source_attempt_errors','frame_verified')
                fetch.dump(cache/'receipts'/(game['id']+'.json'),{k:game[k] for k in keys})
    elif warm:
        for k in GAMES:
            # The physical-map clearance retrieved these same full official responses.
            data,_,_=fetch.extract((warm/(k+'.bin')).read_bytes())
            fetch.atomic(cache/'replays'/(k+'.json.gz'),gzip.compress(fetch.canonical(data),compresslevel=4,mtime=0))
    faults=[p.parent/'unscored/manifest.json' for p in prior if (p.parent/'unscored/manifest.json').is_file()]
    found={}
    prepare(REV,prior,HERE,(2,0),unscored=found)
    if set(found)!={g['id'] for p in faults for g in fetch.load(p)['games']}:raise ValueError('Unscored roster differs')
    current=fetch.load(manifest);check_archive(current)
    if {g['id'] for g in current['games']}!=GAMES or current['accumulated_games']!=1638:raise ValueError('Full census differs')
    ledgers=[]
    for n,p in zip(names,prior):
        f=p.parent/'validation_maps.jsonl.gz';data=fetch.git('show',REV+':'+str(f.relative_to(fetch.ROOT)))
        if data!=f.read_bytes():raise ValueError(n+' historical ledger changed')
        ledgers.append((n,fetch.sha(data)))
    validate(cache,HERE,REV,PREV,ledgers,previous_fault_ledgers=faults)
    def run(script,*args):subprocess.run([sys.executable,str(TOP/script),*map(str,args)],check=True)
    run('appendix_499886f1/cases.py','--manifest',manifest,'--cache',cache)
    run('appendix_0137949d/cash.py','--manifest',manifest,'--cache',cache)
    run('turn_action_windows.py',HERE,'--cache',cache)
    run('context_catalog.py','--manifest',manifest,'--output',HERE/'context_index.tsv.gz')
    run('context_audit.py','--index',HERE/'context_index.tsv.gz','--validation',HERE/'validation_games.jsonl.gz','--out',HERE/'context_receipt.json','--jobs','1')
    horizons.HERE=HERE;horizons.main()
    hf=HERE/'future_economy_check.json';labels=fetch.load(hf)
    labels['scope']='Both physical seats of two new official replays, with full and censored 10/30-turn outcomes distinguished.'
    fetch.dump(hf,labels)
    report.build()
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-public-response-cache',type=Path)
    opt=parser.parse_args()
    with TemporaryDirectory(prefix='top5-87688aac-') as cache:go(Path(cache),opt.previous_public_response_cache)

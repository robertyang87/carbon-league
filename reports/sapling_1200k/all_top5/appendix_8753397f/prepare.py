"""Rebuild new authenticated Top Five replays and their per-seat public decisions."""
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

REV='8753397f2289fcdf6c879b177573cf33f3349ee9'
PREV='87688aac9aaf5d0bf5c9f59b420617a15a2bf850'
GAMES={'3f03828a6fbb0bba1de8f139a92dc8f8-1','41dfdb005a732fdb5bfd2f9fae11ebd0-1','494137282f2dda7069f925855ac71b11-1','a68e30daa450c29f4b5d1054eeb89d66-1','b94c2df567830f5577abba35aef41028-1','c19209c05361fa99206782f89178c88f-1'}

def go(cache,warm):
    prior=previous_manifests(TOP/'appendix_87688aac')
    names=[str(p.parent.relative_to(TOP)) for p in prior]
    delta=inventories(PREV,REV,names)
    if {k for k,v in delta['all_added_references_by_id'].items() if v['class']=='top'}!=GAMES or delta['new_count']!=len(GAMES) or delta['unscored_new_ids'] or delta['new_mutual']:
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
    prepare(REV,prior,HERE,(len(GAMES),0),unscored=found)
    if set(found)!={g['id'] for p in faults for g in fetch.load(p)['games']}:raise ValueError('Unscored roster differs')
    current=fetch.load(manifest);check_archive(current)
    if {g['id'] for g in current['games']}!=GAMES or current['accumulated_games']!=1644:raise ValueError('Full census differs')
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
    labels['scope']='Both physical seats of six new official replays, with full and censored 10/30-turn outcomes distinguished.'
    fetch.dump(hf,labels)
    report.build()
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-public-response-cache',type=Path)
    opt=parser.parse_args()
    with TemporaryDirectory(prefix='top5-8753397f-') as cache:go(Path(cache),opt.previous_public_response_cache)

"""Build the single post-8753397f game, keeping the score cutoff separate from archive publication."""
import argparse
import gzip
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import tarfile
import subprocess

HERE=Path(__file__).resolve().parent
TOP=HERE.parent
sys.path.insert(0,str(TOP))
import fetch
from appendix_628e28d0.archive import prepare
from appendix_628e28d0 import horizons
from appendix_97de1ebc.collect import inventories
from continue_appendix import check_archive,compare_boards,previous_manifests
from validation_delta import run as validate

BASE='8753397f2289fcdf6c879b177573cf33f3349ee9'
REV='c8236c8c6d7264352a84ceb0ca6522e74dd5acc7'
GID='59276abc83c340ebc786f531e5a2430d-1'
# The prior six-score appendix was physically checked at BASE and published after this score cutoff.
SIX_LEDGER='7d11f3c597db52f3e98afd9dd31ba9bc543a864346698b76e4653bae4ce5c714'

def run(cache,warm=None):
    prior=previous_manifests(TOP/'appendix_8753397f')
    names=[str(p.parent.relative_to(TOP)) for p in prior]
    delta=inventories(BASE,REV,names)
    assert [i for i,r in delta['all_added_references_by_id'].items() if r['class']=='top']==[GID]
    assert delta['new_count']==1 and not delta['unscored_new_ids'] and delta['new_mutual']==0
    fetch.dump(HERE/'score_delta.json',delta);fetch.CACHE=cache
    manifest=HERE/'manifest.json'
    if manifest.exists():
        old=fetch.load(manifest);check_archive(old)
        with tarfile.open(HERE/'replays.tar') as archive:
            for g in old['games']:
                fetch.atomic(cache/g['path'],archive.extractfile(g['path']).read())
                keys=('id','revision','source','sha256','canonical_sha256','seed','environment_uuid','transitions','terminal','clean_terminal','source_attempt_errors','frame_verified')
                fetch.dump(cache/'receipts'/(g['id']+'.json'),{k:g[k] for k in keys})
    elif warm:
        data,_,_=fetch.extract(warm.read_bytes())
        fetch.atomic(cache/'replays'/(GID+'.json.gz'),gzip.compress(fetch.canonical(data),compresslevel=4,mtime=0))
    faults=[p.parent/'unscored/manifest.json' for p in prior if (p.parent/'unscored/manifest.json').exists()]
    registered={};prepare(REV,prior,HERE,(1,0),unscored=registered)
    current=fetch.load(manifest);check_archive(current)
    assert [g['id'] for g in current['games']]==[GID] and current['accumulated_games']==1645
    assert set(registered)=={g['id'] for p in faults for g in fetch.load(p)['games']}
    parents=[]
    for path,name in zip(prior,names):
        ledger=path.parent/'validation_maps.jsonl.gz'
        if path==prior[-1]:expected=SIX_LEDGER
        else:expected=fetch.sha(fetch.git('show',REV+':'+str(ledger.relative_to(fetch.ROOT))))
        assert fetch.sha(ledger.read_bytes())==expected
        parents.append((name,expected))
    validate(cache,HERE,REV,BASE,parents,previous_fault_ledgers=faults)
    def invoke(file,*args):
        subprocess.run([sys.executable,str(TOP/file),*map(str,args)],check=True)
    invoke('appendix_499886f1/cases.py','--manifest',manifest,'--cache',cache)
    invoke('appendix_0137949d/cash.py','--manifest',manifest,'--cache',cache)
    invoke('turn_action_windows.py',HERE,'--cache',cache)
    invoke('context_catalog.py','--manifest',manifest,'--output',HERE/'context_index.tsv.gz')
    invoke('context_audit.py','--index',HERE/'context_index.tsv.gz','--validation',HERE/'validation_games.jsonl.gz','--out',HERE/'context_receipt.json','--jobs','1')
    horizons.HERE=HERE;horizons.main()
    labels=fetch.load(HERE/'future_economy_check.json');labels['scope']='Both seats in this one new game; 10/30-turn outcomes separate from public input and right-censored outcomes.'
    fetch.dump(HERE/'future_economy_check.json',labels)
    compare_boards(prior,HERE,current,faults)
    invoke('appendix_c8236c8c/report.py')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--official-replay-cache',type=Path)
    opt=p.parse_args()
    with TemporaryDirectory(prefix='top5-c8236c8c-') as cache:run(Path(cache),opt.official_replay_cache)

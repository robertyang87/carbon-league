"""Independently resolve added index references against the older whole-repository score union."""
import gzip
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

HERE=Path(__file__).resolve().parent;TOP=HERE.parent
sys.path.insert(0,str(TOP))
import fetch
from appendix_32b555ee.collect import scores
from appendix_97de1ebc.collect import inventories
from appendix_628e28d0.archive import prepare
from appendix_628e28d0 import horizons
from continue_appendix import check_archive,compare_boards,previous_manifests
from validation_delta import run as validate
OLD='c8236c8c6d7264352a84ceb0ca6522e74dd5acc7'
REV='ea2249324c75a820c983b56a2557278ac703364d'
GID='257964777b5dab551f8600956952127a-1'


def resolve(delta):
    tree=fetch.git('ls-tree','-r',OLD).decode().splitlines()
    blobs={s.split('\t',1)[0].split()[2] for s in tree if fetch.sourcefile(s.split('\t',1)[1])}
    raw=io.BytesIO(subprocess.check_output(['git','-C',str(fetch.ROOT),'cat-file','--batch'],input=('\n'.join(sorted(blobs))+'\n').encode()))
    before=set()
    for oid in sorted(blobs):
        so,kind,n=raw.readline().decode().split()
        assert so==oid and kind=='blob'
        before.update(scores(raw.read(int(n))));assert raw.read(1)==b'\n'
    ann={gid:dict(previous_score_anywhere=gid in before,**card) for gid,card in delta['all_added_references_by_id'].items()}
    frozen={}
    for p in fetch.git('diff','--name-only',OLD,REV).decode().splitlines():
        if not p.startswith('data/') or '_corpus/' not in p or not p.endswith('.json.gz'): continue
        gid=Path(p).name.removesuffix('.json.gz');frozen[p]=dict(id=gid,previous_score_anywhere=gid in before,current_new_reference=gid in ann,
            blob=fetch.git('rev-parse',REV+':'+p).decode().strip(),is_new_top5=gid==GID)
    assert GID not in before and sum(r['class']=='top' for r in ann.values())==1
    return dict(older_distinct_score_ids=len(before),distinct_new_reference_ids=len(ann),
                globally_new_score_ids=[k for k,v in ann.items() if not v['previous_score_anywhere']],
                prior_score_mirror_ids=[k for k,v in ann.items() if v['previous_score_anywhere']],added_references=ann,
                added_raw_corpus_files=frozen,new_top5_ids=[GID])


def process(cache):
    prior=previous_manifests(TOP/'appendix_c8236c8c');names=[str(p.parent.relative_to(TOP)) for p in prior]
    delta=inventories(OLD,REV,names);assert {k for k,v in delta['all_added_references_by_id'].items() if v['class']=='top'}=={GID}
    assert delta['new_count']==1 and not delta['new_mutual'] and not delta['unscored_new_ids']
    fetch.dump(HERE/'score_delta.json',delta);fetch.dump(HERE/'reference_audit.json',resolve(delta))
    fetch.CACHE=cache; mf=HERE/'manifest.json'
    if mf.exists():
        old=fetch.load(mf);check_archive(old)
        with tarfile.open(HERE/'replays.tar') as ar:
            for g in old['games']:
                fetch.atomic(cache/g['path'],ar.extractfile(g['path']).read())
                keys=('id','revision','source','sha256','canonical_sha256','seed','environment_uuid','transitions','terminal','clean_terminal','source_attempt_errors','frame_verified')
                fetch.dump(cache/'receipts'/(g['id']+'.json'),{k:g[k] for k in keys})
    faults=[p.parent/'unscored/manifest.json' for p in prior if (p.parent/'unscored/manifest.json').exists()]
    incomplete={};prepare(REV,prior,HERE,(1,0),unscored=incomplete)
    m=fetch.load(mf);check_archive(m);assert m['accumulated_games']==1646 and [i['id'] for i in m['games']]==[GID]
    assert set(incomplete)=={g['id'] for m in faults for g in fetch.load(m)['games']}
    parents=[]
    for n,p in zip(names,prior):
        f=p.parent/'validation_maps.jsonl.gz';raw=fetch.git('show',REV+':'+str(f.relative_to(fetch.ROOT)))
        assert raw==f.read_bytes();parents.append((n,fetch.sha(raw)))
    validate(cache,HERE,REV,OLD,parents,previous_fault_ledgers=faults)
    def invoke(p,*args): subprocess.run([sys.executable,str(TOP/p),*map(str,args)],check=True)
    invoke('appendix_499886f1/cases.py','--manifest',mf,'--cache',cache)
    invoke('appendix_0137949d/cash.py','--manifest',mf,'--cache',cache)
    invoke('turn_action_windows.py',HERE,'--cache',cache)
    invoke('context_catalog.py','--manifest',mf,'--output',HERE/'context_index.tsv.gz')
    invoke('context_audit.py','--index',HERE/'context_index.tsv.gz','--validation',HERE/'validation_games.jsonl.gz','--out',HERE/'context_receipt.json','--jobs','1')
    horizons.HERE=HERE;horizons.main();hf=HERE/'future_economy_check.json';h=fetch.load(hf)
    h['scope']='Only this new official game, both seats, labels separated by future horizon and censoring.';fetch.dump(hf,h)
    compare_boards(prior,HERE,m,faults);invoke('appendix_ea224932/report.py')
if __name__=='__main__':
    with TemporaryDirectory(prefix='league-ea224932-') as temp:process(Path(temp))

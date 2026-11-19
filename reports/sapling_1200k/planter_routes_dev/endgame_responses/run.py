"""Paired fresh opponent streams on both registered roots, using the tested native responder."""
import gzip,json,os,subprocess,sys
from pathlib import Path
from select_roots import HERE,BASE,ROOT,B,tape,read,write
sys.path.insert(0,str(BASE/'root_recovery_branches'))
import live

PACK=ROOT/'data/weights/sapling_1200k/planter_routes_it50_v3.py';PLAN=read(HERE/'protocol.json');sha=PLAN['candidate_sha256']
def configure():
    live.PACK=PACK;live.PACK_HASH=sha
    if live.digest(PACK.read_bytes())!=sha:raise ValueError('different candidate')
def main():
    configure();doc=read(HERE/'roots.json');raw=json.loads(gzip.decompress(tape.read_bytes()))
    assert raw['pack_sha256']==sha and live.digest(tape.read_bytes())==doc['raw_sha256']
    if len(sys.argv)>1:
        family,rep,arm=sys.argv[1:4];where=HERE/'arms'/(family+'_'+rep)
        cfg=read(where/'live_protocol.json')
        if cfg['runtime_sha256']!=live.digest(Path(live.__file__).read_bytes()):raise ValueError('native responder changed')
        live.run(arm,where,cfg,resume=True);return
    for family,r in doc['selected'].items():
        for rep in range(4):
            where=HERE/'arms'/f'{family}_{rep}';where.mkdir(parents=True,exist_ok=True)
            case=dict(seed=raw['seed'],seat=raw['candidate_seat'],family=family,root_t=r['t'],
                      source_dir=str(B.relative_to(ROOT)),record=dict(seed=raw['seed'],seat=raw['candidate_seat'],source=str(tape.relative_to(ROOT)),source_sha256=doc['raw_sha256']),
                      root={**r,'alternative':r['replacement'],'tree':'no-target-tree'},candidate_hash=sha,
                      root_registry_sha256=live.digest((HERE/'roots.json').read_bytes()),runtime_sha256=live.digest(Path(live.__file__).read_bytes()))
            path=where/'live_protocol.json'
            if path.exists():
                if read(path)!=case:raise ValueError('changed protocol')
            else:live.save(path,case)
            live.prepare(where)
    for family in doc['selected']:
        for rep in range(4):
            jobs=[]
            for i,arm in enumerate(('parent','alternative')):
                # Swap physical CPU assignments on alternating replicate.
                cpu=PLAN['cpus'][(i+rep)%2]
                jobs.append(subprocess.Popen(['taskset','-c',str(cpu),sys.executable,str(Path(__file__).resolve()),family,str(rep),arm],env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')))
            statuses=[child.wait() for child in jobs]
            if statuses!=[0,0]:raise RuntimeError((family,rep,statuses))
if __name__=='__main__':main()

from pathlib import Path
import json,sys,hashlib
HERE=Path(__file__).resolve().parent
TOP=HERE.parent
sys.path.insert(0,str(TOP))
import appendix_batch,fetch,validation_delta
if __name__=='__main__':
    rev=fetch.git('rev-parse','afc3951b2').decode().strip()
    parents=[TOP/'manifest.json',*sorted(p for p in TOP.glob('appendix_*/manifest.json') if p.parent!=HERE and (p.parent/'validation.json').exists())]
    latest=max(parents,key=lambda p:fetch.load(p).get('accumulated_games',0))
    faults=list(TOP.glob('appendix_*/unscored/manifest.json'))
    excluded={};appendix_batch.prepare(rev,parents,HERE,unscored=excluded)
    assert set(excluded)=={g['id'] for p in faults for g in fetch.load(p)['games']}
    ledger=[(str(p.parent.relative_to(TOP)),fetch.sha((p.parent/'validation_maps.jsonl.gz').read_bytes())) for p in parents]
    validation_delta.run(fetch.CACHE,HERE,rev,fetch.load(latest)['revision'],ledger,previous_fault_ledgers=faults)

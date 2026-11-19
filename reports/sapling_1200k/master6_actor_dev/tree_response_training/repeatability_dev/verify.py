"""Read sealed streams and independently replay all four fresh pairs plus both originals, with no native opponent call."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import run


def existing(path,value):
    if not path.is_file() or run.load(path)!=value:
        raise ValueError(f'missing or different immutable repeatability receipt: {path}')


def audit():
    run.bind()
    job=run.load(run.D/'execution.json')
    if job['run_sha256']!=run.sha(Path(run.__file__)) or job['authority_sha256']!=run.sha(Path(run.authority.__file__)):
        raise ValueError('different original generator or official income checker')
    receipts={p:(run.sha(p),p.stat().st_mtime_ns) for p in run.D.rglob('*') if p.is_file() and p.suffix in {'.py','.json','.gz'} and p.name!='verify_receipt.json' and '__pycache__' not in p.parts}
    for i,entry in enumerate(job['jobs']):
        pick,r,old,wd,seed=run.group(i)
        for name,h in pick['old_files'].items():
            f=old/name
            if run.sha(f)!=h:raise ValueError('original historical branch changed')
            receipts[f]=(h,f.stat().st_mtime_ns)
        if [run.sha(wd/f) for f in ('live_protocol.json','root.json.gz','live_shared_init.json')]!=[entry[k] for k in ('case_sha256','root_sha256','stream_sha256')]:
            raise ValueError('fresh paired stream inputs differ from registration')
    previous=run.retain
    try:
        run.retain=existing
        parts=[run.audited(i) for i in range(len(job['jobs']))]
        with redirect_stdout(StringIO()):run.signed_report(parts)
    finally:
        run.retain=previous
    if any((run.sha(p),p.stat().st_mtime_ns)!=stamp for p,stamp in receipts.items()):raise ValueError('audit mutated evidence')
    return {'new_original_calls':0,'new_sample_groups':0,'new_groups_verified':len(parts),'original_groups_verified':2,
            'preserved_files_and_mtimes':len(receipts),'results_sha256':run.sha(run.D/'results.json')}


if __name__=='__main__':print(json.dumps(audit(),sort_keys=True))

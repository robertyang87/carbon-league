"""Restore the selected compressed teacher replays from repository archives, verifying each payload."""
from collections import defaultdict
from contextlib import ExitStack
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'all_top5'))
from context import git_blob

def restore(target, diagnostic=False):
    target.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((HERE/'data_split.json').read_bytes())['games']
    chosen={r['id']:r for r in frozen if r['split']!='excluded'}
    if diagnostic:
        extra=json.loads((HERE/'data_loss_split.json').read_bytes())['games']
        chosen.update({r['id']:r for r in extra})
    groups=defaultdict(list)
    for name in sorted({r['batch'] for r in chosen.values()}):
        for r in json.loads((HERE.parent/'all_top5'/name/'manifest.json').read_bytes())['games']:
            if r['id'] not in chosen:continue
            loc=r['archive_location'];source=loc.get('archive') or loc['repo_path']
            if Path(source).is_absolute() or '..' in Path(source).parts:raise ValueError('archive path')
            dst=target/(r['id']+'.json.gz')
            if dst.exists():
                if hashlib.sha256(dst.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('stale replay '+r['id'])
                continue
            groups[source].append((r['id'],loc.get('member'),r['sha256']))
    for source,rows in groups.items():
        with ExitStack() as stack:
            src=ROOT/source
            stream=stack.enter_context(src.open('rb')) if src.exists() else io.BytesIO(git_blob(ROOT,'HEAD:'+source))
            arc=stack.enter_context(tarfile.open(fileobj=stream,mode='r:*')) if rows[0][1] else None
            for gid,member,sha in rows:
                if arc:
                    info=arc.getmember(member)
                    if not info.isfile():raise ValueError('unexpected archive entry')
                    data=arc.extractfile(info).read()
                else:stream.seek(0);data=stream.read()
                if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('archive hash '+gid)
                dest=target/(gid+'.json.gz');tmp=dest.with_suffix('.part');tmp.write_bytes(data);tmp.replace(dest)
    print('verified',len(chosen),'map payloads in',target)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('/workspace/sapling-all-top5-20261119/replays'))
    p.add_argument('--include-diagnostic',action='store_true');args=p.parse_args()
    restore(args.out,args.include_diagnostic)

"""Repository paths, immutable input identities, and atomic file publication."""
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ACTOR=HERE.parent
BASE=ACTOR.parent
ROOT=BASE.parents[1]
TRAIN=ACTOR/'blended_actor_dev'
PROTO=HERE/'protocol.json'
PUBLISHED='2cd0e6457d09a1fdce9fb507954ed39a3deb1f8f'
CACHE=Path('/workspace/sapling-master6-actor-data')

def sha(p):
    return hashlib.sha256(p if isinstance(p,bytes) else Path(p).read_bytes()).hexdigest()

def read(p):
    p=Path(p)
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())

def save(p,value):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
    raw=(json.dumps(value,sort_keys=True,indent=None if p.suffix=='.gz' else 2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    if p.suffix=='.gz':raw=gzip.compress(raw,mtime=0)
    if p.exists():
        if raw!=p.read_bytes():raise ValueError('existing output disagrees '+str(p))
        return
    tmp=p.with_name(p.name+'.'+str(os.getpid())+'.part')
    tmp.write_bytes(raw);os.replace(tmp,p)

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def git(*args):
    return subprocess.check_output(['git',*map(str,args)],cwd=ROOT)

def published(path,rev=PUBLISHED):
    return git('show',f'{rev}:{path.relative_to(ROOT)}')==path.read_bytes()

def require(test,message):
    if not test:raise ValueError(message)

"""Shared immutable record utilities and frozen Carbon League code locations."""
import gzip
import hashlib
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
ACTOR=HERE.parent
SAP=ACTOR.parent
PARENT=ROOT/"data/weights/sapling_1200k/terminal_return_fit_v1.py"
DIRECTIONS=("NONE","UP","RIGHT","DOWN","LEFT")
OFF=((0,0),(-1,0),(0,1),(1,0),(0,-1))

def move(pos,cmd,n=15):
    y,x=divmod(pos,n);a,b=OFF[DIRECTIONS.index(cmd or "NONE")]
    return ((y+a)%n)*n+(x+b)%n

def dist(a,b,n=15):
    y,x=divmod(a,n);u,v=divmod(b,n)
    return min(abs(y-u),n-abs(y-u))+min(abs(x-v),n-abs(x-v))

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def dig(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def read(p):
    raw=p.read_bytes()
    return json.loads(gzip.decompress(raw) if p.suffix==".gz" else raw)

def save(p,v):
    if p.exists():
        if read(p)!=v:raise ValueError(f"frozen {p} differs")
        return
    tmp=p.with_name(f".{p.name}.{os.getpid()}.part")
    data=(json.dumps(v,separators=(",",":"),ensure_ascii=False,allow_nan=False)+"\n").encode()
    tmp.write_bytes(gzip.compress(data,mtime=0) if p.suffix==".gz" else data)
    tmp.replace(p)

def source(row):
    p=ROOT/row["raw"]
    assert sha(p)==row["raw_sha256"] and sha(ROOT/row["official"])==row["official_sha256"]
    v=read(p)
    assert (v["seed"],v["candidate_seat"],v["initial_terrain_u8_sha256"])==(row["seed"],row["seat"],row["map"])
    assert v["pack_sha256"]==sha(PARENT) and not v["audit_messages"] and not v["errors"]
    return v

"""Freeze all public score cards in main, retrieve complete replays, verify them, and publish in volumes."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from contextlib import contextmanager
from datetime import datetime,timezone
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import ssl
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from urllib.request import Request,urlopen

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"reports/master_family/deep_review"))
from recover_public import canonical,extract,identity,key_of,validate,walk_matches

REV="5d7190d3680081317d446532b0b7ebe8d3d38ee2"
CACHE=Path("/workspace/sapling-all-top5-20261119")
TARGET=ROOT/"data/sapling_all_top5_20261119"
SERVER="https://117.72.210.134"
ID=re.compile(r"([a-f0-9]{32}-[0-9]+)\.json\.gz$")
RANKS=[("9d61f0056baf7f63703a671cd438bef2","claude-AlphaCarbon-Master5s"),("ecde65e596acbeb9ba5feca22e05f100","claude-AlphaCarbon-Master6"),("d1ab80fcc01010c949592984c617db30","RyanX-AIPowercore-it720-preview"),("b3fa633052e17329f34054bab0090128","RyanX-AIPowercore-M9XJ-630"),("2d4e4c772f057208b7121ed76dba4f33","ClaudeFable5.2-v7")]
NAMES={n:s for s,n in RANKS}
ALLOWED={"UP","RIGHT","DOWN","LEFT","RECCOLLECTOR","RECPLANTER","NONE",None}
def sha(b):return hashlib.sha256(b).hexdigest()
def git(*args):return subprocess.check_output(["git","-C",str(ROOT),*args])
def load(path):
    return json.loads(path.read_bytes())
def blob(path):
    return git("show",REV+":"+path)
def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:f.write(data)
    Path(f.name).replace(path)
def dump(path,obj):
    atomic(path,(json.dumps(obj,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n").encode())
def tree():
    return {line.split("\t",1)[1]:line.split(" ",2)[2].split("\t",1)[0] for line in git("ls-tree","-r",REV).decode().splitlines()}
def sourcefile(path):
    name=Path(path).name
    return ((("match" in name and not ("receipt" in name or "mismatch" in name))
             or name in {"index.json","top5_score_rows.json","latest5_inventory_sources.json.gz","latest5_inventory.json"}
             or (name.endswith(".json") and (any(token in name for token in ("card","_sources","_inventory","_score","replay_","_replay","_missing","_manifest")) or "/deep_review/by_source/" in path)))
            and (name.endswith(".json") or name.endswith(".json.gz")))
def cardrow(row):
    return {k:row[k] for k in ("id","game","name_a","name_b","order","scores","winner","created","state") if k in row}
def prepare(m,write=True,unscored=None):
    paths=tree();sources={}; rows={};extra=Counter()
    candidates=[path for path in paths if sourcefile(path)]
    proc=subprocess.Popen(["git","-C",str(ROOT),"cat-file","--batch"],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    for path in sorted(candidates):
        proc.stdin.write((REV+":"+path+"\n").encode());proc.stdin.flush()
        size=int(proc.stdout.readline().split()[2]);raw=proc.stdout.read(size);proc.stdout.read(1)
        if raw[:2]==b"\x1f\x8b":raw=gzip.decompress(raw)
        try: doc=json.loads(raw)
        except ValueError as exc: raise ValueError("invalid JSON index "+path) from exc
        found=0
        for row in walk_matches(doc):
            if not {row["name_a"],row["name_b"]}&set(NAMES):continue
            if row.get("state","done")!="done":
                extra["nonterminal_reference"]+=1;continue
            k=key_of(row)
            if unscored is not None and sorted(row["order"])==["a","b"] and any(row["scores"].get(s) is None for s in row["order"]):
                record=unscored.setdefault(k,{"card":cardrow(row),"sources":[]})
                if record["card"]!=cardrow(row) or k in rows:
                    raise ValueError("conflicting unscored mirror: "+k)
                record["sources"].append(path)
                continue
            if unscored is not None and k in unscored:
                raise ValueError("conflicting numeric and unscored mirror: "+k)
            found+=1
            if k not in rows:rows[k]={"card":cardrow(row),"sources":set()}
            elif identity(rows[k]["card"])!=identity(row):raise ValueError("conflicting mirror: "+k+" "+path)
            rows[k]["sources"].add(path)
        sources[path]=dict(git_blob=paths[path],references=found)
    proc.stdin.close()
    if proc.wait():raise RuntimeError("git objects unavailable")
    direct=defaultdict(list)
    for p,h in paths.items():
        match=ID.fullmatch(Path(p).name)
        if match:direct[match.group(1)].append({"repo_path":p,"git_blob":h})
    bundled={}
    snap="reports/sapling_1200k/latest5_inventory.json"
    if snap in paths:
        for g in json.loads(blob(snap)).get("games",[]):
            if "archive" in g:bundled[g["id"]]={k:g[k] for k in ("archive","member","sha256")}
    oldfile="data/sapling_ladder_income_nov19/index.json"
    out=[]
    for k,group in sorted(rows.items()):
        card=group["card"]; players=list(identity(card))
        out.append(dict(id=k,seat_names=[r[0] for r in players],seat_strategy_ids=[NAMES.get(r[0]) for r in players],
                        score_by_seat=[r[1] for r in players],cohort="mutual" if all(r[0] in NAMES for r in players) else "external",
                        self_play=players[0][0]==players[1][0],score_sources=sorted(group["sources"]),card=card,
                        frozen_git_replays=direct[k],frozen_archive=bundled.get(k),path="replays/"+k+".json.gz",
                        seed=None,sha256=None,frame_verified=False))
    m.update(schema="top-five-all-appearances/1",state="score_cards",revision=REV,ranking_as_of_utc="2026-11-19T02:53:08Z",
             top5=[dict(rank=i+1,id=s,name=n) for i,(s,n) in enumerate(RANKS)],
             coverage=dict(score_cards=len(out),mutual=sum(g["cohort"]=="mutual" for g in out),external=sum(g["cohort"]=="external" for g in out),
                           self_play=sum(g["self_play"] for g in out),frame_verified=0,score_only=len(out),recognized_score_index_paths=len(sources)),
             index_sources=sources,omissions=dict(extra),games=out)
    if write:dump(HERE/"manifest.json",m)
    print("prepared",m["coverage"],flush=True)


def previous_cache():
    paths=subprocess.run(["rg","--files","--hidden","--no-ignore","/workspace","-g","*.json.gz","-g","!**/.git/**","-g","!**/site-packages/**"],capture_output=True,text=True,check=True)
    index=defaultdict(list)
    for line in paths.stdout.splitlines():
        p=Path(line);m=ID.fullmatch(p.name)
        if m and CACHE not in p.parents:index[m.group(1)].append(p)
    return index


def verify(env,row):
    r=validate(env,row["card"]); cfg=env["configuration"]
    if not isinstance(r["random_seed"],int):raise ValueError("missing seed")
    for i,frame in enumerate(env["steps"]):
        full,partial=[s["observation"] for s in frame]
        if (full.get("player")!=0 or partial.get("player")!=1 or len(full.get("full_carbon",[]))!=225
            or "full_carbon" in partial or len(partial.get("carbon",[]))!=225 or len(full.get("carbon",[]))!=225):
            raise ValueError("observation visibility "+str(i))
        if i:
            for x in frame:
                cmd=x.get("action")
                if not isinstance(cmd,dict):
                    # Fault or extinction replays can have null actions, explicitly retain status.
                    if x.get("status") in {"ERROR","INVALID","TIMEOUT","DONE"} and cmd is None:continue
                    raise ValueError("missing actions "+str(i))
                if set(cmd.values())-ALLOWED:raise ValueError("unexpected actions "+str(i))
    return r


class Retrieval:
    def __init__(self,prior,delay):
        self.prior=prior;self.lock=threading.Lock();self.next_request=0.;self.delay=delay
        self.archives={};self.web=threading.BoundedSemaphore(3)
    def get(self,url):
        for attempt in range(3):
            with self.lock:
                wait=max(0,self.next_request-time.monotonic())
                self.next_request=max(time.monotonic(),self.next_request)+self.delay
            time.sleep(wait)
            try:
                req=Request(url,headers={"Accept-Encoding":"gzip","User-Agent":"carbon-league-replay-audit/1"})
                with self.web,urlopen(req,context=ssl._create_unverified_context(),timeout=30) as res:
                    b=res.read();return b
            except Exception as exc:
                if attempt==2:raise
                time.sleep(2**attempt)
    def oldmember(self,entry):
        k=entry["archive"]
        with self.lock:
            if k not in self.archives:
                with tarfile.open(fileobj=io.BytesIO(blob(k))) as tar:
                    self.archives[k]={p.name:tar.extractfile(p).read() for p in tar.getmembers() if p.isfile()}
            b=self.archives[k][entry["member"]]
            if sha(b)!=entry["sha256"]:raise ValueError("archived member SHA")
            return b
    def one(self,row):
        key=row["id"];f=CACHE/row["path"];receipt=CACHE/"receipts"/(key+".json")
        if f.exists() and receipt.exists():
            rec=load(receipt)
            if rec["id"]==row["id"] and sha(f.read_bytes())==rec["sha256"] and rec["revision"]==REV:return rec
        errors=[];known=row["frozen_git_replays"]
        def candidates():
            if f.exists():yield dict(method="recovered_cache"),f.read_bytes
            if known:
                for loc in known:
                    p=ROOT/loc["repo_path"]
                    local=p.read_bytes() if p.is_file() else None
                    if local is not None and hashlib.sha1(b"blob "+str(len(local)).encode()+b"\0"+local).hexdigest()==loc["git_blob"]:
                        yield dict(method="frozen_git",**loc),lambda b=local:b
                    else:yield dict(method="frozen_git",**loc),lambda q=loc["git_blob"]:git("cat-file","blob",q)
            if row["frozen_archive"]:
                e=row["frozen_archive"];yield dict(method="frozen_archive",**e),lambda:self.oldmember(e)
            for p in self.prior[key]:
                yield dict(method="existing_workspace_cache",cache_name=p.parent.name),p.read_bytes
            url=SERVER+f"/replay/{row['card']['id']}/{row['card']['game']}"
            yield dict(method="official_public_get",url=url),lambda:self.get(url)
        for src,getdata in candidates():
            try:
                b=getdata();env,_,_=extract(b);detail=verify(env,row)
                content=canonical(env)
                gz=b if b.startswith(b"\x1f\x8b") and gzip.decompress(b).lstrip().startswith(b'{"id"') else gzip.compress(content,compresslevel=4,mtime=0)
                if src["method"] in ("frozen_git","frozen_archive") and b.startswith(b"\x1f\x8b") and json.loads(gzip.decompress(b)).get("steps"):
                    gz=b
                rec=dict(id=key,revision=REV,source=src,sha256=sha(gz),canonical_sha256=sha(content),seed=detail["random_seed"],
                         environment_uuid=detail["environment_uuid"],transitions=detail["last_step"],terminal=detail["statuses"],
                         clean_terminal=detail["statuses"]==["DONE","DONE"],source_attempt_errors=errors,frame_verified=True)
                atomic(f,gz);dump(receipt,rec);return rec
            except Exception as exc:errors.append(dict(source=src,error=repr(exc)[:350]))
        return dict(id=key,frame_verified=False,source_attempt_errors=errors)


def publish(m,results,phase):
    for row in m["games"]:
        if row["id"] in results:
            row.update(results[row["id"]])
    ok=[g for g in m["games"] if g["frame_verified"]]
    m["state"]=phase
    m["coverage"].update(frame_verified=len(ok),score_only=len(m["games"])-len(ok),clean_terminal=sum(g["clean_terminal"] for g in ok),
                         verified_frames=sum(g["transitions"]+1 for g in ok),verified_transitions=sum(g["transitions"] for g in ok),
                         sources=dict(Counter(g["source"]["method"] for g in ok)))
    dump(HERE/"manifest.json",m)
    print(datetime.now(timezone.utc).isoformat(),phase,len(ok),"/",len(m["games"]),m["coverage"]["sources"],flush=True)


def volumes(m):
    rows=[r for r in m["games"] if r["frame_verified"]]
    remaining=[]
    for r in rows:
        s=r["source"];method=s["method"]
        if method=="frozen_git":r["archive_location"]={k:s[k] for k in ("repo_path","git_blob")}
        elif method=="frozen_archive":r["archive_location"]={k:s[k] for k in ("archive","member")}
        else:remaining.append(r)
    chunks=[];used=0
    for r in remaining:
        n=(CACHE/r["path"]).stat().st_size
        if not chunks or used+n>35_000_000:chunks.append([]);used=0
        chunks[-1].append(r);used+=n
    out=[]
    for i,subset in enumerate(chunks,1):
        # Per-replay payloads are already gzipped. A deterministic plain tar
        # preserves member bytes without re-deflating hundreds of MB.
        path=TARGET/f"replays-{i:03d}.tar"; path.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as tmp:
            with tarfile.open(fileobj=tmp,mode="w",format=tarfile.USTAR_FORMAT) as tar:
                for row in subset:
                    member=row["path"];data=(CACHE/member).read_bytes()
                    if sha(data)!=row["sha256"]:raise ValueError("cache digest changed "+member)
                    ti=tarfile.TarInfo(member);ti.size=len(data);ti.mtime=0;ti.mode=0o644
                    tar.addfile(ti,io.BytesIO(data))
            tmp.flush()
            with tarfile.open(tmp.name) as tar:
                if len(tar.getnames())!=len(subset):raise ValueError("volume count")
                for row in subset:
                    if sha(tar.extractfile(row["path"]).read())!=row["sha256"]:raise ValueError("volume checksum")
        Path(tmp.name).replace(path)
        digest=sha(path.read_bytes());repo_path=str(path.relative_to(ROOT))
        out.append(dict(archive=repo_path,sha256=digest,bytes=path.stat().st_size,members=len(subset)))
        for r in subset:r["archive_location"]=dict(archive=repo_path,member=r["path"])
    m["new_archives"]=out


def description(m):
    rows=m["games"]; verified=[r for r in rows if r["frame_verified"]]
    by_name={name:dict(appearances=sum(name in r["seat_names"] for r in rows),
                       seat0=sum(r["seat_names"][0]==name for r in rows),
                       seat1=sum(r["seat_names"][1]==name for r in rows),
                       wins=sum(r["score_by_seat"][r["seat_names"].index(name)]>r["score_by_seat"][1-r["seat_names"].index(name)] for r in rows if name in r["seat_names"]),
                       draws=sum(r["score_by_seat"][0]==r["score_by_seat"][1] for r in rows if name in r["seat_names"]))
             for name in NAMES}
    m["coverage"].update(registered_player_appearances=sum(v["appearances"] for v in by_name.values()),
                       score_card_references=sum(s["references"] for s in m["index_sources"].values()),
                       distinct_replay_sha256=len({r["canonical_sha256"] for r in verified}),
                       distinct_map_seeds=len({r["seed"] for r in verified}),
                       distinct_environment_uuids=len({r["environment_uuid"] for r in verified}))
    m["by_strategy"]=by_name
    m["input_scope"]="Only completed score cards present in the frozen repository, including nested score exports and the captured website index. Dedup by match id and game; order maps a/b scoreboard to physical seats. Each individual score reference must yield the same identity-sorted physical scores."
    m["status_scope"]="Terminal flags/details retained; rows with execution faults are in coverage, excluded by clean_terminal for ordinary-strategy inference."
    dup=defaultdict(list)
    for r in verified:dup[r["seed"]].append(r["id"])
    m["duplicate_map_ids"]={k:v for k,v in dup.items() if len(v)>1}
    dup=defaultdict(list)
    for r in verified:dup[r["environment_uuid"]].append(r["id"])
    m["duplicate_environment_ids"]={k:v for k,v in dup.items() if len(v)>1}
    dump(HERE/"manifest.json",m)


def user_asset(path,ref):
    local=ROOT/path
    return local.read_bytes() if local.is_file() else git("show",ref+":"+path)


def restore(m,dest,assets_ref):
    """Rebuild the shared replay directory solely from frozen Git objects and referenced volumes."""
    volumes=defaultdict(list)
    dest.mkdir(parents=True,exist_ok=True)
    def emit(row,b):
        if sha(b)!=row["sha256"]:
            env,_,_=extract(b); b=gzip.compress(canonical(env),compresslevel=4,mtime=0)
        if sha(b)!=row["sha256"]:raise ValueError("restored bytes differ: "+row["id"])
        f=dest/row["path"]
        if not f.exists():atomic(f,b)
        elif sha(f.read_bytes())!=row["sha256"]:raise ValueError("existing restore has different bytes: "+row["id"])
    for row in m["games"]:
        if not row["frame_verified"]:continue
        loc=row["archive_location"]
        if "git_blob" in loc:emit(row,git("cat-file","blob",loc["git_blob"]))
        else:volumes[loc["archive"]].append(row)
    ar={v["archive"]:v["sha256"] for v in m.get("new_archives",[])}
    for name,records in volumes.items():
        b=user_asset(name,assets_ref) if name in ar else blob(name)
        if name in ar and sha(b)!=ar[name]:raise ValueError("volume digest "+name)
        with tarfile.open(fileobj=io.BytesIO(b)) as tar:
            for row in records:emit(row,tar.extractfile(row["archive_location"]["member"]).read())
    print("restored",m["coverage"]["frame_verified"],flush=True)


def run(a):
    path=HERE/"manifest.json"
    if a.prepare or not path.exists():prepare({})
    if a.prepare:return
    m=load(path)
    if a.restore:
        restore(m,a.restore,a.assets_ref);return
    if a.summarize:
        catalog={};prepare(catalog,write=False)
        origin={g["id"]:g for g in catalog["games"]}
        if set(origin)!={g["id"] for g in m["games"]}:raise ValueError("full score cross-check changed cohort")
        for g in m["games"]:
            if (g["seat_names"],g["score_by_seat"])!=(origin[g["id"]]["seat_names"],origin[g["id"]]["score_by_seat"]):
                raise ValueError("physical seats changed")
            g["score_sources"]=origin[g["id"]]["score_sources"]
        m["index_sources"]=catalog["index_sources"]
        m["coverage"]["recognized_score_index_paths"]=len(catalog["index_sources"])
        description(m);return
    if m["revision"]!=REV:raise ValueError("manifest revision")
    CACHE.mkdir(parents=True,exist_ok=True)
    with (CACHE/"fetch.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=previous_cache()
        obj=Retrieval(previous,a.delay);results={}
        # Alternate cached states, Git objects and public-HTTP-only fixtures so
        # downstream analysis can start on each cohort before the last group ends.
        groups=[[],[],[]]
        for g in m["games"]:
            gid=g["id"]
            groups[0 if previous[gid] or g["frozen_archive"] else 1 if g["frozen_git_replays"] else 2].append(g)
        order=[grp[i] for i in range(max(map(len,groups))) for grp in groups if i<len(grp)]
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures={pool.submit(obj.one,g):g["id"] for g in order}
            for i,f in enumerate(as_completed(futures),1):
                rec=f.result();results[rec["id"]]=rec
                if not rec["frame_verified"]:print("failed",rec["id"],rec["source_attempt_errors"][-1],flush=True)
                if i%32==0:publish(m,results,"retrieving")
        publish(m,results,"validating_archives")
        volumes(m)
        publish(m,results,"complete" if all(g["frame_verified"] for g in m["games"]) else "partial")
        description(m)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare",action="store_true")
    p.add_argument("--summarize",action="store_true")
    p.add_argument("--restore",type=Path)
    p.add_argument("--assets-ref",default="HEAD")
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--delay",type=float,default=.35,help="minimum gap in seconds between starts of public requests")
    args=p.parse_args()
    if not 1<=args.workers<=6 or args.delay<.20:p.error("bounded public concurrency and request pacing")
    run(args)

"""Check original deployed partner bytes and compare only exams with bound contestant/opponent identities."""
import ast
import base64
from collections import Counter,defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import zlib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
BASE="e714939d7c85e54b746e8b130269cf780a523153"
EXAM="d599ec679718dd4407a718aa6ad810e35e2d3983"
S="reports/sapling_1200k/master6_actor_dev/strategic_destination_learning"
SOURCE={k:"code/exp/"+k+".py" for k in ("it74","it85","it95")}
SOURCE.update(it50="code/packs/ppou2_m_it50.py",ede48=S+"/bc.py")
FIELDS={"_Q1","_M1","_B1","_HYBRID","CONF","_SRC"}
H29="reports/war-room/exam_snapshots/h29"
H31="reports/war-room/exam_snapshots/h31"


def git(*args):
    return subprocess.check_output(["git","-c","protocol.allow=never",*args],cwd=ROOT,env=dict(os.environ,GIT_NO_LAZY_FETCH="1"),stderr=subprocess.DEVNULL)


def h(data):
    return hashlib.sha256(data).hexdigest()


def obj(revision,path):
    return git("show",revision+":"+path)


def weights(path):
    raw=obj(BASE,path)
    fields={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(raw).body if isinstance(n,ast.Assign)
            and isinstance(n.targets[0],ast.Name) and n.targets[0].id in FIELDS}
    packed=zlib.decompress(base64.b64decode(fields["_B1"],validate=True));tensors=fields["_Q1"];pos=0;param={}
    assert sum(r[3] for r in tensors)==len(packed)
    for name,shape,kind,n in tensors:
        assert name not in param
        param[name]=h(packed[pos:pos+n]);pos+=n
    return dict(path=path,blob=git("rev-parse",BASE+":"+path).decode().strip(),sha256=h(raw),md5=hashlib.md5(raw).hexdigest(),
                nbytes=len(raw),actor_sha256=h(packed),actor_byte_length=len(packed),tensor_count=len(tensors),
                tensors=param,shapes={a:[b,c] for a,b,c,_ in tensors},metadata=json.loads(fields["_M1"]),
                confidence=fields.get("CONF"),mix=fields.get("_HYBRID"),
                encoder_sources={k:h(v.encode()) for k,v in fields.get("_SRC",{}).items()})


def rows(rev,path):
    raw=obj(rev,path)
    return [json.loads(line) for line in raw.splitlines()],h(raw)


def stats(rows):
    wins=sum(r["margin"]>0 for r in rows);losses=sum(r["margin"]<0 for r in rows)
    assert len({(r["seed"],r["side"]) for r in rows})==len(rows)
    for r in rows:
        if "sa" in r and "sb" in r:
            assert abs(r["margin"]-(r["sa"]-r["sb"]))<0.012
        label=(1. if r["margin"]>0 else 0. if r["margin"]<0 else .5)
        assert abs(r['u']-(.5*label+.5/(1+math.exp(-max(-60,min(60,r['margin']/1000))))))<1e-5
    return dict(n=len(rows),win=wins,draw=len(rows)-wins-losses,loss=losses,
                margin=statistics.mean(r["margin"] for r in rows),u=statistics.mean(r["u"] for r in rows),
                seeds=[min(r["seed"] for r in rows),max(r["seed"] for r in rows)],
                by_seat={str(s):sum(r["side"]==s for r in rows) for s in (0,1)},
                reported_errors=sum(bool(r.get("err")) or bool(r.get("serr")) or bool(r.get("berr")) for r in rows),
                hashes={k:dict(Counter(r[k] for r in rows)) for k in ("ah","bh","hh")})


def main():
    assets={k:weights(p) for k,p in SOURCE.items()}
    for k,v in assets.items():
        assert v["shapes"]==assets["it50"]["shapes"]
    assert assets["ede48"]["actor_sha256"]==assets["it50"]["actor_sha256"]
    published={k:{**{key:val for key,val in v.items() if key not in {"tensors","shapes"}},
                  "different_serialized_tensors_vs_it50":sum(z!=assets["it50"]["tensors"][n] for n,z in v["tensors"].items())}
                  for k,v in assets.items()}
    named={}
    for k in ("it74","it85","it95"):
        for foe in ("Master2","Echo2"):
            if k=="it74":p=f"data/it74_test/it74_{foe.lower()}.jsonl"
            elif foe=="Master2":p=f"{H29}/{k}_m2.jsonl"
            else:p=f"{H31}/{k}/{k}_echo2.jsonl"
            records,digest=rows(BASE,p)
            assert all(r["ah"]==assets[k]["md5"] for r in records)
            card=dict(source=p,sha256=digest,revision=BASE,**stats(records))
            if k!="it74":
                paired=(f"{H29}/{k}base_m2.jsonl" if foe=="Master2" else
                        f"{H31}/{k}/m3s2r1_echo2_base{'95' if k=='it95' else ''}.jsonl")
                original,ph=rows(BASE,paired);by={(r['seed'],r['side']):r for r in original}
                assert len(original)==len(by)==len(records)
                n=len(records);delta=[]
                for r in records:
                    ref=by[(r['seed'],r['side'])];assert ref['bh']==r['bh'];delta.append(r['u']-ref['u'])
                card['paired_vs_m3s2r1']=dict(source=paired,sha256=ph,n=n,mean_du=statistics.mean(delta),
                                             paired_se=statistics.stdev(delta)/math.sqrt(n))
            named[f"{k}/{foe}"]=card
    for foe,p in (("Echo4","synced/h29/bigeval_live/m50_e4.jsonl"),("original3s2","synced/h29/bigeval_live/abD_it50_3s2.jsonl")):
        r,digest=rows(EXAM,p)
        if foe=="original3s2" and "u" not in r[0]:
            for a in r:a["u"]=.5*(1 if a["margin"]>0 else 0 if a["margin"]<0 else .5)+.5/(1+math.exp(-a["margin"]/1000))
        assert all(a["ah"]==assets["it50"]["md5"] for a in r);named["it50/"+foe]=dict(source=p,sha256=digest,revision=EXAM,**stats(r))
    # Look up the identity values in both positions of every locally available tracked JSONL blob (deduplicated by blob).
    tree_paths=defaultdict(list)
    for rev in (BASE,EXAM):
        for line in git("ls-tree","-r",rev).decode().splitlines():
            head,path=line.split("\t",1)
            if path.endswith(".jsonl"):tree_paths[head.split()[2]].append((rev,path))
    tokens={k:v["md5"] for k,v in assets.items() if k!="ede48"}
    examined=0;missing=[];hits=defaultdict(list)
    for blob,paths in tree_paths.items():
        try:raw=git("cat-file","blob",blob)
        except subprocess.CalledProcessError:
            missing.append(blob);continue
        examined+=1
        if not any(token.encode() in raw for token in tokens.values()):continue
        matches=Counter()
        for ln in raw.splitlines():
            if not any(v.encode() in ln for v in tokens.values()):continue
            try:r=json.loads(ln)
            except ValueError:continue
            if not isinstance(r,dict) or 'ah' not in r or 'bh' not in r or 'margin' not in r:continue
            for k,v in tokens.items():
                if v==r['ah']:matches[k,'a',r['bh']]+=1
                if v==r['bh']:matches[k,'b',r['ah']]+=1
        for (k,side,other),n in matches.items():hits[k].append(dict(blob=blob,example=paths[0],seat_key=side,other_md5=other,rows=n))
    out=dict(revisions=[BASE,EXAM],assets=published,exam_summary=named,
             coverage=dict(unique_jsonl_blobs=len(tree_paths),readable=examined,missing=missing,
                           candidate_opponent_matches=hits),
             note="Official scores and packed actor weights identify published participants. Historical native clock depth missing in the 300-game it50 archive. JSONL inventory covers these two Git snapshots only.")
    (HERE/"audit.json").write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False)+"\n")


if __name__=="__main__":
    main()

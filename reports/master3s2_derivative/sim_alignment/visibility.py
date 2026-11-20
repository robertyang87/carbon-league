"""Track what the parent could actually infer, then score map forecasts with referee-only truth."""
import hashlib
import gzip
import importlib.util
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
HERE=Path(__file__).resolve().parent
D=HERE.parent
def sha(raw):return hashlib.sha256(raw).hexdigest()
def load(path):
    s=importlib.util.spec_from_file_location("sim_audit_parent",path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def read(p):
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix==".gz" else p.read_bytes())
def report(p,r):
    raw=(json.dumps(r,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    if p.suffix==".gz":raw=gzip.compress(raw,mtime=0)
    p.write_bytes(raw)
def data(card):
    path=D/"ladder/replays"/(card["match_id"]+"-1.json.gz")
    raw=path.read_bytes();j=json.loads(gzip.decompress(raw))
    return j,sha(raw)
def obs(fr,seat):
    f=fr[0]["observation"];v=fr[seat]["observation"]
    return dict(players=f["players"],step=f["step"],player=seat,trees=f["trees"],carbon=v["carbon"],remainingOverageTime=v["remainingOverageTime"])
def go():
    m=load(D/"cold_start/candidate.py");book=read(D/"ladder/ds-pro-v1-placement.json");out=[]
    for c in book["games"]:
        tape,h=data(c);frames=tape["steps"];seat=c["candidate_seat"]
        tr=m.BeliefTracker(m._Bel,seat);rows=[]
        for t,f in enumerate(frames[:-1]):
            o=obs(f,seat);ps=o["players"]
            est,known=tr.step(ps,t,o["carbon"]);mask=np.array(known,dtype=bool)
            truth=np.array(f[0]["observation"]["full_carbon"]);err=np.abs(truth-est)
            rows.append(dict(step=t,unknown=int((~mask).sum()),known_err=float(err[mask].sum()),
                             unknown_err=float(err[~mask].sum()),max_known=float(err[mask].max()),
                             unknown_actual_nonzero=int((truth[~mask]>0).sum()),
                             known_errors_gt_1=int(np.sum(err[mask]>1)),unknown_errors_gt_1=int(np.sum(err[~mask]>1))))
            tr.set_actions(ps,frames[t+1][seat]["action"] or {})
        out.append(dict(match=c["match_id"],seat=seat,replay_sha=h,rows=rows))
    report(HERE/"visibility_rows.json.gz",out)
    sums=[]
    for lo,hi in ((0,49),(50,149),(150,249),(250,298),(0,298)):
        rows=[r for g in out for r in g["rows"] if lo<=r["step"]<=hi];n=len(rows);u=sum(r["unknown"] for r in rows)
        sums.append(dict(range=[lo,hi],steps=n,roots_with_unknown=sum(r["unknown"]>0 for r in rows),
                         unknown_fraction=u/(225*n),known_mae=sum(r["known_err"] for r in rows)/(225*n-u),
                         unknown_mae=sum(r["unknown_err"] for r in rows)/u if u else None,
                         frac_known_with_error_above_1=sum(r["known_errors_gt_1"] for r in rows)/(225*n-u),
                         fraction_unknown_with_error_above_1=sum(r["unknown_errors_gt_1"] for r in rows)/u if u else None))
    report(HERE/"visibility.json",dict(parent=sha((D/"cold_start/candidate.py").read_bytes()),panel=sha((D/"ladder/ds-pro-v1-placement.json").read_bytes()),
                                    periods=sums,rows_sha256=sha((HERE/"visibility_rows.json.gz").read_bytes())))
if __name__=="__main__":go()

"""Held public-arm release and per-slot isolation checks, separate full inference reruns."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import platform
import sys
import types

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from common import PROTOCOL, public_frames, read, sha, save
from verify_entry import run_pack, array_hash
from runtime import _carbon_inputs

ARMS = {"decoder_student":"candidate", "decoder_constant":"constant"}
CARDS = [c for c in PROTOCOL["cards"] if c["split"]=="val"]
TMP = Path("/tmp") / ("sapling-carbon-decoder-held-"+sha((HERE/"protocol.json").read_bytes())[:16])
TMP.mkdir(exist_ok=True)


def role_map(o):
    p = o["players"][o["player"]]
    return {**dict.fromkeys(p[1], "base"), **{u:w[2].lower() for u,w in p[2].items()}}


def run_lane(arm, cpu, mode):
    os.sched_setaffinity(0,{cpu})
    sarm = ARMS[arm]
    pack = HERE/(sarm+".py")
    source = pack.read_bytes()
    meta = read(HERE/"packages.json")["arms"][sarm]
    assert meta["sha256"] == sha(source) and meta["bytes"] == len(source) <= 10_000_000
    if mode == "audit":
        p = types.ModuleType("public_audit")
        p.__file__ = str(pack)
        exec(compile(source,str(pack),"exec"),p.__dict__)
        audit = Audit(p, arm)
    for card in CARDS:
        frames = [[],[]]
        for obs, _, joint, cfg in public_frames(card):
            for s in (0,1):
                frames[s].append(obs[s])
        for seat, observations in enumerate(frames):
            path = TMP/f"{arm}_{mode}_{card['seed']}_{seat}.json.gz"
            assert len(observations)==299
            if mode == "timing":
                r = run_pack(pack,"timing",cpu,observations,cfg)
                assert r["ready"]["pack_sha256"] == meta["sha256"]
            else:
                rows = []
                audit.card=card
                for obs in observations:
                    action, counts = audit.step(obs, cfg)
                    rows.append(dict(action=action,counts=counts,roles=role_map(obs)))
                r = dict(rows=rows)
            save(path,r)
            print(arm,mode,card["seed"],seat,flush=True)


class Audit:
    """Check values at the actual consumers while allowing each policy its own decision history."""

    def __init__(self, module, arm):
        self.m = m = module
        self.F = F = sys.modules["cnet2_feat"]
        P = sys.modules["cnet2_np"].Policy
        self.ref = read(HERE.parent/"normal_entry.json.gz")["incumbent"]
        self.frozen = self.ref[0]["audit"]["rows"][0]["frozen"]
        contents = (HERE.parent/"weights.npz").read_bytes()
        assert sha(contents)==read(HERE/"protocol.json")["weights"]
        parts = np.load(HERE.parent/"weights.npz")
        self.weights = [parts[k] for k in ("wbva" if arm=="decoder_student" else "c")]
        assert all(np.array_equal(a,b) for a,b in zip(self.weights,m._CARBON_WEIGHTS,strict=True))
        old_views = m._carbon_views
        old_decode = m._return_decode
        old_encode = P.encode
        old_init = F.Prefix.__init__
        old_feats = F.Prefix.feats
        old_advance = F.Prefix.advance
        old_lin = P.lin
        old_basis = m._return_basis
        old_mixed = m._mixed_actions

        def views(x,aux,feat,F,weights):
            assert aux is feat.aux
            self.x,self.aux = x,aux
            order = F.CANON if feat.seat else F.GRID
            source = (feat.est/100).astype(np.float32)[order]
            raw_base = aux["est"].copy()
            map_base = x["maps"].copy()
            assert np.array_equal(F.f16(source),raw_base)
            assert np.array_equal(F.q8(source),map_base[0])
            sx,sa = old_views(x,aux,feat,F,weights)
            assert np.array_equal(raw_base,aux["est"]) and np.array_equal(map_base,x["maps"])
            assert np.array_equal((feat.est/100).astype(np.float32)[order],source)
            features,eligible = _carbon_inputs(x,feat,F)
            expected = np.ravel((map_base[2]==0) & (map_base[14:18]==0).all(0))
            assert np.array_equal(eligible,expected)
            if len(self.weights)==1:
                delta=self.weights[0]
            else:
                w,b,v,a=self.weights
                delta=np.maximum(features@w+b,0)@v+a
            est=np.clip(features[:,0]+delta,0,1)
            encoded=map_base[0].copy()
            native=raw_base.copy()
            encoded[eligible]=F.q8(est[eligible])
            native[eligible]=F.f16(est[eligible])
            assert np.array_equal(sx["maps"][0],encoded) and np.array_equal(sa["est"],native)
            assert np.array_equal(sx["maps"][1:],x["maps"][1:])
            assert sx.keys()==x.keys() and sa.keys()==aux.keys()
            for k in x.keys()-{"maps"}: assert np.array_equal(sx[k],x[k])
            for k in aux.keys()-{"est"}: assert np.array_equal(sa[k],aux[k])
            self.student_x,self.student_aux = sx,sa
            self.c["eligible_cells"]+=int(eligible.sum())
            self.c["changed_native_cells"]+=int(np.sum(native!=raw_base))
            self.c["changed_encoder_cells"]+=int(np.sum(encoded!=map_base[0]))
            self.c["prediction_views"]+=1
            return sx,sa

        def encode(policy,x,style):
            idx=m._NETS.index(policy)
            assert x is (self.x,self.student_x)[idx] and style==2
            tok,ctx=old_encode(policy,x,style)
            if idx==0 and self.reference is not None:
                assert self.reference["partner_x"]==array_hash(x)
                assert self.reference["partner_encoding"]==array_hash({"tokens":tok,"context":ctx})
                self.c["archived_partner_activations_matched"]+=1
            self.c["encodes"]+=1
            return tok,ctx

        def decode(x,aux,encoded,apply_residual,student_aux):
            assert x is self.x and aux is self.aux and student_aux is self.student_aux
            assert [e[0] for e in encoded]==m._NETS
            self.paths.append(bool(apply_residual))
            self.encoded=encoded
            self.prefixes=[]
            self.enacted=[[],[]]
            self.branch_flag=apply_residual
            y=old_decode(x,aux,encoded,apply_residual,student_aux)
            assert len(self.prefixes)==2 and self.enacted[0]==self.enacted[1]
            assert self.enacted[0]==[(int(s),int(y[s])) for s in np.flatnonzero(x["tok_mask"][:F.OWN])]
            self.decoded[bool(apply_residual)]=y.copy()
            return y

        def initial(prefix,x,aux):
            role=len(self.prefixes)
            assert role<2 and x is self.x and (self.aux,self.student_aux)[role] is aux
            old_init(prefix,x,aux)
            self.prefixes.append(prefix)

        def features(prefix,s):
            i=self.prefixes.index(prefix)
            result=old_feats(prefix,s)
            if i==0:
                self.slot=int(s)
                self.moves=[result]
                self.calls=[]
                self.actions=[]
            else:
                assert int(s)==self.slot
                self.moves.append(result)
                unit,move,legal=result
                old_unit,old_move,old_legal=self.moves[0]
                assert np.array_equal(old_unit,unit) and np.array_equal(old_legal,legal)
                assert np.array_equal(old_move[:,:7],move[:,:7]) and np.array_equal(old_move[:,8:],move[:,8:])
                self.check_state()
                pos=self.x["tok_pos"][s]
                for j, pair in enumerate(self.moves):
                    native = (self.aux,self.student_aux)[j]["est"]
                    carbon = native[F.DEST[pos]] if s<10 else np.array([0,native[pos],native[pos],0,0])
                    assert np.array_equal(pair[1][:,7],carbon)
                changed=int(np.sum(old_move[:,7]!=move[:,7]))
                self.c["branch_slots"]+=1
                self.c["move7_changed_rows"]+=changed
                self.c["slots_with_move7_change"]+=int(changed>0)
                self.c["move7_changed_"+("base" if s>=10 else ("collector" if self.x["tok_kind"][s]==0 else "planter"))]+=changed
            return result

        def advance(prefix,slot,action):
            i=self.prefixes.index(prefix)
            assert int(slot)==self.slot and self.moves[i][2][action]
            self.enacted[i].append((int(slot),int(action)))
            if i==0:
                assert self.calls==[(0,"din"),(0,"V"),(1,"din"),(1,"V")]
                self.actions.append(int(action))
            else:
                assert self.actions==[int(action)]
            old_advance(prefix,slot,action)
            if i==1:
                self.check_state()
                self.c["synchronized_advances"]+=1

        def lin(policy,x,name):
            if name in ("din","V"):
                i=m._NETS.index(policy)
                unit,move,_=self.moves[i]
                if name=="V":
                    assert np.array_equal(x,move)
                else:
                    _,tokens,context,_=self.encoded[i]
                    components=[tokens[self.slot],context,unit,move.flatten()]
                    assert np.array_equal(x,np.concatenate(components))
                self.calls.append((i,name))
                self.c["consumer_"+("partner_" if i==0 else "student_")+name]+=1
            return old_lin(policy,x,name)

        def basis(x,aux,slot,move):
            assert self.branch_flag and self.x is x and aux is self.aux and slot==self.slot
            assert np.array_equal(move,self.moves[0][1])
            self.c["shared_return_basis_calls"]+=1
            return old_basis(x,aux,slot,move)

        def mixed(x,aux):
            y=old_mixed(x,aux)
            assert self.paths==[False,True]
            old=self.decoded[False]; revised=self.decoded[True]
            protected=(np.arange(F.OWN)>=10) | (x["tok_kind"][:F.OWN]==1)
            if np.any(old[protected]!=revised[protected]):
                expected=old;key="guard_protected"
            elif m._return_conflicts(x,revised)-m._return_conflicts(x,old):
                expected=old;key="guard_conflict"
            else:
                expected=revised;key="guard_return"
            assert np.array_equal(expected,y)
            self.c[key]+=1
            return y

        m._carbon_views=views; m._return_decode=decode; m._return_basis=basis; m._mixed_actions=mixed
        P.encode=encode; P.lin=lin
        F.Prefix.__init__=initial; F.Prefix.feats=features; F.Prefix.advance=advance

    def check_state(self):
        a,b=self.prefixes
        assert a is not b
        for k in ("cash","slots","plants","rec_pl"): assert getattr(a,k)==getattr(b,k)
        for k in ("reserved","undecided"): assert np.array_equal(getattr(a,k),getattr(b,k))

    def step(self,obs,cfg):
        self.c=Counter()
        self.paths=[]; self.decoded={}
        self.reference=(self.ref[obs["player"]]["audit"]["rows"][obs["step"]] if self.card==CARDS[0] else None)
        y=self.m.agent(obs,cfg)
        if obs["step"]==0:
            m=self.m
            assert self.frozen==dict(hybrid=m._HYBRID,partner_weights=array_hash(m._NETS[0].w),
                student_weights=array_hash(m._NETS[1].w),shared_source=m._SRC_SHA256,return_coef=array_hash({"coef":m._RETURN_THETA}))
            self.c["frozen_model_identity"]+=1
        assert self.c["encodes"]==2 and self.c["prediction_views"]==1
        n=2*sum(self.x["tok_mask"][:self.F.OWN])
        assert n==self.c["branch_slots"]==self.c["synchronized_advances"]
        assert self.c["shared_return_basis_calls"]==sum(self.x["tok_mask"][:10] & (self.x["tok_kind"][:10]==0))
        return y,dict(self.c)


def report():
    from held_schema import summarize
    rows=[]; inits=[]
    names=["decoder_student","decoder_constant"]
    for c in CARDS:
        for s in (0,1):
            by={}
            for name in names:
                by[name]={m:read(TMP/f"{name}_{m}_{c['seed']}_{s}.json.gz") for m in ("timing","audit")}
                meas=by[name]["timing"]
                assert meas["ready"]["pack_sha256"]==sha((HERE/(ARMS[name]+".py")).read_bytes())
                inits.append(dict(arm=name,seed=c["seed"],seat=s,init_external_ms=meas["init_ms"],ready=meas["ready"]))
            for t in range(299):
                row=dict(seed=c["seed"],t=t,seat=s,roles={},commands={},audit={},timing={})
                for name in names:
                    a,b=(by[name][key]["rows"][t] for key in ("timing","audit"))
                    assert a["action"]==b["action"]
                    row["commands"][name]=a["action"]
                    row["timing"][name]={k:v for k,v in a.items() if k!="action"}
                    row["audit"][name]=b["counts"]
                    if row["roles"]: assert row["roles"]==b["roles"]
                    row["roles"]=b["roles"]
                rows.append(row)
    meta=dict(packs=read(HERE/"packages.json"),protocol_sha256=sha((HERE/"protocol.json").read_bytes()),
              script_sha256=sha(Path(__file__).read_bytes()),report_helper_sha256=sha((HERE/"held_schema.py").read_bytes()),
              cards=CARDS,worker_inits=inits,environment=platform.platform(),
              method="Full archived trajectory on each physical seat/package; fresh isolated single-core timing subprocesses via serial JSONL, independent instrumented run with action parity on every call. Only the signed archived joint actions drive the simulated board. Counters refer to both native decode branches, including any branch later suppressed by the guard.")
    summary=summarize(rows,meta)
    save(HERE/"held_turns.json.gz",rows)
    summary["turns_sha256"]=sha((HERE/"held_turns.json.gz").read_bytes())
    save(HERE/"held_summary.json",summary)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("mode",choices=["timing","audit","report"])
    ap.add_argument("arm",nargs="?",choices=list(ARMS))
    ap.add_argument("cpu",nargs="?",type=int)
    args=ap.parse_args()
    if args.mode=="report":
        report()
    else:
        run_lane(args.arm,args.cpu,args.mode)

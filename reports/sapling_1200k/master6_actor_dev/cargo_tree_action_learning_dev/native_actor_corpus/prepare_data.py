"""Sample original expert turns and compute the fixed physical mixture terms before fitting."""
from common import *
from collections import Counter
import inspect
import time

RULE = read(HERE/"protocol.json")

def teacher_reader(m):
    src = inspect.getsource(m._return_decode)
    marker = "action = int(np.argmax(np.where(legal, combined, -np.inf)))"
    if src.count(marker) != 1:
        raise ValueError("decoder source mismatch")
    src = src.replace("encoded, apply_residual):", "encoded, apply_residual, y, fixed, parent, cut):")
    src = src.replace(marker, """if slot >= cut:
            return labels
        fixed[slot] = combined - _HYBRID["student_weight"] * scores[1]
        parent[slot] = combined
        action = int(y[slot])
        if not legal[action]:
            raise ValueError("invalid retained teacher prefix")""")
    ns = dict(vars(m))
    exec(compile(src, str(PACK) + "#logged-prefix", "exec"), ns)
    return ns["_return_decode"]

def derive(c,m,teacher,k):
    F = sys.modules["cnet2_feat"]
    seat = 1-c["seat"]
    f = F.Feat(seat)
    rows, meta = [], []
    counts = Counter()
    for t,obs,cfg,order,use in iter_public(c,k):
        x = f.observe(obs)
        if not use: continue
        # Select the guard from the parent's two free runs, before extracting teacher command labels.
        enc = []
        for n in m._NETS:
            tokens,ctx=n.encode(x,2)
            enc.append((n,tokens,ctx,n.lin(ctx,"h0")))
        a=m._return_decode(x,f.aux,enc,False)
        b=m._return_decode(x,f.aux,enc,True)
        sent,rejected=policy.guard(m,x,a,b)
        r=actual_features(f,x,order,m)
        fixed=np.zeros_like(r["residual"])
        parent=np.zeros_like(fixed)
        teacher(x,f.aux,enc,not rejected,r["y"],fixed,parent,r["cut"])
        # fixed already includes this root's guard-selected residual.
        r.pop("residual"); r["fixed"]=fixed; r["parent"]=parent
        mask=x['tok_mask'][:F.OWN]
        cut=r.pop("cut")
        counts["sampled_roots"]+=1
        counts["active_slots"]+=int(mask.sum())
        counts["representable_actionable"]+=int(r["eligible"].sum())
        counts["constrained_single_legal"]+=int((mask[:cut] & (r['legal'][:cut].sum(1)==1)).sum())
        counts["roots_with_bad_prefix"]+=int(cut<F.OWN)
        counts["bad_slot_and_blocked_suffix"]+=int(mask[cut:].sum())
        counts["residual_guard_selected"]+=int(not rejected)
        meta.append(dict(t=t,game=c["sha256"],seed=c["seed"],terrain=c["terrain"],expert_seat=seat,cut=int(cut),
                         suffix_active=int(mask[cut:].sum()),representable_actionable=int(r["eligible"].sum()),
                         native_parent_action=f.to_actions(sent),guard_rejected=rejected))
        rows.append(r)
    return {k:np.stack([r[k] for r in rows]) for k in rows[0]},meta,dict(counts)

def main():
    source=read(HERE/"sources.json")
    cards=sorted((c for c in source["cards"] if c["split"]=="train"),key=lambda c:c["sha256"])
    if len(cards)!=26 or sha(EXP/"panel.json")!=RULE["panel_sha256"] or sha(BASE)!=RULE["weights_sha256"]:
        raise ValueError("frozen sources changed")
    out=HERE/"work";out.mkdir(exist_ok=True)
    m=package();teacher=teacher_reader(m)
    receipts=[];clock=time.monotonic()
    for i,c in enumerate(cards):
        name=c["sha256"][:24]
        data=out/(name+".npz"); registry=out/(name+".json")
        version={q:sha(HERE/q) for q in ("common.py","prepare_data.py","protocol.json","sources.json")}
        if data.is_file() and registry.is_file():
            rec=read(registry)
            if rec["version"]!=version or rec["data_sha"]!=sha(data):
                raise ValueError("stale materialized features")
        else:
            d,rows,counts=derive(c,m,teacher,RULE["roots_per_tape"]["train"])
            with (out/(name+".tmp")).open("wb") as f: np.savez_compressed(f,**d)
            os.replace(out/(name+".tmp"),data)
            rec=dict(version=version,game=c["sha256"],rows=rows,counts=counts,data_sha=sha(data),file=data.name)
            save(registry,rec)
        receipts.append(rec)
        print(i+1, c["seed"],c["expert_seat"],rec["counts"],round(time.monotonic()-clock,1),flush=True)
    summary=dict(tapes=len(receipts),counts=dict(sum((Counter(r["counts"]) for r in receipts),Counter())),
                 sources_sha=sha(HERE/"sources.json"),protocol_sha=sha(HERE/"protocol.json"),
                 shards=[{k:r[k] for k in ("game","file","data_sha","counts")} for r in receipts])
    save(HERE/"training_roots.json.gz",[x for r in receipts for x in r["rows"]])
    save(HERE/"data.json",summary)

if __name__=="__main__":main()

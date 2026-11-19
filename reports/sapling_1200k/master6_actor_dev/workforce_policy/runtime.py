"""Conservative one-use substitution on the observed v1 over-recruitment state."""
_WF_USED=False
_WF_LAST={}
_WF_MU=np.asarray(_WF_MODEL["mu"],dtype=np.float64)
_WF_SD=np.asarray(_WF_MODEL["scale"],dtype=np.float64)
_WF_COEF=np.asarray(_WF_MODEL["coefficients"],dtype=np.float64)


def _workforce(x,aux,enc,parent,branch):
    global _WF_USED,_WF_LAST
    step=299-int(x["left"])
    if step==0:
        _WF_USED=False
    _WF_LAST={"step":step,"reason":"outside_or_already_used","changed":False}
    if _WF_USED or not 8<=step<=170:return parent
    nplanter=int(np.count_nonzero(x["tok_mask"][:10]&(x["tok_kind"][:10]==1)))
    if nplanter<2 or not np.any(parent[10:12]==2):
        _WF_LAST["reason"]="no_redundant_planter_order";return parent
    F=sys.modules["cnet2_feat"]
    original,rows=_wf_decode(x,aux,enc,branch,teacher=parent)
    if original is None or not np.array_equal(original,parent):
        _WF_LAST["reason"]="unreproducible_parent";return parent
    _WF_LAST["reason"]="no_safe_confident_change"
    for r in rows:
        s=r["slot"];info=r["info"]
        if parent[s]!=2 or not all(r["legal"]):continue
        if info["risk"] or info["occupied"] or info["free"]<=0:continue
        z=np.clip((r["features"]-_WF_MU)/_WF_SD,-4,4)
        prob=r["scores"]+np.concatenate(([0.],np.clip(z@_WF_COEF,-3,3)))
        prob=np.exp(prob-prob.max());prob/=prob.sum()
        alt=int(prob.argmax())
        if alt==2 or prob[alt]<.5 or prob[alt]-prob[2]<.1:continue
        if alt==0 and info["ncollector"]<1:continue
        # Earlier bases already spent out of info.cash. Reserve enough for unchanged suffix and observable payable worker orders.
        other_hires=sum(int(parent[j]>0) for j in range(s+1,F.OWN) if x["tok_mask"][j])
        if info["cash"]-info["spend"]-30*(other_hires+(alt!=0))<30:continue
        forced,_=_wf_decode(x,aux,enc,branch,forced_slot=s,forced=alt,record=False)
        if forced is None or np.flatnonzero(forced!=parent).tolist()!=[s]:continue
        if _return_conflicts(x,forced)-_return_conflicts(x,parent):continue
        _WF_USED=True
        _WF_LAST={"step":step,"reason":"committed","changed":True,"slot":s,"base":_FEAT.own_uid[s],
                  "original":F.BASE_ACTS[2],"alternative":F.BASE_ACTS[alt],"probabilities":prob.tolist(),"info":info,
                  "existing_planters":nplanter}
        return forced
    return parent

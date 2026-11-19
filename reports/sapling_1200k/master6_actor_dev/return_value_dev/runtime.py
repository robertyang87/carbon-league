"""Every candidate is checked with the exact same autoregressive legality mask as its parent."""
_RV_ORIGINAL=agent
_RV_DECODE=_mixed_actions
_RV_CONTEXT=None
_RV_TRACE={}

def _mixed_actions(x,aux):
    global _RV_CONTEXT
    _RV_CONTEXT=(x,aux)
    return _RV_DECODE(x,aux)

def _rv_adopt(ranked, base, feat, x, aux, fm):
    for row in ranked:
        modified={**base,row["uid"]:row["alternative"]}
        awards=feat.labels(modified)
        prefix=fm.Prefix(x,aux)
        for idx,act in enumerate(awards):
            if act<0:continue
            if not prefix.feats(idx)[2][act]: break
            prefix.advance(idx,int(act))
        else:return row,modified
    return None,base

def agent(observation,configuration=None):
    global _RV_TRACE
    base=_RV_ORIGINAL(observation,configuration)
    ranked=_RV_MODEL.ranked(observation,base,configuration,_RV_W) if configuration is not None else []
    row,issued=_rv_adopt(ranked,base,_FEAT,*_RV_CONTEXT,sys.modules["cnet2_feat"])
    _RV_TRACE={"base":dict(base),"selected":row,"ranked":len(ranked)}
    return issued

"""Build-time suffix for an optional single guarded freight intervention on production Sapling."""
_CARGO_PARENT_AGENT=agent
_CARGO_ORIG_MIX=_mixed_actions
_CARGO_ROWS={}
_CARGO_PASSES=[]
_CARGO_CTX=None
_CARGO_SEAT=None
_CARGO_T=None
_CARGO_USED=False
_CARGO_TRACE=None


def _cargo_record(slot, scores, legal):
    _CARGO_ROWS[int(slot)]=tuple(float(s) if ok else None for s,ok in zip(scores,legal))


_CARGO_ORIG_DECODER=_return_decode


def _return_decode(x,aux,encoded,apply_residual):
    if not _CARGO_SCORING:
        return _CARGO_ORIG_DECODER(x,aux,encoded,apply_residual)
    _CARGO_ROWS.clear()
    y=_cargo_scored_decode(x,aux,encoded,apply_residual)
    _CARGO_PASSES.append((y,dict(_CARGO_ROWS)))
    return y


_CARGO_SCORING=False


def _mixed_actions(x,aux):
    global _CARGO_CTX
    _CARGO_PASSES.clear()
    labels=_CARGO_ORIG_MIX(x,aux)
    if _CARGO_SCORING:
        scores,=[s for y,s in _CARGO_PASSES if y is labels]
        _CARGO_CTX=(x,aux,labels.copy(),scores)
        _CARGO_PASSES.clear()
    return labels


def agent(observation, configuration=None):
    global _CARGO_CTX,_CARGO_SEAT,_CARGO_T,_CARGO_USED,_CARGO_TRACE,_CARGO_SCORING
    view=_CARGO_ACTIONS.public(observation)
    t,seat=view["step"],view["player"]
    if t==0 or seat!=_CARGO_SEAT or (_CARGO_T is not None and t<_CARGO_T):
        _CARGO_USED=False
    _CARGO_SEAT,_CARGO_T=seat,t
    _CARGO_CTX=None
    _CARGO_SCORING=bool(_CARGO_ENABLED and not _CARGO_USED and 180<=t<=295)
    parent=_CARGO_PARENT_AGENT(view,configuration)
    chosen=None
    result=parent
    if not _CARGO_ENABLED:
        why="disabled"
    elif not 180<=t<=295:
        why="outside_window"
    elif _CARGO_USED:
        why="spent"
    else:
        cfg={k:(configuration or {}).get(k,v) for k,v in _CARGO_CFG.items()}
        state=types.SimpleNamespace(_loaded_trace=_CARGO_CTX,_FEAT=_FEAT,_local_edit=_local_edit)
        choices=_CARGO_ACTIONS.candidates(view,cfg,parent,state)
        chosen=next((p for p in choices if p["threat"]["strict"]),None)
        if chosen is None:
            why="no_strict_safe_alternative"
        else:
            x,aux,_,_=_CARGO_CTX
            result,why=_local_edit(view,cfg,parent,{chosen["uid"]:chosen["action"]},x,aux,_FEAT)
            if why!="accepted":
                raise ValueError("validated protected order changed in a single call")
            _CARGO_USED=True
    _CARGO_TRACE={"seat":seat,"step":t,"parent":parent,"chosen":chosen,"reason":why}
    _CARGO_SCORING=False
    _CARGO_CTX=None
    return result

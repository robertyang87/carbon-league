"""Optional one-step mortality veto after the original whole-pay selector has consumed its window."""
_B_A_AGENT = agent
_B_LAST = None


def _B_distance(left,right,size):
    y,x=divmod(left,size)
    v,w=divmod(right,size)
    dy,dx=abs(y-v),abs(x-w)
    return min(dy,size-dy)+min(dx,size-dx)


def agent(observation,configuration=None):
    global _B_LAST
    public={k:observation[k] for k in ("step","player","players","trees","carbon")}
    cursor=0 if public["step"]==0 else len(_PAY_TRACE)
    action=_B_A_AGENT(public,configuration)
    _B_LAST=None
    if len(_PAY_TRACE)==cursor:
        return action
    if len(_PAY_TRACE)!=cursor+1:
        raise ValueError("A selected multiple opportunities on one turn")
    trace=_PAY_TRACE[-1]
    if trace["window"] not in _PAY_USED:
        raise ValueError("A failed to consume its opportunity")
    workers=public["players"][1-public["player"]][2].values()
    size=(configuration or {}).get("size",15)
    distances={role:min((_B_distance(trace["cell"],w[0],size) for w in workers if w[2]==role),default=99)
               for role in ("COLLECTOR","PLANTER")}
    dc,dp=distances["COLLECTOR"],distances["PLANTER"]
    restore=_B_RISK.pick_parent(_B_WEIGHTS,trace["changed"],dc,dp,trace["kind"])
    _B_LAST={"window":trace["window"],"t":trace["t"],"uid":trace["uid"],"cell":trace["cell"],"kind":trace["kind"],
             "A_changed":trace["changed"],"restore_parent":restore,"dc":dc,"dp":dp,
             "risk":_B_RISK.estimate(_B_WEIGHTS,dc,dp,trace["kind"])}
    return dict(trace["base"]) if restore else action

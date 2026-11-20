"""Freeze the trained conditional rent estimate from the registered root, independently of either suffix."""
from study import HERE,advance,check,dump,read,setup,sha,source,tree,views

def run():
    selection=read(HERE/"selection.json")
    if not selection.get("ticket"):
        return
    t=selection["ticket"];s=t["seat"];k=t["step"];uid=selection["intervention"]["info"]["worker"]
    _,acts,cfg,env,sim=setup(t,source(t));past={}
    for i,orders in enumerate(acts[:k]):
        if k-i in (3,5):past[k-i]=views(env)[s]["players"]
        advance(env,sim,orders)
    obs=views(env)[s];x=tree.features(obs,uid,cfg,past)
    wp=HERE/"rent/weights.json"
    check(sha(wp.read_bytes())=="0a9fb5f2bcafae18cf1a25fafc72e9fe9d6b84ac7e75c27155a93e3282ea42ab","Wrong readout")
    h=read(wp)
    z=h["intercept"]+sum(w*(v-a)/b for v,w,a,b in zip(x,h["beta"],h["center"],h["scale"],strict=True))
    pred=max(0.,z) if cfg["episodeSteps"]-1-k>8 else 0.
    dump("root_rent_forecast.json",dict(seed=t["seed"],seat=s,step=k,worker=uid,cell=obs["players"][s][2][uid][0],
         quoted_price=tree.quoted_price(obs["players"],s,cfg),predicted_tail_if_successfully_planted=pred,
         zero_slopes_same_intercept=h["intercept"],preaction_features=x,weights=sha(wp.read_bytes()),
         source_root=sha((HERE/"selection.json").read_bytes()),
         forecast_target="Conditional tree cash during first tenure from ninth transition onwards. No leave-arm or official-margin prediction."))

if __name__=="__main__":run()

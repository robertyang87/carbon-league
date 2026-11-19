"""Map-weighted, arm-specific regressions. Fit/calibration and blind readout are separate calls."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.special import expit, logit
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from economic import NAMES,_predict
from plan import HERE,PRE,read,save,sha

def dataset(want):
    panel=read(HERE/"plan.json")["maps"];receipt=read(HERE/"label_receipt.json");out=[]
    for i,g in enumerate(panel):
        if g["split"] not in want:continue
        path=HERE/"labels"/receipt["files_in_map_order"][i]
        assert sha(path)==receipt["source_sha256"][i]
        with np.load(path,allow_pickle=False) as z:
            out.append((z["x"],z["y"],np.full(len(z["x"]),i),z["ident"]))
    return [np.concatenate(x) for x in zip(*out)]

def mapw(m):
    ids,idx,counts=np.unique(m,return_inverse=True,return_counts=True)
    return len(m)/(len(ids)*counts[idx])

def weighted(y,w):
    return float(np.average(y,weights=w))

def export(est,kind,x):
    if kind=="death":raw=est.decision_function(x[:8])
    else:raw=est.predict(x[:8])
    trees=[]
    treesums=np.zeros(len(raw))
    for (tree,) in est.estimators_:
        a=tree.tree_
        trees.append({"l":a.children_left.tolist(),"r":a.children_right.tolist(),"t":a.threshold.tolist(),
                      "f":a.feature.tolist(),"v":(est.learning_rate*a.value[:,0,0]).tolist()})
        treesums+=est.learning_rate*tree.predict(x[:len(raw)])
    initial=float((raw-treesums)[0])
    out={"initial":initial,"trees":trees}
    assert np.allclose([_predict(out,row) for row in x[:len(raw)]],raw,rtol=1e-10,atol=1e-9)
    return out

def train():
    x,y,m,orders=dataset(["train"]); cx,cy,cm,corders=dataset(["calibration"])
    arms={}
    for moving,kind in ((False,"stay"),(True,"move")):
        ix=(orders[:,4]!=0)==moving;ci=(corders[:,4]!=0)==moving
        a,b,g=x[ix],y[ix],m[ix];c,d,u=cx[ci],cy[ci],cm[ci];w=mapw(g);cw=mapw(u)
        common=dict(n_estimators=56,learning_rate=.08,max_depth=3,min_samples_leaf=25,random_state=7115)
        death=GradientBoostingClassifier(**common).fit(a,b[:,0],sample_weight=w)
        net=GradientBoostingRegressor(loss="squared_error",**common).fit(a,b[:,1],sample_weight=w)
        cal=LogisticRegression(C=.1,max_iter=300).fit(death.decision_function(c)[:,None],d[:,0],sample_weight=cw)
        dslope=float(cal.coef_[0,0]);dint=float(cal.intercept_[0])
        z=net.predict(c);zm=weighted(z,cw);target=weighted(d[:,1],cw)
        # Identity-centered ridge of strength one percent of calibration variance, preserving units.
        var=weighted((z-zm)**2,cw);nslope=(weighted((z-zm)*(d[:,1]-target),cw)+.01*var)/(1.01*var) if var else 1.
        nint=target-nslope*zm
        arms[kind]={"death":export(death,"death",a),"net":export(net,"net",a),"death_cal":[dslope,dint],
                    "net_cal":[nslope,nint],"baseline_death":weighted(b[:,0],w),"baseline_net":weighted(b[:,1],w),
                    "train_n":len(a),"training_maps":len(set(g)),"calibration_n":len(c),"calibration_maps":len(set(u))}
    save(HERE/"model.json",{"plan_sha256":sha(HERE/"plan.json"),"labels_sha256":sha(HERE/"label_receipt.json"),
                           "features":NAMES,"arms":arms})

def evalset(title,mask,results,models):
    y,m,o,p,n,bq,bn=results
    y,m,o,p,n,bq,bn=y[mask],m[mask],o[mask],p[mask],n[mask],bq[mask],bn[mask]
    w=mapw(m)
    def means(x):return weighted(x,w)
    def ll(v):return means(-y[:,0]*np.log(v)-(1-y[:,0])*np.log1p(-v))
    return {"maps":len(set(m)),"n":len(m),"actual_deaths":int(y[:,0].sum()),"actual_purchases":int(y[:,2].sum()),
        "actual_net":means(y[:,1]),"predicted_net":means(n),"death_brier":means((y[:,0]-p)**2),
        "baseline_brier":means((y[:,0]-bq)**2),"death_logloss":ll(p),"baseline_logloss":ll(bq),
        "net_mae":means(abs(y[:,1]-n)),"baseline_mae":means(abs(y[:,1]-bn)),
        "net_rmse":means((y[:,1]-n)**2)**.5,"baseline_rmse":means((y[:,1]-bn)**2)**.5,
        "death_calibration":[{"n":int(len(q)),"forecast":weighted(p[q],w[q]),"observed":weighted(y[q,0],w[q])}
            for lo,hi in zip([0,.05,.2,.5,.8],[.05,.2,.5,.8,1.01]) if (q:=np.where((lo<=p)&(p<hi))[0]).size]}

def evaluate():
    model=read(HERE/"model.json");splitrows={};grouped={}
    maps=read(HERE/"plan.json")["maps"]
    facts=[]
    for split in ("new_opponent","known_new_map"):
        x,y,m,o=dataset([split]);p=np.zeros(len(x));n=p.copy();bq=p.copy();bn=p.copy()
        for moving,kind in ((False,"stay"),(True,"move")):
            ix=(o[:,4]!=0)==moving;s=model["arms"][kind]
            ds,di=s["death_cal"];ns,ni=s["net_cal"]
            p[ix]=expit(ds*np.array([_predict(s["death"],row) for row in x[ix]])+di)
            n[ix]=ns*np.array([_predict(s["net"],row) for row in x[ix]])+ni
            bq[ix]=s["baseline_death"];bn[ix]=s["baseline_net"]
        data=(y,m,o,p,n,bq,bn)
        masks={"all":np.ones(len(x),bool),"stay":o[:,4]==0,"move":o[:,4]!=0}
        splitrows[split]={k:evalset(k,mask,data,model) for k,mask in masks.items()}
        grouped[split]={str(i):evalset("one_map",m==i,data,model) for i in sorted(set(m))}
        for i in np.argsort(np.abs(y[:,1]-n))[-3:][::-1]:
            facts.append({"map":maps[m[i]]["terrain"],"id":maps[m[i]]["id"],"seat":maps[m[i]]["seat"],"worker":int(o[i,1]),
                "t":int(o[i,0]),"origin":int(o[i,2]),"dest":int(o[i,3]),"action_code":int(o[i,4]),"predicted_net":n[i],
                "actual_first_tenure_net":y[i,1],"first_tenure_gross":y[i,3],"purchase_fee":y[i,4],"actual_death":int(y[i,0])})
    save(HERE/"factual_holdout.json",{"model_sha256":sha(HERE/"model.json"),"by_split":splitrows,"map_scores":grouped,
                                     "absolute_net_error_examples":facts,"score_unit":"mean of per-map means"})
    for s,r in splitrows.items():
        print(s,r["all"],flush=True)
if __name__=="__main__":
    {"fit":train,"eval":evaluate}[sys.argv[1]]()

"""Multivariate simplex-preserving empirical leaf distributions, one unit of loss weight per map."""
import json
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from protocol import HERE,sha
from data import CACHE,generate,write
import head

def load(split):
    xs,axs,ys,es,olds,keys,ids,seats=[],[],[],[],[],[],[],[]
    receipt=json.loads((HERE/(split+"_sources.json")).read_bytes())
    for g in receipt:
        p=CACHE/g["file"]
        if sha(p)!=g["npz_sha256"]:raise ValueError("invalid training cache")
        with np.load(p,allow_pickle=False) as z:
            for name,arr in (("x",xs),("xa",axs),("y",ys),("extra",es),("old",olds),("keys",keys)):arr.append(z[name])
            ids.extend([g["seed"]]*g["roots"]);seats.extend([g["seat"]]*g["roots"])
    return tuple(np.concatenate(rows) for rows in (xs,axs,ys,es,olds,keys))+(np.array(ids),np.array(seats))

def train_model(x, xa, y, ids, baseline):
    features=[0,1,2] if baseline else list(range(len(head.NAMES)))
    xx=np.concatenate([x[:,features],xa[:,features]]).astype(np.float32)
    yy=np.concatenate([y,y])
    # Split selection cares about distributions as well as timing. Scale time moments
    # by horizon width to keep 100-step magnitudes from dominating probabilities.
    n=len(head.CATEGORIES); h=len(head.HORIZONS)
    scales=np.ones(y.shape[1]);scales[n+h:]=1/np.array(head.HORIZONS)
    u,counts=np.unique(ids,return_counts=True);table=dict(zip(u, np.mean(counts)/counts))
    ww=np.tile([table[i] for i in ids],2)
    forest=ExtraTreesRegressor(n_estimators=16,max_depth=8,min_samples_leaf=60,max_features=.75,
          random_state=11962026,n_jobs=1,bootstrap=False)
    forest.fit(xx,yy*scales,sample_weight=ww)
    obj=dict(version=1,features=list(head.NAMES),causes=list(head.CATEGORIES),horizons=list(head.HORIZONS),
          training_protocol=sha(HERE/"protocol.json"),training_receipt=sha(HERE/"train_sources.json"),trees=[])
    for e in forest.estimators_:
        tree=e.tree_;nodes=[];values=[]
        for i in range(tree.node_count):
            if tree.feature[i]<0:
                nodes.append([-len(values)-1,0,0,0])
                values.append([float(z) for z in tree.value[i,:,0]/scales])
            else:nodes.append([features[int(tree.feature[i])],float(tree.threshold[i]),int(tree.children_left[i]),int(tree.children_right[i])])
        obj["trees"].append(dict(nodes=nodes,values=values))
    # Check exported model with sklearn on deterministic, diverse training indices.
    ids=np.linspace(0,len(xx)-1,40,dtype=int)
    pred=forest.predict(xx[ids])/scales
    full=np.concatenate([x,xa])[ids]
    for z,(row,expected) in enumerate(zip(full,pred)):
        expected_cash=row[0]*expected[:2].sum()
        if abs(head.from_features(obj,row)["expected_cash"]-expected_cash)>1e-6:
            raise ValueError("serialized predictor differs from sklearn")
    return obj
if __name__=="__main__":
    generate("train")
    x,xa,y,_,_,_,ids,_=load("train")
    for b,file in ((False,"weights.json.gz"),(True,"distance_control.json.gz")):
        obj=train_model(x,xa,y,ids,b); write(HERE/file,obj)

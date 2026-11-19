"""Train/preregistered mortality calibration on realised top-five stays, without held labels."""
import gzip
import math
from collections import Counter
import numpy as np

from registry import HERE,ACTOR,read,sha,save
import risk

DATA=ACTOR/'planter_intent_labels'

def get_data():
    split=read(HERE/'splits.json.gz')
    declarations={r['id']:r for r in split['maps']}
    anchors=read(DATA/'maps.json.gz')['maps']
    assert len(anchors)==len(declarations)==1542
    si=[];phys=[]; names=[];tset=[]
    for k,r in enumerate(anchors):
        d=declarations[r['id']]
        assert r['map_index']==k and r['terrain']==d['terrain'] and r['sha256']==d['raw_sha256']
        si.append(['train','new_opponent','known_new_map'].index(d['split']));phys.append(d['seat'])
        names.append(d['opponent']);tset.append(d['terrain'])
    assert not {n for n,s in zip(names,si) if s==0}&{n for n,s in zip(names,si) if s==1}
    assert len(tset)==len(set(tset))
    phys=np.asarray(phys);si=np.asarray(si)
    fields=('map_index','enemy_collector_dist','enemy_planter_dist','kind','y_survived','y_paid')
    table={k:[] for k in fields}
    parts=sorted(DATA.glob('labels_*.npz'))
    full=read(DATA/'coverage.json');checked=read(DATA/'verification.json')
    assert len(parts)==25 and {f.name:sha(f) for f in parts}==checked["shard_sha256"]=={f["path"]:f["sha256"] for f in full["files"]}
    for k in ("coverage.json","maps.json.gz","schema.json"):
        assert sha(DATA/k)==checked["source_sha256"][k]
    assert full["maps_sha256"]==sha(DATA/"maps.json.gz") and full["schema_sha256"]==sha(DATA/"schema.json")
    intended=set()
    for f in parts:
        with np.load(f,allow_pickle=False) as z:
            mi=z['map_index'];intended.update(np.unique(mi).tolist())
            keep=(z['seat']==phys[mi])&(z['action']==0)&(z['affordable']==1)
            for k in fields:table[k].append(z[k][keep])
    assert len(intended)==1542
    table={k:np.concatenate(v) for k,v in table.items()}
    count=np.bincount(table['map_index'],minlength=len(anchors))
    table.update(split=si[table['map_index']],death=1.-table['y_survived'].astype(float),weight=1./count[table['map_index']])
    table['near']=(np.minimum(table['enemy_collector_dist'],table['enemy_planter_dist'])<=1).astype(int)
    table['groups']=6*np.minimum(table['enemy_collector_dist'],2)+2*np.minimum(table['enemy_planter_dist'],2)+table['kind']
    return table,anchors,names,parts

def model(table):
    a=table['split']==0
    w=table['weight'][a]; y=table['death'][a]; total=w.sum()
    prior=float(np.dot(w,y)/total)
    def conditional(k,n):
        v=table[k][a];mass=np.bincount(v,weights=w,minlength=n); deaths=np.bincount(v,weights=w*y,minlength=n)
        alpha=10*mass/total
        return [float((d+p*prior)/(m+p)) if m else prior for d,m,p in zip(deaths,mass,alpha)],mass.tolist()
    rr,mass=conditional('groups',18);br,_=conditional('near',2)
    return {"global":prior,"near":br,"risk":rr,"group_map_equivalents":mass,"train_rows":int(a.sum()),
            "train_maps":len(set(table["map_index"][a].tolist())),"threshold":0.50,
            "protocol_sha256":sha(HERE/'protocol.json'),"upstream_verification_sha256":sha(DATA/'verification.json')}

def calibrated(table,anchors,names,m):
    result={}
    for i,key in enumerate(('train','new_opponent','known_new_map')):
        idx=table['split']==i
        mat={n:v[idx] for n,v in table.items()};w=mat['weight'];y=mat['death'];maps=mat['map_index']
        globalpred=np.full(len(w),m['global'])
        probs={'global':globalpred,'near_only':np.asarray(m['near'])[mat['near']],
               'role_and_quote':np.asarray(m['risk'])[mat['groups']]}
        losses={}
        recal={}
        for label,p in probs.items():
            safe=np.clip(p,1e-9,1.-1e-9)
            losses[label]={"brier":float(np.average((p-y)**2,weights=w)),"logloss":float(np.average(-y*np.log(safe)-(1-y)*np.log1p(-safe),weights=w))}
            bins=np.minimum((p*5).astype(int),4)
            recal[label]=[{"n":int(k.sum()),"map_mass":float(w[k].sum()),"predicted":float(np.average(p[k],weights=w[k])),
                           "observed_death":float(np.average(y[k],weights=w[k]))} for b in range(5) if (k:=(bins==b)).any()]
        def frequency(mask):
            if not mask.any():return None
            return {"n":int(mask.sum()),"paid":int(mat['y_paid'][mask].sum()),"died":int(y[mask].sum()),
                    "maps":len(set(maps[mask].tolist())),"map_weight_death":float(np.average(y[mask],weights=w[mask])),
                    "map_weight_paid":float(np.average(mat['y_paid'][mask],weights=w[mask]))}
        rival={};by_map=[]
        for index in sorted(set(maps.tolist())):
            take=maps==index
            by_map.append({"map":anchors[index]['id'],"opponent":names[index],"actual_stays":int(take.sum()),
                           "near":frequency(take&(mat['near']==1)),"far":frequency(take&(mat['near']==0))})
        for n in sorted({names[k] for k in set(maps.tolist())}):
            chosen=np.isin(maps,[k for k in set(maps.tolist()) if names[k]==n])
            rival[n]={"near":frequency(chosen&(mat['near']==1)),"far":frequency(chosen&(mat['near']==0))}
        result[key]={"rows":len(maps),"physical_maps":len(set(maps.tolist())),"opponents":len(rival),"risk":losses,"calibration":recal,
                     "near":frequency(mat['near']==1),"far":frequency(mat['near']==0),"by_opponent":rival,"by_map":by_map}
    return result

def main():
    table,anchors,names,parts=get_data();m=model(table)
    save(HERE/'risk_weights.json',m)
    doc=calibrated(table,anchors,names,m)
    save(HERE/'observational.json.gz',{"model_sha256":sha(HERE/'risk_weights.json'),"training_judgment":"death on realised affordable stay; outside choices are not intervention outcomes",
                                      "per_split":doc,"shards":{f.name:sha(f) for f in parts}})
    for k,z in doc.items():print(k,z['rows'],z['physical_maps'],z['risk'])

if __name__=='__main__':main()

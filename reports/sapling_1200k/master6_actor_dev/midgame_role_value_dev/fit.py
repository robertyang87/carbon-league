"""Split by physical map; compare separate role-value heads with constant per-role baselines."""
import numpy as np
from data import sr,HERE
FIELDS=('phase','own_collectors','own_planters','cash','own_trees','enemy_trees','tree_price')
def features(b):
    return [b['t']/100,b['workersC']/5,b['workersP']/3,np.log1p(b['pre_cash']),b['trees']/10,b['enemy_trees']/10,np.log1p(b['tree_price'])]
def predict(head,b):
    x=(np.array(features(b))-head['mu'])/head['sd']
    return head['offset']+x@head['weights']
def main():
    games=sr.load(HERE/'hires.json.gz');heads={};metrics={}
    for role in ('C','P'):
        train=[r for r in games if r['role']==role and r['seed']<=621758324]
        test=[r for r in games if r['role']==role and r['seed']>621758324]
        x=np.array([features(r['base']) for r in train]);y=np.array([r['net50'] for r in train])
        mu=x.mean(0);sd=np.maximum(x.std(0),.1);x=(x-mu)/sd
        h=dict(mu=mu.tolist(),sd=sd.tolist(),offset=float(y.mean()),weights=np.linalg.solve(x.T@x+10*np.eye(x.shape[1]),x.T@(y-y.mean())).tolist())
        heads[role]=h;est=np.array([predict(h,r['base']) for r in test]);target=np.array([r['net50'] for r in test])
        metrics[role]=dict(train=len(train),test=len(test),observed_mean=float(target.mean()),mae=float(np.abs(target-est).mean()),constant_mae=float(np.abs(target-y.mean()).mean()))
    sr.retain(HERE/'model.json',dict(features=FIELDS,heads=heads,heldout=metrics))
    print(metrics)
if __name__=='__main__':main()

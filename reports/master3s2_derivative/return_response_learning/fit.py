"""Logged-trajectory training, then true-response-pair calibration, with physically disjoint maps."""
from pathlib import Path
import gzip,hashlib,json,sys,time
import numpy as np
from scipy.optimize import minimize
from model import rr_features,rr_predict,rr_scores

HERE=Path(__file__).resolve().parent;DER=HERE.parent;ROOT=DER.parents[1];PAIR=DER/'action_rank_data'
sys.path.insert(0,str(ROOT/'reports/sapling_1200k'))
from turning_point_ledger import CashEvents,FastCarbon

def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def save(name,obj):
    p=HERE/name;p.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n');return sha(p.read_bytes())

def samples(cards):
    dataset=[]
    for card in cards:
        p=ROOT/card['path'];assert sha(p.read_bytes())==card['sha256'];trace=read(p);seed=card['seed'];s=card['expert_seat'];assert s==1-card['candidate_seat']
        sim=FastCarbon(trace['game_configuration']).reset(seed=seed);assert sha(bytes(map(int,sim.carbon)))==card['terrain'];sim.cash=CashEvents(sim.cash);cash=[list(sim.cash)];pending=[];deposits={}
        for t,joint in enumerate(trace['actions']):
            if 24<=t<=294 and t%3==0:
                X,carriers=rr_features(sim._players_obs(),s,t,joint[s]);pending.extend([(t,u,load,x) for x,(u,load) in zip(X,carriers)])
            sim.cash.events.clear();sim.step(*joint);cash.append(list(sim.cash))
            for event in sim.cash.events:
                if event[0]=='own_delivery' and event[1]==s and event[2]>0:deposits.setdefault(event[3],[]).append((t,event[2]))
        assert sha(json.dumps(cash,sort_keys=True,separators=(',',':')).encode())==card['official_cash_path_sha256'],('cash',seed,s)
        for t,u,load,x in pending:
            actual=next((v for j,v in deposits.get(u,[]) if t<=j<min(299,t+12)),0.);dataset.append((card['terrain'],x,min(1.,actual/load),load))
    return dataset

def arrays(data):
    ids=np.array([d[0] for d in data]);groups=np.unique(ids);w=np.zeros(len(ids));
    for g in groups:
        ix=ids==g;w[ix]=1./(len(groups)*sum(ix))
    return np.array([d[1] for d in data]),np.array([d[2] for d in data]),w,ids

def fitnet(data,cols):
    X,y,w,_=arrays(data);x=X[:,cols];n=x.shape[1];h=8;first=np.random.default_rng(37011).normal(0,.22,(n,h));z0=np.r_[first.ravel(),np.zeros(h),np.full(h,.03),0.]
    def unpack(z):return z[:n*h].reshape(n,h),z[n*h:n*h+h],z[n*h+h:-1],z[-1]
    def objective(z):
        W,b,O,c=unpack(z);H=np.tanh(x@W+b);q=1/(1+np.exp(-np.clip(H@O+c,-35,35)));delta=(2*w*(q-y)*q*(1-q));dh=delta[:,None]*O*(1-H*H)
        loss=np.dot(w,(q-y)**2)+.001*(np.sum(W*W)+np.dot(O,O));grad=np.r_[(x.T@dh+.002*W).ravel(),dh.sum(axis=0),H.T@delta+.002*O,delta.sum()];return loss,grad
    fit=minimize(objective,z0,method='L-BFGS-B',jac=True,options={'maxiter':70,'ftol':1e-10});W,b,O,c=unpack(fit.x);return dict(columns=list(cols),w0=W.tolist(),b0=b.tolist(),w1=O.tolist(),b1=float(c),iterations=int(fit.nit),loss=float(fit.fun))

def ordinary_report(data,heads):
    X,y,w,ids=arrays(data);preds={name:rr_predict(X,head) for name,head in heads.items()};preds['distance']=np.clip((12-X[:,9]*8)/12,0,1);out={}
    for k,v in preds.items():
        by={str(i):dict(n=int(sum(ids==i)),mse=float(np.mean((v[ids==i]-y[ids==i])**2)),prediction=float(v[ids==i].mean()),observed=float(y[ids==i].mean())) for i in np.unique(ids)}
        out[k]=dict(brier=float(w@((v-y)**2)),n=len(y),maps=by)
    return out

def response_pairs(cards,heads,beta=None):
    output=[]
    for c in cards:
        stem=f"{c['seed']}_{c['seat']}_{c['step']}";folder=PAIR/c['split'];pair=read(folder/(stem+'_pair.json'));rr=read(folder/(stem+'_root.json.gz'))
        if pair['status']!='complete':output.append(dict(seed=c['seed'],terrain=c['terrain'],status=pair['status']));continue
        assert pair['root_sha256']==sha((folder/(stem+'_root.json.gz')).read_bytes());rs=rr['public_root'];s=c['seat'];assert (c['step'],s)==(rs['step'],rs['player'])
        commands=[arm['order'] for arm in rr['arms']];scores=rr_scores(rs['players'],s,c['step'],commands,heads);native=[a['original_value'] for a in rr['arms']];label=pair['target'];margins=[];wdl=[];deposits=[]
        for arm in ['first','second']:
            file=folder/(stem+'_'+arm+'.json.gz');e=pair['arms'][arm];assert sha(file.read_bytes())==e['evidence_sha256'];branch=read(file);margins.append(e['margin']);wdl.append(e['win']);deposits.append([sum(r['flow'][side].get('own_delivery',0.) for r in branch['rows'][:12]) for side in [s,1-s]])
        a,b=deposits;diff=float(scores[0]-scores[1]);orig=float(native[0]-native[1]);info=dict(seed=c['seed'],terrain=c['terrain'],seat=s,step=c['step'],status='complete',commands=commands,cargo_scores=scores.tolist(),receipt_delta=diff,native=native,native_delta=orig,target=label,margins=margins,margin_delta=margins[0]-margins[1],wins=wdl,deposit12=deposits,own_deposit_delta=a[0]-b[0],net_deposit_delta=(a[0]-a[1])-(b[0]-b[1]))
        if beta is not None:info['correction']=diff*beta;info['adjusted_delta']=orig+diff*beta
        output.append(info)
    return output

def main():
    plan=read(HERE/'protocol.json');assert sha((HERE/'protocol.json').read_bytes())=='8e869a588e02938b037bde2aff378c973803d5ac64bd1a0adba4a3d32ba4638f'
    cards=plan['ordinary'];train=samples([r for r in cards if r['split']=='train']);heads=dict(enhanced=fitnet(train,range(21)),state_only=fitnet(train,range(8)));train_pairs=response_pairs([r for r in plan['pairs'] if r['split']=='train'],heads);good=[r for r in train_pairs if r['status']=='complete'];D=np.asarray([r['receipt_delta']/3000 for r in good]);res=np.array([r['target']-r['native_delta'] for r in good]);beta=float(np.clip((D@res/len(D))/(D@D/len(D)+.0001),0,3))/3000
    weights={**heads,'pair_scale':beta,'input_width':21,'public_map_fit':sorted({r[0] for r in train}),'steps':[279,285],'depth':8,'receipt_window':12};digest=save('weights.json',weights);save('fit.json',dict(protocol=sha((HERE/'protocol.json').read_bytes()),weights=digest,ordinary=ordinary_report(train,heads),pair_train=train_pairs,pair_scale=beta));print('FROZEN',digest,beta,flush=True)
    val=samples([r for r in cards if r['split']=='val']);obs=ordinary_report(val,heads);pair=response_pairs([r for r in plan['pairs'] if r['split']=='held'],heads,beta);good=[r for r in pair if r['status']=='complete'];active=[r for r in good if abs(r['correction'])>1e-6];base=np.array([r['native_delta'] for r in good]);adj=np.array([r['adjusted_delta'] for r in good]);actual=np.array([r['target'] for r in good]);marg=np.array([r['margin_delta'] for r in good]);flip=(base>=0)!=(adj>=0);gain=np.where(flip,np.where(adj>=0,marg,-marg),0);fixed=int(sum(flip&(gain>0)));broken=int(sum(flip&(gain<0)));m0=float(np.mean((base-actual)**2));m1=float(np.mean((adj-actual)**2));check=dict(ordinary_brier_better_than_both=obs['enhanced']['brier']<min(obs['state_only']['brier'],obs['distance']['brier']),four_hold_roots=len(good)==4,active_three=len(active)>=3,correct_two=fixed>=2,no_new_wrong=broken==0,actual_margin_gain=float(gain.mean())>0,paired_mse_improves=m1<m0,scale_positive=beta>0)
    save('held.json',dict(weight_sha=digest,ordinary=obs,pairs=pair,correct_flips=fixed,wrong_flips=broken,mean_actual_margin_change=float(gain.mean()),pair_mse=[m0,m1],checks=check,qualified_for_development=all(check.values())));print('HELD',[(k,v['brier']) for k,v in obs.items()],check,flush=True)

if __name__=='__main__':main()

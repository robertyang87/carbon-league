"""Compare estimated-successor scoring with the actual original opponent's own private observation. Training maps only."""
from collections import defaultdict
import gzip
import json
import zipfile
import numpy as np
from scipy.special import logsumexp
from study import HERE,DERIV,ROOT,initialize,read,mapping,put,sha,ROLES


def main():
    m=initialize();w=json.loads((HERE/'weights.json').read_bytes())['betas']['roles']
    s=json.loads((ROOT/'code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json').read_bytes())
    cfg={k:v['default'] for k,v in s['configuration'].items() if isinstance(v,dict) and 'default' in v}
    panel=defaultdict(list);parity=[];ident=[];roster=0;money_diff=[];load_diff=[];quant=[]
    for card in json.loads((HERE/'protocol.json').read_bytes())['tickets']:
        if card['split']!='train':continue
        tape,_=read(card);seat=card['seat'];op=1-seat
        with zipfile.ZipFile(DERIV/card['rel']) as z:
            obs=[json.loads(line) for line in gzip.decompress(z.read(card['arm']+'/observations.jsonl.gz')).splitlines()]
        trackers=[m.BeliefTracker(m._Bel,side) for side in (0,1)]
        previous=None;projected=None
        for t,((pub,actions),fr) in enumerate(zip(tape,obs,strict=True)):
            private=fr['observations'][op]
            assert fr['public_observation_sha256'][op]==sha((json.dumps(private,separators=(',',':'),allow_nan=False)+'\n').encode())
            est,_=trackers[seat].step(pub['players'],t,pub['carbon'])
            truth,known=trackers[op].step(private['players'],t,private['carbon'])
            if projected is not None:
                ps=projected['players']; ids=mapping(ps,private['players']); assert ids is not None
                roster+=1
                x=m._quant(m.encode(ps,t,op,projected['carbon'],[1.]*m.N,previous)).astype(np.float32)
                expected=m.encode(private['players'],t,op,truth,known,previous)
                xt=np.clip(np.rint(expected*100.),0,255)*.01
                true_tensor=m._quant(expected)
                quant.append(float(np.max(np.abs(xt-true_tensor))))
                zw,zb,_=m._RNET(x[None],m._SID)
                tw,tb,_=m._NET(true_tensor[None].astype(np.float32),m._SID)
                if len(parity)<8:
                    nw,nb=m._POLICY.forward(xt,m._SID)
                    parity.append([float(np.max(np.abs(tw[0]-nw))),float(np.max(np.abs(tb[0]-nb)))])
                for z in (0,1):money_diff.append(abs(ps[z][0]-private['players'][z][0]))
                for role in ROLES:
                    for uid,item in ps[op][1 if role=='BASE' else 2].items():
                        if role=='BASE':
                            if ps[op][0]<30 or len(ps[op][2])>=10:continue
                            pos=item;raw=zb[0,pos];nat=tb[0,pos];idx=m.BIDX
                        else:
                            pos,cargo,kind=item
                            if kind!=role:continue
                            raw=zw[0,pos];nat=tw[0,pos];idx=m.WIDX
                            real=private['players'][op][2][ids[op][uid]]
                            assert (pos,kind)==(real[0],real[2])
                            load_diff.append(abs(cargo-real[1]))
                        physical=ids[op][uid]
                        act=idx[actions[op].get(physical,'NONE')]; beta=w[role]
                        for name,z in (('projected',raw),('private_oracle',nat)):
                            for unit,coeff in (('frozen',1.),('roles',beta)):
                                l=coeff*np.asarray(z,np.float64)
                                panel[(role,name,unit)].append(float(logsumexp(l)-l[act]))
                        # Negative control uses the same persistent unit; exclude births and deaths on higher/lower ticks.
                        if all(physical in tape[n][0]['players'][op][1 if role=='BASE' else 2] for n in (t-1,t+1)):
                            conf=[]
                            for n in (t-1,t,t+1):
                                k=idx[tape[n][1][op].get(physical,'NONE')]
                                conf.append(float(logsumexp(raw)-raw[k]))
                            ident.append((role,conf))
                projected=None
            if t>=2 and (t-2)%5==0 and t<=292:
                root=dict(players=pub['players'],trees=pub['trees'],step=t,player=seat,carbon=list(map(float,est)),remainingOverageTime=pub['remainingOverageTime'])
                projected=m._turn(root,cfg,actions)
            for z in (0,1):
                trackers[z].set_actions(pub['players'],actions[z])
            previous=pub['players']
    out=dict(training_roots=roster,all_rosters_equal=True,real_private_used_only_for_diagnostic=True,
       max_step_cash_difference=max(money_diff),max_worker_cargo_difference=max(load_diff),
       average_step_cash_difference=np.mean(money_diff),average_worker_cargo_difference=np.mean(load_diff),
       max_native_input_quantization_difference=max(quant),native_numpy_vs_torch_max=np.max(parity,axis=0).tolist(),
       per_role_NLL={'/'.join(k):dict(n=len(v),mean=np.mean(v)) for k,v in panel.items()},
       shifted_previous_same_next_NLL={r:dict(n=sum(k==r for k,_ in ident),nll=np.mean([v for k,v in ident if k==r],axis=0).tolist()) for r in ROLES})
    print(put(HERE/'alignment.json',out),json.dumps(out,separators=(',',':'))[:440])
if __name__=='__main__':main()

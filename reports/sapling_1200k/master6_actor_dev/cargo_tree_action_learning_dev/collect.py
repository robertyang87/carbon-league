"""Causal opportunities, cross-map split, credit labels, actual prefix features from the released it50 mixture."""
import collections,gzip,hashlib,importlib.util,inspect,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ACTOR=HERE.parent;FULL=ACTOR/'deployed_mix_policy_gradient_dev/full_episode'
ECO=FULL/'direct_original/economy';sys.path.insert(0,str(ECO));from analyze import read,sha,process,index
from opportunities import D,TO,DIRECTIONS,predict_fee
MIX=ACTOR/'deployed_mix_policy_gradient_dev';sys.path.insert(0,str(MIX));from study import arena,policy
PARENT=FULL/'selected.py';BASE=FULL/'selected.npz';CFG=arena.read(HERE/'contract.json')
def save(name,obj):
    p=HERE/name;b=json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode();p.write_bytes(gzip.compress(b,mtime=0) if p.suffix=='.gz' else b)
def hash_id(k):return hashlib.sha256(k.encode()).hexdigest()
def instrument(path):
    m=policy.load(path)
    source=inspect.getsource(m._return_decode);marker='action = int(np.argmax(np.where(legal, combined, -np.inf)))'
    if source.count(marker)!=1:raise ValueError("decoder changed")
    captures={}
    def keep(branch,slot,hidden,z,legal,move):
        captures.setdefault(branch,{})[slot]=(hidden.copy(),z.copy(),legal.copy())
    code=source.replace(marker,'retain(apply_residual,slot,states[1][3],combined,legal,move)\n        '+marker)
    ns=dict(vars(m),retain=keep);exec(code,ns)
    old=m._return_decode
    records={}
    def wrapped(x,aux,states,branch):
        y=ns['_return_decode'](x,aux,states,branch);records[branch]=y.copy();return y
    m._return_decode=wrapped
    return m,captures,records

def candidates(card):
    _,_,ev,states,orders=process(card,True);n=len(orders);found=[];bank=collections.defaultdict(list);plants=collections.defaultdict(list)
    ground=collections.defaultdict(float)
    for e in ev:
        h=e['t'];s=e['recipient'];uid=e['ident'][0] if e['ident'] else None
        if e['kind']=='own_delivery':bank[uid].append((h,e['value']))
        if e['kind']=='tree_absorption':ground[h,s,uid]+=e['value']-e['ident'][2]
        if e['kind']=='plant':
            cell=e['ident'][1];o=states[h][s];opp=o['players'][1-s]
            if min((int(D[cell,w[0]]) for w in opp[2].values()),default=99)<=1:continue
            if sum(o['carbon'][int(k)] for k in TO[cell,1:])<120:continue
            if h+1>=n:continue
            ts=[k for k,(pos,age) in states[h+1][s]['players'][s][3].items() if pos==cell and age==1]
            if len(ts)!=1:raise ValueError('unbound target tree')
            plants[uid].append((h,s,ts[0],-e['value']))
    for s in (0,1):
        bag=collections.defaultdict(list)
        for t,both in enumerate(states):
            if not (90<=t<190 or 195<=t<=min(279,n-20)):continue
            o=both[s];me,opp=o['players'][s],o['players'][1-s];trees={p for tm in o['players'] for p,a in tm[3].values()};bases={p for tm in o['players'] for p in tm[1].values()}
            enemies=[w[0] for w in opp[2].values()]
            for uid,(pos,cargo,role) in me[2].items():
                cmd=orders[t][s].get(uid) or 'STAY';act=DIRECTIONS.index(cmd)
                near=min((int(D[pos,k]) for k in enemies),default=99)
                if role=='COLLECTOR' and t>=195:
                    home=min(int(D[pos,p]) for p in me[1].values())
                    if cargo<40 or not 1<=home<=8:continue
                    threat=min((int(D[pos,w[0]]) for w in opp[2].values() if w[2]=='COLLECTOR' and w[1]<cargo),default=99)
                    tagged=[v for h,v in bank[uid] if t<=h<t+20];amt=max(tagged,default=0.);reward=min(450.,amt)/200
                    group=[int(cargo>=120),int(home>=4),int(threat<=2)];dest=TO[pos,act];h2=min(int(D[dest,p]) for p in me[1].values())
                    cls='hold' if act==0 else 'toward' if h2<home else 'other';task='bank'
                    info=dict(cargo=cargo,home=home,nearest_lighter_enemy=threat,credited=amt,positive=amt>=200,failed=amt==0.)
                elif role=='PLANTER' and t<190:
                    if pos in trees or pos in bases:continue
                    visible=float(sum(o['carbon'][int(k)] for k in TO[pos,1:]))
                    if not (visible>=65 or not any(int(D[pos,k])<=1 for k in enemies)):continue
                    if not any(cell not in trees and cell not in bases for cell in TO[pos]):continue
                    group=[int(visible>=120),int(near<=2),int(len(me[3])>=4)]
                    acquired=next(((h,side,tree,fee) for h,side,tree,fee in plants[uid] if t<=h<t+4),None)
                    earned=0.;fee=0.
                    if acquired:
                        h,side,tree,fee=acquired
                        for u in range(h+1,min(n,h+26)):
                            if tree not in states[u][s]['players'][s][3]:break
                            # Posts to a rival in the seizure transition contribute zero, then tenure is ended.
                            earned+=ground[u,s,tree]
                    net=earned-fee;reward=np.clip(net,-150,150)/60.;task='tree';cls='hold' if act==0 else 'move'
                    info=dict(visible_soil=visible,nearest_enemy=near,owned_trees=len(me[3]),cash=me[0],paid=fee,ground25=earned,positive=bool(acquired and earned>fee),failed=not acquired or earned<=fee)
                else:continue
                item=dict(raw=card['sha256'],seed=card['seed'],terrain=card['terrain'],seat=s,t=t,uid=uid,role=role,task=task,group=group,actual_cmd=cmd,action_class=cls,reward=float(reward),is_master=s!=card['seat'],**info)
                rank=hash_id(f"root-v1/{card['sha256']}/{s}/{t}/{uid}");bag[task].append((rank,item))
        for ls in bag.values():found.extend(v for _,v in sorted(ls)[:16])
    return found,states,orders

def main():
    if sha(PARENT)!=CFG['parent']:raise ValueError("wrong package")
    cards=[r for r in index() if r['source']=='historic_original']
    terrains=sorted(set(c['terrain'] for c in cards),key=lambda k:hash_id('cash-tree-split-v1/'+k))
    if len(terrains)!=29 or len(cards)!=58:raise ValueError('changed cohort')
    split={k:('train' if i<19 else 'val' if i<23 else 'test') for i,k in enumerate(terrains)}
    save('panel.json',dict(split=split,cards=cards,parent=sha(PARENT),weights=sha(BASE),data_contract=sha(HERE/'contract.json')))
    rec=[];H=[];Z=[];M=[];bad=collections.Counter();m,capt,choices=instrument(PARENT)
    for ci,c in enumerate(cards):
        roots,states,orders=candidates(c);by=collections.defaultdict(list)
        for v in roots:by[v['seat'],v['t']].append(v)
        for s in (0,1):
            feat=sys.modules['cnet2_feat'].Feat(s)
            for t,both in enumerate(states):
                x=feat.observe(both[s])
                if (s,t) not in by:continue
                capt.clear();choices.clear();y=m._mixed_actions(x,feat.aux);initial,adjusted=choices[False],choices[True]
                final,rejected=policy.guard(m,x,initial,adjusted)
                if not np.array_equal(final,y):raise ValueError('guard diverged')
                track=capt[not rejected];label=feat.labels(orders[t][s])
                for v in by[s,t]:
                    try:slot=feat.own_uid.index(v['uid'])
                    except ValueError:bad['slot_missing']+=1;continue
                    if slot<0 or slot>=10:raise ValueError("wrong worker")
                    target=int(label[slot]);hidden,logits,legal=track[slot]
                    if not legal[target] or legal.sum()<2:bad['not_actionable_on_released_prefix']+=1;continue
                    if int(np.argmax(np.where(legal,logits,-np.inf)))!=y[slot]:raise ValueError('exact emitted logit mismatch')
                    rec.append(dict(**v,split=split[c['terrain']],slot=int(slot),target=target,parent=int(y[slot]),guard_applied=not rejected))
                    H.append(hidden);Z.append(logits);M.append(legal)
        if (ci+1)%8==0:print('maps/tapes',ci+1,'of',len(cards),'rows',len(rec),flush=True)
    save('rows.json.gz',rec);np.savez_compressed(HERE/'public_prefix_scores.npz',hidden=np.asarray(H,np.float32),logits=np.asarray(Z,np.float32),legal=np.asarray(M,bool))
    save('data.json',dict(roots=len(rec),filtered=dict(bad),panel=sha(HERE/'panel.json'),row_sha=sha(HERE/'rows.json.gz'),arrays_sha=sha(HERE/'public_prefix_scores.npz')))
if __name__=='__main__':main()

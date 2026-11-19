"""Build same-prefix, jointly re-decoded single-step alternatives using actual simultaneous enemy orders as labels."""
import collections,gzip,hashlib,inspect,json,sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;EXP=HERE.parent
sys.path.insert(0,str(EXP))
from collect import policy, D, TO, arena
from analyze import FastCarbon,CashEvents,sha,read
CFG=read(HERE/'contract.json')
def write(path,obj):
    b=json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    tmp=path.with_suffix('.tmp');tmp.write_bytes(gzip.compress(b,mtime=0) if path.suffix=='.gz' else b);tmp.replace(path)
def root_seed(x):return hashlib.sha256(x.encode()).hexdigest()

def category(obs,uid):
    s=obs['player'];t=obs['step'];own,en=obs['players'][s],obs['players'][1-s]
    p,c,role=own[2][uid];opps=list(en[2].values());home=min(int(D[p,q]) for q in own[1].values())
    if role=='COLLECTOR':
        lighter=min((int(D[p,w[0]]) for w in opps if w[2]=='COLLECTOR' and w[1]<=c), default=99)
        if c>=40 and lighter<=2:return 'threat'
        if c>=80 and home<=2:return 'bank'
        if obs['carbon'][p]>=30 and lighter>2 and p not in {q for z in (own,en) for q in z[1].values()} and p not in {q for z in (own,en) for q,_ in z[3].values()}:return 'harvest'
    elif 90<=t<190 and p not in {q for z in (own,en) for q in z[1].values()} and p not in {q for z in (own,en) for q,_ in z[3].values()} and sum(obs['carbon'][int(q)] for q in TO[p,1:])>=120 and min((int(D[p,w[0]]) for w in opps),default=99)>=2:return 'tree'
    return None

class Recorded:
    def __init__(self,pack):
        self.net=policy.load(pack);self.want=(-1,-1);self.data={}
        src=inspect.getsource(self.net._return_decode);key='action = int(np.argmax(np.where(legal, combined, -np.inf)))'
        def record(x,aux,e,route,slot,h,move,leg,scores,combined,action):
            self.x=x;self.aux=aux;self.enc=e
            self.data.setdefault(route,{})[slot]=h.copy(),move.copy(),leg.copy(),np.stack(scores),combined.copy()
            return self.want[1] if self.want[0]==slot and leg[self.want[1]] else action
        assert src.count(key)==1
        ns=dict(vars(self.net),retain=record);exec(src.replace(key,key+'\n        action=retain(x,aux,encoded,apply_residual,slot,states[1][3],move,legal,scores,combined,action)'),ns)
        self.net._return_decode=ns['_return_decode']
    def base(self,obs,cfg):
        self.want=(-1,-1);self.data.clear();labels=self.net._FEAT
        physical=self.net.agent(obs,cfg);feat=self.net._FEAT;sent=feat.labels(physical)
        assert set(self.data)=={False,True}
        branch=self.decode(-1,-1)
        assert np.array_equal(branch,sent)
        return sent,self.data[self.which].copy()
    def decode(self,slot,action):
        self.want=(slot,action);net=self.net
        baseline=net._return_decode(self.x,self.aux,self.enc,False)
        candidate=net._return_decode(self.x,self.aux,self.enc,True)
        protected=(np.arange(len(baseline))>=10)|(self.x['tok_kind'][:len(baseline)]==1)
        self.which=not(np.any(baseline[protected]!=candidate[protected]) or
                           (np.any(baseline!=candidate) and net._return_conflicts(self.x,candidate)-net._return_conflicts(self.x,baseline)))
        return candidate if self.which else baseline

def outcome(sim,state,seat,uid,joint):
    sim.set_state(state);sim.cash=CashEvents(sim.cash);sim.step(*joint)
    paid=sum(e[2] for e in sim.cash.events if e[0]=='own_delivery' and e[1]==seat and e[3]==uid)
    alive=uid in sim.workers;held=sim.workers[uid].carbon if alive else 0.
    bought=any(e[0]=='plant' and e[1]==seat and e[3]==uid for e in sim.cash.events)
    return [alive,held,paid,bought]

def choose(raw,card):
    s=card['seat'];sim=FastCarbon(raw['game_configuration']).reset(seed=card['seed']);pool=collections.defaultdict(list)
    for t,joint in enumerate(raw['actions']):
        o=sim.obs(s)
        for uid in o['players'][s][2]:
            tag=category(o,uid)
            if tag:pool[tag].append((root_seed(f"safety-pre-v1/{card['sha256']}/{t}/{uid}"),t,uid))
        sim.step(*joint)
    return sorted((t,uid,tag) for tag,b in pool.items() for _,t,uid in sorted(b)[:12])

def build():
    panel=read(EXP/'panel.json');source=EXP/'candidate.py';assert sha(source)==CFG['parent'] and sha(EXP/'candidate.npz')==CFG['weights']
    assert len(panel['cards'])==58 and len(panel['split'])==29 and len(set(c['seed'] for c in panel['cards']))==29
    assert not {c['seed'] for c in panel['cards']} & {621758343,621758344,621758345,621758346,621758347,621758348,621758349}
    model=Recorded(source);rows=[];arrays=collections.defaultdict(list);counts=collections.Counter()
    for card in panel['cards']:
        path=arena.ROOT/card['path'];raw=read(path);s=card['seat'];cfg=raw['game_configuration'];assert sha(path)==card['sha256']
        receipt=read(arena.ROOT/card['receipt_path'])
        if 'official' in receipt:assert receipt['official']['transcript_sha256']==card['sha256'] and receipt['official']['statuses']==['DONE','DONE']
        roots=choose(raw,card);by=collections.defaultdict(list)
        for t,u,tag in roots:by[t].append((u,tag))
        sim=FastCarbon(cfg).reset(seed=card['seed']);fork=FastCarbon(cfg)
        for t,orders in enumerate(raw['actions']):
            obs=sim.obs(s)
            if t in by or t==0:
                sent,base=model.base(obs,cfg);feat=model.net._FEAT;st=sim.get_state()
                for uid,tag in by[t]:
                    slot=feat.own_uid.index(uid);h,move,leg,logits,z=base[slot]
                    p,cargo,role=obs['players'][s][2][uid];mask=np.zeros(5,bool);ys=np.zeros((5,4),np.float32);suffix_changes=[]
                    for a in np.flatnonzero(leg):
                        dec=model.decode(slot,int(a))
                        if not np.array_equal(dec[:slot],sent[:slot]) or dec[slot]!=a:counts['guard_rejected_branch']+=1;continue
                        physical=feat.to_actions(dec);team=[dict(o) for o in orders];team[s]=physical
                        ys[a]=outcome(fork,st,s,uid,team);mask[a]=True;suffix_changes.append(int(np.sum(dec[slot+1:]!=sent[slot+1:])))
                        if a==sent[slot]:
                            assert np.array_equal(dec,sent)
                            if physical==orders[s]:counts['identical_factual_joint']+=1
                    assert mask[sent[slot]]
                    if mask.sum()<2:counts['one_action']+=1;continue
                    ev=readout_features(h,move,logits,z,model.net)
                    for k,v in {'h':h,'move':move,'proj':ev,'logits':logits,'mixed':z,'mask':mask,'labels':ys}.items():arrays[k].append(v)
                    counts['alternatives_with_downstream_changes']+=sum(i>0 for i in suffix_changes)
                    rows.append(dict(t=t,uid=uid,slot=slot,tag=tag,own_seat=s,seed=card['seed'],terrain=card['terrain'],split=panel['split'][card['terrain']],cargo=cargo,parent=int(sent[slot]),initial_legal=int(leg.sum()),raw=card['sha256']))
            else:model.net._FEAT.observe(obs)
            sim.step(*orders)
        assert np.allclose(sim.cash,raw['official_score_by_seat'],atol=.011,rtol=0)
        print(card['seed'],s,len(rows),flush=True)
    np.savez_compressed(HERE/'pairs.npz',**{k:np.stack(v) for k,v in arrays.items()});write(HERE/'roots.json.gz',rows)
    write(HERE/'dataset.json',dict(parent=sha(source),panel=sha(EXP/'panel.json'),contract=sha(HERE/'contract.json'),arrays=sha(HERE/'pairs.npz'),roots=sha(HERE/'roots.json.gz'),counts=counts,rows=len(rows)))

def readout_features(h,move,logits,z,net):
    return net._NETS[1].lin(move,'V')

if __name__=='__main__':build()

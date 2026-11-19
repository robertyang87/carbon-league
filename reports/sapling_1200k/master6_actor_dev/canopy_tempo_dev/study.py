"""Fit on the frozen elite split; independently measure opportunity conversion on real native sources."""
from collections import Counter,defaultdict
from hashlib import sha256
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import policy
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
ROOT=BASE.parents[1]
ELITE=BASE/'top5_h2h_data.json'
CACHE=Path('/workspace/sapling-all-top5-20261119/replays')
NATIVE=HERE.parent/'tree_response_training'
sys.path.insert(0,str(ROOT/'code'))
from fastsim import FastCarbon


def read(path):
    data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=1,allow_nan=False)+'\n')


def native_game(spec):
    path=ROOT/spec['raw_path'];raw=read(path);cfg=raw['game_configuration']
    if sha256(path.read_bytes()).hexdigest()!=spec['raw_sha256'] or raw['pack_sha256']!=read(HERE/'protocol.json')['parent_sha256'] or raw['examiner_md5']!='57f9a594cce4f2ec9e3573d8d9d26ca1' or raw['errors']:raise ValueError('native source')
    sim=FastCarbon(cfg).reset(seed=spec['seed'])
    if sim.carbon!=raw['initial_full_carbon'] or sha256(bytes(int(x) for x in sim.carbon)).hexdigest()!=spec['map_sha256']:raise ValueError('native map')
    for joint in raw['actions']:
        v=[sim.obs(s) for s in (0,1)];sim.step(*joint);after=sim.full_observation()
        yield v,joint,after,cfg
    if list(sim.cash)!=spec['scores']:raise ValueError('wrong official cash')


def public_turns(card):
    raw=(CACHE/(card['id']+'.json.gz')).read_bytes()
    # The all_top5 manifest previously validated the raw blob and its replay with two engines.
    if sha256(raw).hexdigest()!=card['sha256']:raise ValueError('changed elite replay')
    tape=json.loads(gzip.decompress(raw));cfg=tape['configuration']
    for pair,nxt in zip(tape['steps'],tape['steps'][1:]):
        base=pair[0]['observation'];views=[dict(player=s,step=base['step'],players=base['players'],carbon=pair[s]['observation']['carbon']) for s in (0,1)]
        yield views,[a['action'] or {} for a in nxt],nxt[0]['observation'],cfg


def collect(turns,map_id,who=None):
    stats=[defaultdict(Counter),defaultdict(Counter)];rows=[];active={}
    for obs,joint,after,cfg in turns:
        t=obs[0]['step'];phase=policy.stage(t);pre=obs[0]['players'];now=after['players']
        oldids=set(pre[0][3])|set(pre[1][3]);posttree={q:(s,tid) for s,pl in enumerate(now) for tid,(q,age) in pl[3].items()}
        for s,pl in enumerate(pre):
            c=stats[s][phase]; births=set(now[s][3])-oldids;c['all_fresh_paid']+=len(births);c['all_seize']+=len(set(now[s][3])-set(pl[3])-births)
            c['all_planter_turns']+=sum(w[2]=='PLANTER' for w in pl[2].values())
            st=policy.context(obs[s],cfg,joint[s])
            for uid,(p,_,role) in pl[2].items():
                if role!='PLANTER' or p in st['trees'] or p in st['bases']:continue
                c['fresh_cell']+=1
                if pl[0]>=st['price']:c['fresh_affordable']+=1
                x=policy.proposal(st,uid)
                if x is None:continue
                held=(joint[s].get(uid) or 'NONE')=='NONE';tree=posttree.get(p)
                paid=bool(tree and tree[0]==s and tree[1] in births)
                if held!=paid:raise ValueError(f'safe opportunity payment discrepancy {t}:{s}:{uid}:{held}:{paid}')
                c['clean']+=1;c['clean_paid']+=paid;c['clean_move']+=not paid
                lo,hi=policy.bounds(st,uid)
                row=dict(map=map_id,seat=s,uid=uid,t=t,x=x,hold=held,price=st['price'],gross=None,lo=lo,hi=hi,
                     order=joint[s].get(uid) or 'NONE',pos=p,whos=who,late_exit=policy.order(st,uid,{'features':policy.NAMES,'beta':[0]*len(x)},{'middle':False,'late':False}))
                rows.append(row)
                if paid:
                    row['gross']=0.
                    if (s,tree[1]) in active:raise ValueError('duplicate fresh tree')
                    active[(s,tree[1])]=row
        for (s,tid),r in list(active.items()):
            if tid not in now[s][3]:del active[s,tid]
            else:r['gross']+=after['trees'][tid][1]
    for r in rows:
        if r['hold'] and (r['gross']+.011<r['lo'] or (r['hi'] is not None and r['gross']>r['hi']+.011)):
            raise ValueError(('violated observed payoff bound',r))
    return stats,rows


def elite_list():
    a=read(ELITE);official={}
    for f in (BASE/'all_top5').glob('**/manifest.json'):
        try:entries=read(f).get('games',[])
        except (ValueError,OSError):continue
        for r in entries:
            if r.get('frame_verified'):official[r['id']]=r
    for c in a['games']:
        if any('master3' in n.lower() for n in c['models']):continue
        r=official[c['id']]
        if r['seed']!=c['seed'] or r['canonical_sha256']!=c['canonical_sha256']:raise ValueError('elite split/roster mismatch')
        yield c['id'],c['split'],r


def mapmean(rows,fn):
    group=defaultdict(list)
    for r in rows:group[r['map']].append(float(fn(r)))
    return float(np.mean([np.mean(v) for v in group.values()])) if group else None


def fit(train):
    x=np.array([r['x'] for r in train]); y=np.array([r['hold'] for r in train],dtype=float)
    n=Counter(r['map'] for r in train);w=np.array([1/(len(n)*n[r['map']]) for r in train]);b=np.zeros(x.shape[1])
    for k in range(300):
        p=1/(1+np.exp(-(x@b)))
        reg=.015*b;reg[0]=0.;b-=.5*(x.T@(w*(p-y))+reg)
    return dict(features=list(policy.NAMES),beta=list(b),training_maps=len(n),training_clean_opportunities=len(train))


def main():
    area=read(HERE/'protocol.json'); native=[]; native_per=[]
    native_reference=read(HERE.parent/'early_lane_dev/protocol.json')
    for n in range(1,6):
        path=NATIVE/('batches/intake_%02d/source_receipt.json'%n if n>1 else 'source_receipt.json');d=read(path)
        if sha256(path.read_bytes()).hexdigest()!=native_reference['batches'][str(n)]['source_sha256']:raise ValueError('changed native registration')
        for s in d['games']:
            st,rr=collect(native_game(s),s['map_sha256'],s['seat']);native.extend(rr)
            native_per.append(dict(map=s['map_sha256'],seat=s['seat'],source=s['raw_sha256'],by_role={'v1':st[s['seat']],'examiner':st[1-s['seat']]}))
    total={}; native_quorum={}
    for phase in ('opening','middle','late'):
        split=lambda i,j:[g['by_role'][i][phase][j] for g in native_per]
        total[phase]={key:{k:sum(split(key,k)) for k in native_per[0]['by_role'][key][phase]} for key in ('v1','examiner')}
        cs=defaultdict(lambda:defaultdict(Counter))
        for g in native_per:
            for role,b in g['by_role'].items():cs[g['map']][role].update(b[phase])
        rates={key:v['clean_paid']/v['clean'] for key,v in total[phase].items()}
        diffs=[v['v1']['clean_paid']/max(1,v['v1']['clean'])-v['examiner']['clean_paid']/max(1,v['examiner']['clean']) for v in cs.values()]
        native_quorum[phase]=dict(rates=rates,n_maps_lower=sum(v<0 for v in diffs),n_maps_higher=sum(v>0 for v in diffs))
    save(HERE/'native_audit.json',dict(phases=total,quorum=native_quorum,by_game=native_per))
    print('native',native_quorum,flush=True)
    elite={'train':[],'val':[]}; rosters={}
    for key,split,g in elite_list():
        st,rr=collect(public_turns(g),g['id']);elite[split].extend(rr);rosters[key]=dict(split=split,sha256=g['sha256'],seat_names=g['seat_names'],n=len(rr))
    train=elite['train'];val=elite['val'];w=fit(train)
    prior={k:mapmean([r for r in train if policy.stage(r['t'])==k],lambda r:r['hold']) for k in ('opening','middle','late')}
    constant=mapmean(train,lambda r:r['hold']);score=lambda r:policy.probability(w,r['x'])
    val_brier={k:mapmean(val,lambda r:(pred(r)-r['hold'])**2) for k,pred in
        (('model',score),('phase',lambda r:prior[policy.stage(r['t'])]),('pooled',lambda r:constant))}
    def metrics(rs):
        buys=[r for r in rs if r['hold']];nm=set(r['map'] for r in buys)
        return dict(eligible=len(rs),actual_buys=len(buys),maps_with_buys=len(nm),map_equal_first_tenure_net=mapmean(buys,lambda r:r['gross']-r['price']),
                 positive_map_means=sum(np.mean([r['gross']-r['price'] for r in buys if r['map']==m])>0 for m in nm))
    held=metrics([r for r in val if policy.stage(r['t'])=='middle' and score(r)>=.5]);held['positive_map_means']=int(held['positive_map_means'])
    brier=val_brier['model']<min(val_brier['phase'],val_brier['pooled'])
    quorum=native_quorum
    mid=brier and quorum['middle']['rates']['v1']<quorum['middle']['rates']['examiner'] and quorum['middle']['n_maps_lower']>5 and held['actual_buys']>=12 and held['maps_with_buys']>=5 and held['map_equal_first_tenure_net']>0 and held['positive_map_means']>held['maps_with_buys']/2
    late=brier and quorum['late']['rates']['v1']>quorum['late']['rates']['examiner'] and quorum['late']['n_maps_higher']>5
    gates=dict(middle=bool(mid),late=bool(late));fire=[]
    for r in native:
        if r['whos']!=r['seat']:continue
        phase=policy.stage(r['t']);pv=score(r)
        if phase=='middle' and not r['hold'] and pv>=.5 and r['lo']>r['price']+.02:
            fire.append({k:r[k] for k in ('map','seat','t','uid','pos','order','price','lo','hi')}|dict(phase=phase,prob=pv,gate=gates[phase]))
        if phase=='late' and r['hold'] and pv<.5 and r['hi'] is not None and r['hi']+.02<r['price']:
            fire.append({k:r[k] for k in ('map','seat','t','uid','pos','order','price','lo','hi','gross')}|dict(phase=phase,prob=pv,gate=gates[phase]))
    report=dict(protocol_sha256=sha256((HERE/'protocol.json').read_bytes()).hexdigest(),features_sha256=sha256((HERE/'policy.py').read_bytes()).hexdigest(),
        native=native_quorum,elite_split={k:sum(g['split']==k for g in rosters.values()) for k in elite},
        training_phase_rates=prior,training_mean=constant,heldout_brier=val_brier,heldout_mid_predicted_pay=held,
        by_held_map={m:dict(n=sum(r['map']==m for r in val),brier=mapmean([r for r in val if r['map']==m],lambda r:(score(r)-r['hold'])**2)) for m in {r['map'] for r in val}},
        gates=gates,before_prefix_and_escape_native_eligible=fire,elite_rosters=rosters,
        limitations='First-tenure cash of actually paid trees conditions on selection. Conditional pay rates and the middle payoff lower bound do not include other policies, income sources or early extinction.')
    save(HERE/'weights.json',w);save(HERE/'study.json',report)
    print('elite',report['elite_split'],val_brier,held,gates,Counter(r['phase'] for r in fire),flush=True)

if __name__=='__main__':main()

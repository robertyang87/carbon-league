"""Measure a six-tick deposit/steal surrogate under recorded orders and fit a shared action scorer."""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import time

import numpy as np
from scipy.optimize import minimize

import model as net

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
MANIFEST="00e8e4cfa850a05e5d51024b38e3dca2019f5b4042138396c684d8739effbde0"
CACHE=Path("/workspace/sapling-all-top5-20261119/replays")
D=net.D; C=net.CELLS
head_dir=HERE.parent/"ladder/cargo_defense"
def module(name,file):
    s=importlib.util.spec_from_file_location(name,file);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
old=module("prior_freight_risk",head_dir/"runtime.py")
rules=module("frozen_arrival_labels",head_dir/"labels.py")
OLD=json.loads((head_dir/"weights.json").read_bytes())
OLDW=np.array(OLD["weights"],np.float32)


def sha(raw):return hashlib.sha256(raw).hexdigest()
def save(path,d):
    raw=(json.dumps(d,sort_keys=True,separators=(",",":"),allow_nan=False,ensure_ascii=False)+"\n").encode();path.write_bytes(raw);return sha(raw)
def read(ticket):
    fname=CACHE/(ticket["id"]+".json.gz")
    if fname.is_file():raw=fname.read_bytes()
    else:
        a=ticket["source"]
        with tarfile.open(ROOT/a["archive"]) as z:raw=z.extractfile(a["member"]).read()
    if sha(raw)!=ticket["sha"]:raise ValueError("Replay mismatch")
    g=json.loads(gzip.decompress(raw))
    if sha(bytes(g["steps"][0][0]["observation"]["full_carbon"]))!=ticket["terrain"]:raise ValueError("Wrong terrain")
    return g


def choice(vals,dh):
    return min(range(5),key=lambda i:(-vals[i],dh[i],i))


def route(pos,cargo,ps,seat,uid):
    """Choose the next shortest route by geometric exposure before seeing this round's orders."""
    homes=set(ps[seat][1].values());light=[u[0] for u in ps[1-seat][2].values() if u[2]=="COLLECTOR" and u[1]<=cargo]
    ally=[w[0] for k,w in ps[seat][2].items() if k!=uid]
    def key(a):
        z=C[pos][a]
        return min(D[z][h] for h in homes),int(z not in homes)*sum(D[z][e]<=1 for e in light),sum(w==z for w in ally),a
    return min(range(1,5),key=key)


def enemy_tax_cells(before, after, seat):
    """Official trees synthesize income from surviving old trees under their new owner."""
    old_ids = {tid for player in before for tid in player[3]}
    return {cell for tid, (cell, age) in after[1-seat][3].items() if tid in old_ids}


def tree_tax_fraction(cell, enemies, cfg):
    n = int(cell in enemies) + sum(c in enemies for c in net.TREE_RING[cell])
    return min(1.0, n*cfg["collectorAbsorptionRate"])


def collector_count(occupied, focal, cell, players, cfg):
    """Collectors in play after actual recruitment, simultaneous collisions and base clearing."""
    enemy_bases = set(players[1-focal.seat][1].values())
    live = 0
    for z in set(occupied) | {cell}:
        group = occupied.get(z, [])
        if z == cell:
            group = [focal, *group]
        winner = rules.collision(group)
        if winner is None or winner.seat != focal.seat or winner.role != 'COLLECTOR':
            continue
        if z in enemy_bases and sum(w.cargo for w in group) <= cfg["smeltCost"]:
            continue
        live += 1
    return live


def gather(players, cell, focal, occupied, visible_carbon, cfg):
    """Staying carrier on a bare tile after tree aging, collision and base clearing."""
    if any(cell in side[1].values() for side in players):
        return 0.
    for side in players:
        for tree_pos, age in side[3].values():
            if tree_pos == cell:
                if age < cfg['treeLifespan']:
                    return 0.
                visible_carbon = cfg['co2FrmWithered']
    rate = max(0., cfg['initialCollectRate'] -
               cfg['collectDecreaseRate'] * collector_count(occupied, focal, cell, players, cfg))
    return visible_carbon * rate


def labels(frames,cfg,t,seat,uid):
    ps=frames[t][0]["observation"]["players"];at,mass,_=ps[seat][2][uid];rows=[]
    facts=[];homes=set(ps[seat][1].values());foe=set(ps[1-seat][1].values())
    for j in range(t,t+6):
        pp=frames[j][0]["observation"]["players"];nextp=frames[j+1][0]["observation"]["players"]
        orders=[f["action"] or {} for f in frames[j+1]]
        occupied=rules.arrivals(pp,orders,(seat,uid),cfg)
        # In the real engine the incoming owner collects the tax, newborn trees collect nothing.
        en=enemy_tax_cells(pp, nextp, seat)
        facts.append((pp,occupied,en,orders[seat].get(uid,'NONE')))
    visible_carbon=frames[t][seat]["observation"]["carbon"]
    for k in range(5):
        pos=at;cargo=mass;gain=0.;value=0.;event='open';when=0
        for i,(p,occupied,enemy_trees,_) in enumerate(facts):
            action=k if i==0 else route(pos,cargo,p,seat,uid)
            z=C[pos][action];incoming=cargo*(1-cfg["moveCost"] if action else 1)
            me=rules.Arrival(seat,uid,"COLLECTOR",incoming);others=occupied[z]
            winner=rules.collision([me,*others])
            if z in homes:
                value=(.90**i)*max(incoming-cfg["smeltCost"],0)/mass;event='bank';when=i+1;break
            if z in foe:
                value=-(.90**i)*incoming/mass;event='enemy_bank';when=i+1;break
            if winner!=me:
                if winner and winner.seat!=seat:
                    value=-(.90**i)*incoming/mass;event='stolen'
                else:event='other_collision'
                when=i+1;break
            cargo=incoming
            if i==0 and action==0:
                gain=gather(p,z,me,occupied,float(visible_carbon[z]),cfg)
                cargo+=gain
            # Track the original cargo and its stay harvest; no speculative loot from other carriers.
            taxes=tree_tax_fraction(z,enemy_trees,cfg)
            cargo*=max(0.,1-taxes);pos=z
        rows.append(dict(value=value,event=event,offset=when,gathered=gain))
    return rows


def collect(tickets):
    result=[]
    for ticket in tickets:
        g=read(ticket);f=g['steps'];cfg=g['configuration']
        if g.get('statuses')!=['DONE','DONE']:continue
        if cfg['moveCost'] or cfg['smeltCost']:raise ValueError("Unregistered charges")
        for t in range(11,min(len(f)-6,299),13):
            p=f[t][0]['observation']['players']
            for s in (0,1):
                prior=old.prepare(p,s,t,f[t-1][0]['observation']['players']);homes=set(p[s][1].values())
                for u,(pos,cargo,occ) in p[s][2].items():
                    dist=min(D[pos][h] for h in homes)
                    if occ!='COLLECTOR' or cargo<50 or not 1<=dist<=5:continue
                    current=f[t][s]['observation']['carbon']
                    xx=net.features(p,s,u,t,carbon=current)
                    y=labels(f,cfg,t,s,u);go=[min(D[z][h] for h in homes) for z in C[pos]]
                    risk=old.forecast(prior,u,OLDW,OLD['recruit_probability'])['risk']
                    acted=[(k['action'] or {}) for k in f[t+1]]
                    fixed=net.ACTIONS.index(acted[s].get(u,'NONE'))
                    seen=None
                    for dt in range(1,7):
                        prev=f[t+dt-1][0]['observation']['players'][s][2].get(u)
                        fut=f[t+dt][0]['observation']['players'][s][2].get(u)
                        if not fut:seen=['gone',dt];break
                        if fut[0] in homes and fut[1]==0 and prev[1]>0:seen=['arrived',dt];break
                    result.append(dict(game=ticket['id'],seat=s,t=t,uid=u,load=cargo,distance=dist,X=xx,labels=y,
                           stay_carbon=current[pos],distance_action=choice([-v for v in go],go),
                           risk_action=choice([-v for v in risk],go),home=go,issued=fixed,factual=seen))
    return result


def train(rr):
    counts=Counter(r['game'] for r in rr);q=np.array([1/len(counts)/counts[r['game']] for r in rr])
    X=np.stack([r['X'] for r in rr]);Y=np.array([[c['value'] for c in r['labels']] for r in rr]);Y-=Y.mean(1)[:,None]
    mean=q@X.mean(1);sd=np.maximum(np.sqrt(q@((X-mean)**2).mean(1)),.05);z=(X-mean)/sd
    n=X.shape[2];h=12;rng=np.random.default_rng(53219);w0=rng.normal(0,.14,size=n*h+h+h)
    def parts(w):return w[:n*h].reshape(n,h),w[n*h:n*h+h],w[n*h+h:]
    def loss(w):
        A,b,V=parts(w);H=np.tanh(z@A+b);val=H@V
        err=val-val.mean(1)[:,None]-Y
        weight=q[:,None]*err/5
        deriv=(weight[:,:,None]*V)*(1-H*H)
        grad=np.r_[(z.reshape(-1,n).T@deriv.reshape(-1,h)).ravel(),deriv.sum((0,1)),H.reshape(-1,h).T@weight.ravel()]
        return float(.5*np.sum(weight*err)+.001*w@w),grad+.002*w
    res=minimize(loss,w0,jac=True,method='L-BFGS-B',options=dict(maxiter=160,ftol=1e-9));A,b,V=parts(res.x)
    return dict(features=net.NAMES,mean=mean.tolist(),scale=sd.tolist(),w0=A.tolist(),b0=b.tolist(),w1=V.tolist(),
        n_params=len(res.x),iterations=int(res.nit),converged=bool(res.success),loss=float(res.fun),training_workers=len(rr),
        training_maps=len(counts),feature_sha256=sha((HERE/'model.py').read_bytes()),manifest=MANIFEST)


def summarize(rr,w):
    maps={};row_out=[];runtime_ms=[]
    for r in rr:
        start=time.perf_counter_ns();scores=net.score(r['X'],w);runtime_ms.append((time.perf_counter_ns()-start)/1e6)
        acts={'distance':r['distance_action'],'risk':r['risk_action'],'issued':r['issued'],'learned':choice(scores,r['home'])}
        value=[x['value'] for x in r['labels']]
        row_out.append({k:r[k] for k in ('game','seat','t','uid','load','distance','stay_carbon','factual')}|dict(chosen=acts,features_sha256=sha(r['X'].tobytes()),prediction=scores.tolist(),labels=r['labels']))
        m=maps.setdefault(r['game'],dict(n=0,methods={k:Counter() for k in acts},changes=Counter()))
        m['n']+=1
        for method,a in acts.items():
            d=m['methods'][method];v=r['labels'][a]
            d['regret']+=max(value)-value[a];d['value']+=value[a]
            d['bank_1_4']+=int(v['event']=='bank' and v['offset']<=4);d['bank_5_6']+=int(v['event']=='bank' and v['offset']>=5)
            d['stolen']+=int(v['event']=='stolen');d['held_to_collect']+=int(a==0 and v['gathered']>0)
            d['retained_gather_to_bank']+=v['gathered'] if v['event']=='bank' else 0
            m['changes'][method]+=int(a!=acts['issued'])
    comparison={k:float(np.mean([v['methods'][k]['regret']/v['n'] for v in maps.values()]))
                for k in ('distance','risk','issued','learned')}
    ngood={k:sum(v['methods']['learned']['regret']<v['methods'][k]['regret'] for v in maps.values()) for k in ('distance','risk','issued')}
    return dict(maps=maps,mean_map_regret=comparison,lower_regret_maps=ngood,total=len(rr),maps_present=len(maps),
                verdict=("PASS" if len(maps)==16 and all(comparison['learned']<comparison[k] for k in ('distance','risk')) else 'FAIL'),
                rows=row_out,p50_scorer_ms=float(np.median(runtime_ms)),p95_scorer_ms=float(np.quantile(runtime_ms,.95)))


def rule_checks():
    """Compare the two label primitives to the actual competition board one complete step at a time."""
    engine=ROOT/'code/env_src/carbon_challenge-master'
    sys.path.insert(0,str(engine))
    from zerosum_env.envs.carbon.helpers import Board
    from zerosum_env.envs.carbon import idgen
    schema=json.loads((engine/'zerosum_env/envs/carbon/carbon.json').read_bytes())['configuration']
    cfg={k:v['default'] for k,v in schema.items() if isinstance(v,dict) and 'default' in v}
    results={}
    def case(name,at=112,own=(),enemy=(),friends=None,rivals=None,orders=None,carbon=0.,
             decrease=0.,expected_cash=None,expected_count=1,expect_birth=False,expect_flip=False):
        settings={**cfg,'collectDecreaseRate':decrease}
        players=[[100.,{'a':30},{'f':[at,100.,'COLLECTOR'],**(friends or {})},dict(own)],
                 [100.,{'b':31},rivals or {},dict(enemy)]]
        visible=[0.]*225;visible[at]=carbon
        tree_meta={tid:[None,0.] for side in players for tid in side[3]}
        obs=dict(players=players,carbon=visible,trees=tree_meta,
                 remainingOverageTime=5,player=0,step=25)
        commands=orders or [{},{}]
        idgen.reset()
        after=Board(obs,settings,commands).next().observation['players']
        focal=rules.Arrival(0,'f','COLLECTOR',100.)
        occupied=rules.arrivals(players,commands,(0,'f'),settings)
        pred_n=collector_count(occupied,focal,at,players,settings)
        actual_n=sum(w[2]=='COLLECTOR' for w in after[0][2].values())
        assert actual_n==pred_n==expected_count,(name,pred_n,actual_n,expected_count)
        oldids=set(tree_meta)
        if expect_birth:
            assert set(after[1][3])-oldids,name
        if expect_flip:
            assert 'tr' in after[1][3] and 'tr' in players[0][3],name
        gain=gather(players,at,focal,occupied,visible[at],settings)
        tax=tree_tax_fraction(at,enemy_tax_cells(players,after,0),settings)
        carried=(100.+gain)*(1-tax)
        actual=after[0][2]['f'][1]
        assert abs(actual-carried)<1e-8 and abs(carried-expected_cash)<1e-8,(name,carried,actual,expected_cash)
        results[name]={'official_cargo':actual,'label_cargo':carried,'tax_fraction':tax,
                       'harvest':gain,'official_collectors':actual_n,'label_collectors':pred_n}
    case('tree_under',enemy=[('tr',[112,3])],expected_cash=90.)
    case('tree_cardinal',enemy=[('tr',[113,3])],expected_cash=90.)
    case('tree_diagonal',enemy=[('tr',[128,3])],expected_cash=90.)
    case('tree_diagonal_wrapped',at=0,enemy=[('tr',[224,3])],expected_cash=90.)
    case('two_overlapping_and_single_deduction',enemy=[('t1',[128,2]),('t2',[111,3])],expected_cash=80.)
    case('own_tree_exempt',own=[('tr',[128,3])],expected_cash=100.)
    case('new_tree_exempt',rivals={'p':[128,0.,'PLANTER']},expect_birth=True,expected_cash=100.)
    case('tree_aged_one_taken_by_enemy',own=[('tr',[128,1])],
         rivals={'p':[128,0.,'PLANTER']},expect_flip=True,expected_cash=90.)
    case('ripe_tree_bare_before_gather',enemy=[('tr',[112,cfg['treeLifespan']])],
         expected_cash=100.+cfg['co2FrmWithered']*cfg['initialCollectRate'])
    case('fresh_recruit_in_population',orders=[{'a':'RECCOLLECTOR'},{}],carbon=50.,decrease=.03,
         expected_count=2,expected_cash=112.)
    case('friendly_collision_tie_removed',friends={'w':[140,0.,'COLLECTOR']},
         rivals={'x':[141,0.,'COLLECTOR']},orders=[{},{'x':'LEFT'}],carbon=50.,decrease=.03,
         expected_count=1,expected_cash=113.5)
    case('empty_worker_at_rival_base_removed',friends={'w':[32,0.,'COLLECTOR']},
         orders=[{'w':'LEFT'},{}],carbon=50.,decrease=.03,expected_count=1,expected_cash=113.5)
    return {'engine_sha256':sha((engine/'zerosum_env/envs/carbon/helpers.py').read_bytes()),
            'checks':results}


def main():
    p=argparse.ArgumentParser();p.add_argument('split',choices=['train','held','rules']);p.add_argument('--out',type=Path,required=True);p.add_argument('--weights-sha')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    raw=(HERE/'train_manifest.json').read_bytes();assert sha(raw)==MANIFEST
    checks=rule_checks()
    if args.split=='rules':
        print(json.dumps(checks, indent=2)); return
    cards=[c for c in json.loads(raw)['tickets'] if c['split']==args.split]
    if args.split=='held':
        b=(args.out/'weights.json').read_bytes();assert args.weights_sha and sha(b)==args.weights_sha
    rr=collect(cards)
    if args.split=='train':print('WEIGHTS',save(args.out/'weights.json',train(rr)),len(rr),flush=True)
    else:
        rep=summarize(rr,json.loads(b));rep['weight_sha256']=sha(b);rep['manifest_sha256']=MANIFEST;rep['official_rule_tests']=checks;rep['validation_note']='Corrected rules, same 16 previously exposed v2 held maps.'
        print('EVAL',save(args.out/'offline.json',rep),rep['mean_map_regret'],rep['total'],rep['verdict'])


if __name__=='__main__':main()

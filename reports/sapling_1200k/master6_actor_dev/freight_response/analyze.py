"""Report realised banking before the opponent's heaviest ten-step window and select prospective interventions separately."""
from collections import Counter,defaultdict
from functools import lru_cache
import argparse,gzip,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE/'root_recovery_branches'));sys.path.insert(0,str(BASE))
from branches import destination,distances,MOVES
from all_top5.economy_extract import FastCarbon,CashEvents
PLAN=json.loads((HERE/'protocol.json').read_bytes());N=15;D=distances(N)
def read(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def digest(x):return hashlib.sha256(x).hexdigest()
def save(p,x):
    blob=(json.dumps(x,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
    if p.suffix=='.gz':blob=gzip.compress(blob,mtime=0)
    t=p.with_suffix(p.suffix+'.tmp');t.write_bytes(blob);t.replace(p)
from policy import home,routes,geometry,ranked_actions,choose_intervention
def reconstruct(raw):
    seat=raw['candidate_seat'];T=len(raw['actions']);cfg=raw['game_configuration'];sim=FastCarbon(cfg).reset(seed=raw['seed']);sim.cash=CashEvents(sim.cash)
    hist=[];records=[];banks=[];leg=defaultdict(list);assets=[]
    for t,act in enumerate(raw['actions']):
        obs=sim.obs(seat);hist.append(obs);roster={uid:(p,*w) for p,pl in enumerate(obs['players']) for uid,w in pl[2].items()}
        terrain=list(sim.carbon);trees={tr.pos:tr.age for tr in sim.trees.values()};bases={v for pl in obs['players'] for v in pl[1].values()}
        dest={u:destination(v[1],act[v[0]].get(u) or 'NONE',N) for u,v in roster.items()}
        sim.cash.events.clear();sim.step(*act);post={w.pos:w for w in sim.workers.values()}
        for dead,owner,price,credit,winner in sim.last_collisions:
            if winner is None or price<=0 or dead not in dest:continue
            survivor=post.get(dest[dead])
            if survivor and survivor.is_collector and survivor.player==winner:
                leg[survivor.id].append([t+1,'captured_enemy' if owner!=winner else 'allied_transfer',dest[dead],round(price,5),dead])
        for ev in sim.cash.events:
            kind,p,val,*data=ev
            if kind not in ('own_delivery','opponent_worker_delivery_to_this_base'):continue
            uid,cell=data;entries=leg.pop(uid,[])
            banks.append(dict(at=t+1,seat=p,worker=uid,cell=cell,amount=round(val,5),type=kind,
                              provenance=entries,gross_acquired=round(sum(x[3] for x in entries),5)))
        counts=Counter(w.player for w in sim.workers.values() if w.is_collector)
        for uid,w in sim.workers.items():
            if not w.is_collector or uid not in roster or act[w.player].get(uid) not in ('NONE',None) or w.pos in bases:continue
            if w.pos in trees and trees[w.pos]<cfg['treeLifespan']:continue
            carbon=cfg['co2FrmWithered'] if w.pos in trees else terrain[w.pos]
            rate=max(0,cfg['initialCollectRate']-counts[w.player]*cfg['collectDecreaseRate'])
            if carbon*rate>0:leg[uid].append([t+1,'mined',w.pos,round(carbon*rate,5),None])
        records.append(list(sim.cash))
    hist.append(sim.obs(seat))
    return hist,banks,records
def source_cards():
    for key in PLAN['sources']:
        d=BASE/key;rep=read(d/'results.json');off=read(d/'official_replay.json')['runs'][d.name]['games']
        assert rep['status']=='complete'
        for g in rep['games']:
            o=next(a for a in off if (a['seed'],a['seat'])==(g['seed'],g['candidate_seat']));p=d/g['raw_file']
            assert g['valid'] and not g['errors'] and g['raw_sha256']==digest(p.read_bytes())==o['transcript_sha256'] and o['statuses']==['DONE','DONE']
            yield key,p,g
def hindsight(source,raw,hist,banks):
    seat=raw['candidate_seat'];op=1-seat;T=len(raw['actions']);cash=[0.]*(T+1)
    for b in banks:
        if b['seat']==op and b['type']=='own_delivery':cash[b['at']]+=b['amount']
    start=max(range(T-9),key=lambda j:(sum(cash[j+1:j+11]),-j));end=start+10
    paid=[b for b in banks if start<b['at']<=end and b['seat']==op and b['type']=='own_delivery']
    eids=list(dict.fromkeys(r['worker'] for r in paid));by_worker={}
    for uid in eids:
        roots=[]
        for t in sorted(set(max(0,start-l) for l in (10,5,3,0))):
            o=hist[t];w=o['players'][op][2].get(uid); offers,geom=geometry(o,raw['actions'][t][seat],uid,True)
            roots.append(dict(t=t,lag=start-t,enemy_cell=w[0] if w else None,load=round(w[1],3) if w else None,
                home_dist=home(w[0],o['players'][op][1].values()) if w else None,**geom))
        by_worker[uid]=dict(deposits=[b for b in paid if b['worker']==uid],checkpoints=roots)
    return dict(source=source,seed=raw['seed'],seat=seat,windows_checked=T-9,start=start,end=end,opponent_banked=round(sum(cash[start+1:end+1]),3),workers=by_worker,full_tape_hash=raw['pack_sha256'])
def scan(model_path=None):
    preds=None
    if model_path:
        sys.path.insert(0,str(HERE.parent/'freight_horizon'))
        from predictor import FreightPredictor
        preds=FreightPredictor(model_path)
    output=[];regs={role['name']:dict(map=role,seen=Counter(),qualified=[],geometry=[],selected=None) for role in PLAN['roles']}
    for src,path,g in source_cards():
        raw=read(path);hist,banks,records=reconstruct(raw);assert hist[-1]['step']==len(raw['actions'])
        # Full-horizon physical cash must match its registered official terminal. Extinction is scored separately.
        if g['termination']=='official_episode_limit':assert records[-1]==g['official_score_by_seat']
        if not preds:output.append(hindsight(src,raw,hist,banks))
        for role in PLAN['roles']:
            if src!=role['source'] or raw['seed']!=role['seed']:continue
            assert raw['pack_sha256']==role['sha'] and digest((ROOT/role['pack']).read_bytes())==role['sha'];s=raw['candidate_seat'];target=regs[role['name']]
            for t in range(219,min(290,len(raw['actions']))):
                obs=hist[t];counts=target['seen'];counts['joint_step']+=1;enemy=obs['players'][1-s]
                result=None
                for uid,(pos,load,typ) in enemy[2].items():
                    if typ!='COLLECTOR':continue
                    counts['enemy_collector']+=1
                    if load<60 or not 2<=home(pos,enemy[1].values())<=7:continue
                    counts['loaded_distance']+=1;ps=routes(pos,tuple(sorted(enemy[1].values())))
                    if len({p[0] for p in ps})<2:continue
                    counts['two_exits']+=1;rows,_=geometry(obs,raw['actions'][t][s],uid)
                    if not rows:continue
                    counts['target_with_legal_cover_gain']+=1;counts['actions_with_legal_cover_gain']+=len(rows)
                    if not preds:
                        target['geometry']+=rows;continue
                    if result is None:result=preds.predict(obs,hist[max(0,t-5):t])
                    if uid not in result:raise ValueError('predictor missing eligible loaded opponent '+uid)
                    q=result[uid]
                    chosen=ranked_actions(obs,raw['game_configuration'],s,raw['actions'][t][s],{uid:q})
                    if not chosen:continue
                    counts['target_passing_model']+=1;counts['actions_passing_model']+=len(chosen)
                    for row in chosen:
                        row.update(source=str(path.relative_to(ROOT)),source_sha256=digest(path.read_bytes()))
                        target['qualified'].append(row)
    bind=dict(policy_sha256=digest((HERE/'policy.py').read_bytes()),protocol_sha256=digest((HERE/'protocol.json').read_bytes()),script_sha256=digest(Path(__file__).read_bytes()))
    if preds:
        for reg in regs.values():
            rs=reg['qualified'];reg['selected']=min(rs,key=lambda r:(-r['rank_score'],-r['minimum_exit_coverage'],r['tie'])) if rs else None
            del reg['geometry']
        save(HERE/'roots.json',dict(**bind,model_sha256=digest(Path(model_path).read_bytes()),maps=regs))
    else:
        save(HERE/'surges.json.gz',dict(**bind,games=output));save(HERE/'geometry.json.gz',dict(**bind,maps=regs))
    print([(n,dict(r['seen']),r['selected']) for n,r in regs.items()],flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path);a=ap.parse_args();scan(a.model)

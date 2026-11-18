#!/usr/bin/env python3
"""Small held-out behavior smoke, NOT a terminal-strength or resource certification.
8 games/arm = four maps x both candidate seats; states t=0..60 inclusive.
Uses official int8 cnet2_pack, teacher=2, all optional guards disabled.
Simulator trace observes exact collect/deposit events without changing transitions.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import argparse, collections, hashlib, importlib.util, inspect, json, pathlib, random, shutil, statistics, sys, time, re

SEEDS=[620010000,620010001,620010002,620010003]
SOURCE_NAMES=['cnet2_pack.py','cnet2_feat.py','cnet2_np.py','cnet2_agent.py','fastsim.py','policy_v0_5.py']
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def atomic_json(path,obj):
    p=pathlib.Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n'); q.replace(p)
def module(path,tag):
    s=importlib.util.spec_from_file_location(tag,str(path)); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

class Observer:
    """Line events from the unchanged authoritative simulator, no frame mutation.
    closed_loop: collector collected >0, subsequently delivered >0 to own base;
    Only current positive cargo carries provenance; delivery/empty cargo clears it.
    Friendly-merge provenance propagates to survivor; not a cargo-origin mass balance.
    Base-collision credit is separately reported, not counted as normal delivery.
    """
    def __init__(self,simclass):
        self.code=simclass.step.__code__; lines,start=inspect.getsourcelines(simclass.step)
        needles={'collect':'w.carbon += delta','deposit':'self.cash[p] += w.carbon - smelt',
                 'merge':'winner.carbon += w.carbon; gets = winner.player',
                 'base_collision':'self.cash[self.base_at[pos][0]] += w.carbon; gets = self.base_at[pos][0]'}
        self.lines={}
        for key,needle in needles.items():
            hits=[start+i for i,s in enumerate(lines) if s.strip()==needle]
            if len(hits)!=1: raise RuntimeError(f'Unrecognized simulator event {key}: {hits}')
            self.lines[hits[0]]=key
        self.d=[collections.Counter(),collections.Counter()]; self.collectors=[set(),set()]; self.provenance={}; self.events=[]
    def trace(self,frame,event,arg):
        if frame.f_code is not self.code: return None
        if event!='line' or frame.f_lineno not in self.lines: return self.trace
        x=frame.f_locals; k=self.lines[frame.f_lineno]; w=x['w']; p=w.player; step=x['self'].step_count+1
        if k=='collect':
            amount=float(x['delta']); self.d[p]['collect_events']+=1; self.d[p]['collected_carbon']+=amount
            self.collectors[p].add(w.id); self.provenance.setdefault(w.id,set()).add(w.id)
        elif k=='merge':
            v=x['winner']
            if v.player==p and w.carbon>0: self.provenance.setdefault(v.id,set()).update(self.provenance.get(w.id,set()))
            self.provenance.pop(w.id,None)
            return self.trace
        elif k=='base_collision':
            receiver=x['self'].base_at[x['pos']][0]; amount=float(w.carbon)
            if amount>1e-9: self.d[receiver]['base_collision_credit']+=amount
            self.provenance.pop(w.id,None)
            return self.trace
        else:
            receiver=x['p']; amount=float(w.carbon-x['smelt'])
            carried=self.provenance.pop(w.id,set())  # phase 3 zeroes cargo, including enemy-base delivery
            if amount<=1e-9: return self.trace
            self.d[receiver]['base_received_carbon']+=amount
            if receiver!=p:
                self.d[receiver]['enemy_base_donation']+=amount; return self.trace
            self.d[p]['positive_delivery_events']+=1; self.d[p]['own_positive_delivery_carbon']+=amount
            if carried: self.d[p]['closed_loop_events']+=1
        self.events.append(dict(step=step,player=p,worker=w.id,event=k,amount=amount))
        return self.trace
    def step(self,sim,acts):
        old=sys.gettrace()
        if old is not None: raise RuntimeError('Existing tracer: refusing to overwrite')
        sys.settrace(self.trace)
        try: sim.step(*acts)
        finally: sys.settrace(old)
        # Other phases can exhaust cargo or kill the carrier. No lifetime flags.
        for wid in list(self.provenance):
            if wid not in sim.workers or sim.workers[wid].carbon<=1e-9: self.provenance.pop(wid,None)

def snapshot(sim,obs,p,errors,calls):
    workers=[w for w in sim.workers.values() if w.player==p]; collectors=[w for w in workers if w.is_collector]
    d=obs.d[p]
    return dict(state_t=sim.step_count,recruited_total=sim.worker_counters[p],new_trees_planted=sim.tree_counters[p],
        alive_workers=len(workers),alive_collectors=len(collectors),alive_trees=len(sim.player_trees[p]),
        positive_cargo_collectors=sum(w.carbon>1e-9 for w in collectors),collectors_ever_collected=len(obs.collectors[p]),
        cargo=sum(w.carbon for w in workers),cash=float(sim.cash[p]),agent_calls=calls[p],agent_errors=errors[p],
        completed_collection_delivery_loop=bool(d['closed_loop_events']),
        **{k:float(d[k]) for k in ['collect_events','collected_carbon','positive_delivery_events','own_positive_delivery_carbon',
          'closed_loop_events','base_received_carbon','enemy_base_donation','base_collision_credit']})

def selftest(FastCarbon,Worker):
    # One real harvest then own-base delivery. Trace must leave states byte-identical.
    a=FastCarbon().reset(full_carbon=[0.0]*225); bpos=a.bases[0][0][1]; pos=a._move_nbr['RIGHT'][bpos]
    wid=a._new_worker_id(0); a.workers[wid]=Worker(wid,0,pos,0.0,True); a.player_workers[0].append(wid); a.carbon[pos]=100.
    b=FastCarbon().set_state(a.get_state()); o=Observer(FastCarbon)
    for acts in [({},{}),({wid:'LEFT'}, {})]:
        o.step(a,acts); b.step(*acts); assert a.get_state()==b.get_state(), 'Trace changed transition'
    assert o.d[0]['collected_carbon']>0 and o.d[0]['own_positive_delivery_carbon']>0 and o.d[0]['closed_loop_events']==1
    # After delivering all harvested cargo, collecting ONLY enemy cargo must
    # not claim a second harvest-return loop. Enemy moves onto our empty carrier.
    enemy=a._new_worker_id(1); a.workers[enemy]=Worker(enemy,1,a._move_nbr['RIGHT'][bpos],12.,True); a.player_workers[1].append(enemy)
    o.step(a,({}, {enemy:'LEFT'}))
    assert o.d[0]['positive_delivery_events']==2 and o.d[0]['closed_loop_events']==1
    assert not o.provenance.get(wid)
    # Friendly merge at base preserves delivery, no inference from disappearing IDs.
    c=FastCarbon().reset(full_carbon=[0.0]*225); base=c.bases[0][0][1]; ids=[]
    for action,cargo in [('LEFT',10.),('RIGHT',20.)]:
        wid=c._new_worker_id(0); ids.append(wid); c.workers[wid]=Worker(wid,0,c._move_nbr[action][base],cargo,True); c.player_workers[0].append(wid)
    o=Observer(FastCarbon); o.provenance[ids[1]]={ids[1]}; o.step(c,({ids[0]:'RIGHT',ids[1]:'LEFT'},{}))
    assert o.d[0]['own_positive_delivery_carbon']==30 and o.d[0]['closed_loop_events']==1
    return {'harvest_delivery_trace_parity':True,'friendly_merge_30_carbon_preserved':True,'post_empty_enemy_cargo_not_second_loop':True}

def play(seed,side,pack,opponent,FastCarbon):
    random.seed(7919*seed+104729*side+31)
    sim=FastCarbon().reset(seed=seed); mon=Observer(FastCarbon)
    ag=module(pack,f'candidate_{seed}_{side}').agent; opp=module(opponent,f'opponent_{seed}_{side}').make_agent()
    agents=[None,None]; agents[side]=ag; agents[1-side]=opp
    errors=[0,0]; calls=[0,0]; errs=[]; snaps={}; terminal=False; failed=False; t0=time.monotonic(); cpu0=time.process_time()
    for step in range(1,61):
        acts=[]
        for p in range(2):
            calls[p]+=1
            try:
                action=agents[p](sim.obs(p),sim.config)
                if not isinstance(action,dict): raise TypeError('agent returned non-dict')
                valid_ids=set(sim.player_workers[p]) | {bid for bid,_ in sim.bases[p]}
                if set(action)-valid_ids: raise ValueError('agent returned foreign/unknown entity id')
                if any(v not in ('UP','RIGHT','DOWN','LEFT','RECCOLLECTOR','RECPLANTER',None) for v in action.values()): raise ValueError('unknown action')
                acts.append(action)
            except Exception as e:
                errors[p]+=1; errs.append(dict(step=step,side=p,type=type(e).__name__,message=str(e))); acts.append({}); failed=True
        if failed: break
        mon.step(sim,acts)
        terminal=any(not sim.player_workers[p] and not sim.player_trees[p] and (not sim.bases[p] or sim.cash[p]<min(sim.config['recCollectorCost'],sim.config['recPlanterCost'])) for p in range(2))
        if step in (30,60): snaps[str(step)]={'candidate':snapshot(sim,mon,side,errors,calls),'opponent':snapshot(sim,mon,1-side,errors,calls)}
        if terminal: break
    if terminal:
        for t in (30,60):
            if str(t) not in snaps:
                snaps[str(t)]={'candidate':snapshot(sim,mon,side,errors,calls),'opponent':snapshot(sim,mon,1-side,errors,calls),'early_terminal_carried_forward':True}
    return dict(seed=seed,candidate_side=side,transitions=sim.step_count,terminal=terminal,failed=failed,
       errors=errs,agent_calls_by_side=calls,agent_errors_by_side=errors,snapshots=snaps,
       final_state_sha256=hashlib.sha256(json.dumps(sim.get_state(),sort_keys=True).encode()).hexdigest(),
       events=mon.events,wall_seconds=time.monotonic()-t0,cpu_seconds=time.process_time()-cpu0)

def summarize(rows):
    out={}
    for t in ('30','60'):
        out[t]={}
        for role in ('candidate','opponent'):
            vals=[r['snapshots'][t][role] for r in rows if t in r['snapshots']]
            z={'scheduled_games':len(rows),'games_with_snapshot':len(vals),'completed_loop_games':sum(v['completed_collection_delivery_loop'] for v in vals),
               'games_with_positive_delivery':sum(v['positive_delivery_events']>0 for v in vals),'games_with_positive_cargo_collectors':sum(v['positive_cargo_collectors']>0 for v in vals)}
            for k in ('recruited_total','new_trees_planted','positive_cargo_collectors','collectors_ever_collected','collected_carbon','positive_delivery_events','own_positive_delivery_carbon','cash','cargo','alive_trees','alive_collectors'):
                z[k]={'sum':sum(v[k] for v in vals),'mean':statistics.fmean(v[k] for v in vals) if vals else None,'denominator_games':len(vals)}
            z['agent_errors']=sum(r['agent_errors_by_side'][r['candidate_side'] if role=='candidate' else 1-r['candidate_side']] for r in rows)
            out[t][role]=z
    return out

def make_protocol(code,seeds):
    sources={n:sha(code/n) for n in SOURCE_NAMES}
    protocol={'version':1,'seeds':seeds,'candidate_seats':[0,1],'transitions_per_game':60,'snapshot_states':[30,60],
      'teacher':2,'guards':False,'quantization':'int8','opponent':'policy_v0_5.make_agent; frozen same source as training teacher, distinct maps',
      'source_sha256':sources,'training_seed_exclusion_declared':[620000000,620000001,620000002,620000003,620000004,620000005],
      'seed_registry_check':('No literal 620010000..620010003 in reports/code/automation text at registration; training agent separately confirmed 620000000..005. Not a claim about unrecorded external runs.' if seeds==SEEDS else 'New held-out smoke block reserved outside training seeds 620000000..005. No claim about unrecorded external runs.'),
      'interpretation':'Held-out early behavior smoke only. 4 maps × 2 candidate seats = 8 games; seats share map, not 8 independent maps. No terminal winrate or strength claim. Trace timing is not resource certification.'}
    return protocol

def run_arm(label,weights,cache,out,protocol,code,FastCarbon,checks):
    work=cache/label; work.mkdir(parents=True,exist_ok=True)
    wcopy=work/'weights.npz'
    if pathlib.Path(weights).resolve()!=wcopy.resolve(): shutil.copyfile(weights,wcopy)
    pack=work/'agent.py'
    import cnet2_pack
    cnet2_pack.QMODE='int8'; cnet2_pack.main(str(wcopy),str(pack),teacher=2,shield=0,nocollide=0,endgame=0,pshield=0,rguard=0)
    result={'arm':label,'observer_revision':2,'measurement_script_sha256':sha(__file__),'protocol':protocol,'weight_filename':pathlib.Path(weights).name,'weight_sha256':sha(weights),
       'pack_sha256':sha(pack),'pack_md5':hashlib.md5(pack.read_bytes()).hexdigest(),'pack_bytes':pack.stat().st_size,
       'checks':checks,'status':'running','games':[]}
    dest=out/(label+'.json'); atomic_json(dest,result)
    for seed in protocol['seeds']:
        for side in (0,1):
            row=play(seed,side,pack,code/'policy_v0_5.py',FastCarbon)
            events=row.pop('events')
            row['event_records_sha256']=hashlib.sha256(json.dumps(events,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            row['team']={'candidate_seat':side,'opponent_seat':1-side}
            row['termination']='agent_error' if row['failed'] else ('early_extinction' if row['terminal'] else 'short_horizon_cap')
            result['games'].append(row); result['summary']=summarize(result['games']); atomic_json(dest,result)
            print(json.dumps({'arm':label,'seed':seed,'side':side,'transitions':row['transitions'],'errors':row['errors'],
                'candidate_t60':row['snapshots'].get('60',{}).get('candidate')}),flush=True)
            if row['failed']: raise RuntimeError('Agent failure: saved diagnostic; no fallback continuation')
    result['status']='complete'; result['total_cpu_seconds']=sum(r['cpu_seconds'] for r in result['games'])
    result['total_wall_seconds']=sum(r['wall_seconds'] for r in result['games']); result['summary']=summarize(result['games']); atomic_json(dest,result)
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[2])
    ap.add_argument('--weight',action='append',default=[],metavar='LABEL=PATH',help='Repeat for all arms; frozen labels, no automatic model selection')
    ap.add_argument('--cache',type=pathlib.Path,default=pathlib.Path('/tmp/carbon-sapling-smoke-cache'))
    ap.add_argument('--out',type=pathlib.Path,default=pathlib.Path('/tmp/carbon-sapling-smoke-results'))
    ap.add_argument('--protocol',type=pathlib.Path,help='Existing frozen protocol must match exactly; absent file is created before games')
    ap.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    ap.add_argument('--self-test-only',action='store_true'); args=ap.parse_args()
    if not 1<=len(args.seeds)<=4 or len(set(args.seeds))!=len(args.seeds): ap.error('Use 1..4 unique maps, at most 8 games/arm')
    if any(s in range(620000000,620000006) for s in args.seeds): ap.error('Training/validation map cannot be used for this holdout smoke')
    code=args.repo.resolve()/'code'; sys.path.insert(0,str(code))
    from fastsim import FastCarbon,_Worker
    checks=selftest(FastCarbon,_Worker)
    if args.self_test_only: print(json.dumps(checks)); return
    if not args.weight: ap.error('Supply --weight LABEL=PATH at least once')
    arms=[]
    for spec in args.weight:
        label,sep,path=spec.partition('=')
        if not sep or not re.fullmatch('[A-Za-z0-9_-]+',label): ap.error('Expected safe LABEL=PATH')
        if label in [x[0] for x in arms]: ap.error('Duplicate arm label')
        if not pathlib.Path(path).is_file(): ap.error('Weight file not found: '+path)
        arms.append((label,path))
    protocol=make_protocol(code,args.seeds); pp=args.protocol or args.out/'protocol.json'
    if pp.exists():
        if json.loads(pp.read_text())!=protocol: raise RuntimeError('Protocol/source changed: do not overwrite frozen protocol')
    else: atomic_json(pp,protocol)
    for label,weights in arms:
        dest=args.out/(label+'.json')
        if dest.exists(): raise RuntimeError('Output exists; use a fresh output directory to avoid overwriting evidence: '+str(dest))
        run_arm(label,weights,args.cache,args.out,protocol,code,FastCarbon,checks)
if __name__=='__main__': main()

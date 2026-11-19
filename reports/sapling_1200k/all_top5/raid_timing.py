"""Pairwise pre-action raid opportunities and realized economic outcomes in original replays."""
import argparse
from collections import Counter,defaultdict
import csv
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path
import numpy as np
from defense import HERE,DEST,DIST,MOVE,RECRUIT
from defense_run import atomic,compact

N=225
JOINT=[[tuple(q for q in DEST[p] if q in DEST[r]) for r in range(N)] for p in range(N)]
COLS=('game','step','seat','hunter','victim','hunter_cell','victim_cell','distance','own_home','enemy_home',
      'enemy_toward_home','enemy_last_move','safe_exits','joint_cells','non_enemy_base_cells','supported_cells',
      'escort_needed','could_bank_now','our_cargo','their_cargo',
      'our_action','their_action','selected_crossing','met','captured','banked','carried_at_end','taxed',
      'lost_again','enemy_base_refund','missed_conditional','was_enemy_base_only')
INTS={'our_cargo','their_cargo','captured','banked','carried_at_end','taxed','lost_again','enemy_base_refund'}
DTYPE=np.dtype([(k,'<f4' if k in INTS else '<i4') for k in COLS])
def suffix(uid):return int(uid.rsplit('-',1)[1])

def worker_history(frames,seat,uid,t,bases):
    if t==0:return 0,5
    now=frames[t][0]['observation']['players'][seat][2][uid][0]
    before=frames[t-1][0]['observation']['players'][seat][2].get(uid)
    if before is None:return 0,5
    last=DEST[before[0]].index(now) if now in DEST[before[0]] else 5
    run=0;d=min(DIST[now][b] for b in bases)
    for q in range(t-1,max(-1,t-4),-1):
        w=frames[q][0]['observation']['players'][seat][2].get(uid)
        if w is None:break
        prev=min(DIST[w[0]][b] for b in bases)
        if prev<=d:break
        run+=1;d=prev
    return run,last


def game_rows(index,game,packet):
    frames=game['steps'];cfg=game['configuration']
    lots={(e['t'],e['hunter'],e['victim']):e for e in packet['lots'] if e['method']=='collector'}
    any_capture={(e['t'],e['victim']) for e in packet['lots']}
    refunds={(t,w,v):m for t,s,v,w,p,m in packet['enemy_base_refunds']}
    rows=[]
    for t,frame in enumerate(frames[:-1]):
        pl=frame[0]['observation']['players'];act=[v['action'] or {} for v in frames[t+1]]
        coll=[[(uid,p,c) for uid,(p,c,k) in a[2].items() if k=='COLLECTOR'] for a in pl]
        home=[set(a[1].values()) for a in pl];landed=defaultdict(list)
        count_c=sum(map(len,coll));count_p=sum(len(a[2]) for a in pl)-count_c
        costs={'COLLECTOR':cfg['recCollectorCost']+count_c*cfg['recCollectorIncreaseCost'],'PLANTER':cfg['recPlanterCost']+count_p*cfg['recPlanterIncreaseCost']}
        endpoints=[{},{}]
        for s in (0,1):
            bal=pl[s][0];n=len(pl[s][2])
            for wid,p,c in coll[s]:
                end=DEST[p][MOVE[act[s].get(wid)]];endpoints[s][wid]=end;landed[end].append((wid,c))
            for bid,p in pl[s][1].items():
                typ=RECRUIT.get(act[s].get(bid))
                if typ is not None and bal>=costs[typ] and n<cfg['workerLimit']:
                    bal-=costs[typ];n+=1
                    if typ=='COLLECTOR':landed[p].append((bid,0.))
        for s in (0,1):
            enemy=1-s;os=coll[s];vs=coll[enemy];own_b=home[s];foe_b=home[enemy]
            for vid,vp,vc in vs:
                if vc<50:continue
                hunters=[(wid,p,c) for wid,p,c in os if c<vc and DIST[p][vp]<=2]
                if not hunters:continue
                threatening={q for _,p,c in os if c<=vc and DIST[p][vp]<=2 for q in DEST[p]}
                exits=sum(q not in threatening and q not in own_b for q in DEST[vp])
                towards,last=worker_history(frames,enemy,vid,t,foe_b)
                vdist=min(DIST[vp][b] for b in foe_b);vend=endpoints[enemy][vid]
                for wid,wp,wc in hunters:
                    common=JOINT[wp][vp];legal=[q for q in common if q not in foe_b]
                    # Conservative contact grid: remove cells another equally light enemy could also reach.
                    low=[ep for eid,ep,ec in vs if eid!=vid and ec<=wc and DIST[wp][ep]<=2]
                    robust=sum(not any(q in DEST[e] for e in low) for q in legal)
                    # Escort is observable proximity, not proof the hunter would have prevented a loss.
                    escort=any(ally!=wid and cargo>=50 and DIST[ap][wp]<=2 and any(DIST[ep][ap]<=1 and ec<=cargo for _,ep,ec in vs) for ally,ap,cargo in os)
                    wend=endpoints[s][wid]
                    event=lots.get((t+1,wid,vid),{});fate=event.get('fate',{})
                    refund=refunds.get((t+1,wid,vid),0.)
                    # Counterfactual diagnostic ONLY: all other executed commands fixed, route must give non-enemy-base receipt.
                    missed=(wend!=vend and vend in legal and (t+1,vid) not in any_capture and
                            all(other in (wid,vid) or mass>wc for other,mass in landed[vend]))
                    rows.append((index,t,s,suffix(wid),suffix(vid),wp,vp,DIST[wp][vp],min(DIST[wp][p] for p in own_b),vdist,
                                 towards,last,exits,len(common),len(legal),robust,int(escort),int(wc>0 and any(b in DEST[wp] for b in own_b)),
                                 wc,vc,MOVE[act[s].get(wid)],MOVE[act[enemy].get(vid)],int(wend in legal),int(wend==vend),event.get('mass',0.),
                                 fate.get('banked',0.),fate.get('carried_at_terminal',0.),fate.get('tax_paid_to_enemy_trees',0.),
                                 sum(fate.get(k,0.) for k in ('opponent_cash','opponent_worker','ground_or_destroyed')),refund,int(missed),int(not legal)))
    return np.asarray(rows,dtype=DTYPE)


def sample_stats(x,mask):
    a=x[mask];return {'pair_roots':len(a),'selected':int(a['selected_crossing'].sum()),'same_cell':int(a['met'].sum()),
        'won':int((a['captured']>0).sum()),'cashed':int((a['banked']>0).sum()),'won_without_cash':int(((a['captured']>0)&(a['banked']==0)).sum()),
        'taken_carbon':round(float(a['captured'].sum(dtype='f8')),3),'banked_carbon':round(float(a['banked'].sum(dtype='f8')),3),
        'refunded_contacts':int((a['enemy_base_refund']>0).sum()),'missed_single_actor':int(a['missed_conditional'].sum()),
        'unique_games':int(len(np.unique(a['game']))),
        'not_selected':int((a['selected_crossing']==0).sum()),
        'selected_and_met':int(((a['selected_crossing']>0)&(a['met']>0)).sum()),
        'selected_move_into_crossing':int(((a['selected_crossing']>0)&(a['our_action']!=0)).sum())}


def intervals(x,groups,seeds,indices):
    """Paired ratio-estimator bootstrap over entire map seeds, keeping repeated roots/seats together."""
    idx=np.asarray(groups)
    lower=[]
    for mask in indices:
        a=x[mask];g=idx[a['game']]
        den=np.bincount(g,minlength=seeds.shape[1]);buy=np.bincount(g,weights=(a['banked']>0),minlength=seeds.shape[1]);bval=np.bincount(g,weights=a['banked'],minlength=seeds.shape[1])
        sampled=seeds
        nden=den[sampled].sum(axis=1);valid=nden>0
        if not valid.any():lower.append({});continue
        out={}
        for key,arr in [('banking_events_per_1000',buy),('banked_carbon_per_1000',bval)]:
            v=1000*arr[sampled[valid]].sum(axis=1)/nden[valid]
            out[key]=list(map(float,np.quantile(v,[.025,.975])))
        lower.append(out)
    return lower


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--reuse-rows',action='store_true');a=ap.parse_args()
    man=json.loads((HERE/'manifest.json').read_bytes());games=man['games'];assert all(g['frame_verified'] for g in games)
    receipts=[];parts=[];ranked={v['id']:v['rank'] for v in man['top5']}
    paths=HERE/'raid_timing_rows.npz'
    if a.reuse_rows:
        previous=json.loads(gzip.decompress((HERE/'raid_timing.json.gz').read_bytes()))
        previous_sources=[(g['id'],g['sha256']) for g in previous['games']]
        current_sources=[(g['id'],g['sha256']) for g in games]
        if previous_sources!=current_sources or previous['rows_sha256']!=sha256(paths.read_bytes()).hexdigest():
            raise ValueError('cached rows belong to another ordered set of sources or have changed')
        with np.load(paths,allow_pickle=False) as table:x=table['rows']
        if x.dtype!=DTYPE:
            raise ValueError('incompatible cached rows')
        counts=np.bincount(x['game'],minlength=len(games))
    for i,g in enumerate(games):
        paper=a.cache/'defense'/(g['id']+'.json.gz');packet=json.loads(gzip.decompress(paper.read_bytes()))
        if packet['sha256']!=g['sha256']:raise ValueError('identity mismatch')
        if not a.reuse_rows:
            blob=(a.cache/g['path']).read_bytes()
            if sha256(blob).hexdigest()!=g['sha256']:raise ValueError('raw hash mismatch')
            arr=game_rows(i,json.loads(gzip.decompress(blob)),packet);parts.append(arr);nr=len(arr)
        else:nr=int(counts[i])
        receipts.append(dict(id=g['id'],seed=g['seed'],sha256=g['sha256'],seat_names=g['seat_names'],
                             top5=[ranked.get(v) for v in g['seat_strategy_ids']],cohort=g['cohort'],
                             frames=packet['transitions'],risk_pair_rows=nr,all_capture_lots=sum(e['method']=='collector' for e in packet['lots'])))
        if i%250==249:print('scanned',i+1,sum(len(p) for p in parts),flush=True)
    if not a.reuse_rows:
        x=np.concatenate(parts);buf=io.BytesIO();np.savez_compressed(buf,rows=x);atomic(paths,buf.getvalue())
    all_valid=np.ones(len(x),dtype=bool)
    stages=[('all',all_valid),('base-only',x['non_enemy_base_cells']==0),('eligible',x['non_enemy_base_cells']>0)]
    risk=x['non_enemy_base_cells']>0
    def bins(name,variable,edges):
        for lo,hi in zip(edges,edges[1:]):stages.append((f'{name} {lo}..{hi-1}',risk&(variable>=lo)&(variable<hi)))
    bins('step',x['step'],[0,60,150,240,280,300]);bins('enemy_home',x['enemy_home'],[0,1,2,4,7,16])
    bins('cargo',x['their_cargo'],[50,100,200,400,1000000]);bins('return_streak',x['enemy_toward_home'],[0,1,2,3,4])
    bins('enemy_safe_exits',x['safe_exits'],[0,1,2,3,6]);bins('our_home',x['own_home'],[0,2,4,7,16])
    bins('cargo_ratio_percent',100*x['our_cargo']/x['their_cargo'],[0,10,50,100]);bins('robust_cells',x['supported_cells'],[0,1,2,6])
    for col in ('could_bank_now','escort_needed'):
        for v in (0,1):stages.append((f'{col} {v}',risk&(x[col]==v)))
    rnk=np.asarray([[v if v else 0 for v in g['top5']] for g in receipts])
    who=rnk[x['game'],x['seat']];opp=rnk[x['game'],1-x['seat']]
    for i in range(6):
        stages.append((f'our_rank {i}',risk&(who==i)));stages.append((f'opp_rank {i}',risk&(opp==i)))
    for dist in (1,2):
        for k in (0,1,2):stages.append((f'distance {dist} safe {k}'+('+' if k==2 else ''),risk&(x['distance']==dist)&((x['safe_exits']>=k) if k==2 else (x['safe_exits']==k))))
    unique_seed,reindex=np.unique([g['seed'] for g in receipts],return_inverse=True)
    bootstrap=np.random.default_rng(93921).integers(0,len(unique_seed),(600,len(unique_seed)))
    ci=intervals(x,reindex,bootstrap,[m for _,m in stages])
    stages_out={k:dict(**sample_stats(x,m),cluster_ci=c) for (k,m),c in zip(stages,ci)}
    # Show only facts from public prediction time; actual commands, label and realized future are kept apart for training.
    source={'algorithm':sha256(Path(__file__).read_bytes()).hexdigest(),'n_unique_seeds':len(unique_seed),'games':receipts,'stages':stages_out,
            'rows_file':paths.name,'rows_sha256':sha256(paths.read_bytes()).hexdigest(),
            'metadata_columns':list(COLS[:5]),'pre_action_columns':list(COLS[5:20]),'observed_actions_and_outcomes':list(COLS[20:]),
            'cases':pick_cases(x,receipts,a.cache)}
    atomic(HERE/'raid_timing.json.gz',gzip.compress(compact(source),mtime=0))
    print('completed:',len(x),stages_out['all'],flush=True)


def pick_cases(x,games,root):
    choices={}
    # Keep original full trajectories with both public movement histories. 'Miss' is immediate feasibility only.
    categories={'banked':x['banked']>0,'won_unbanked':(x['captured']>0)&(x['banked']==0),
                'refunded':x['enemy_base_refund']>0,'missed':x['missed_conditional']>0,
                'forecast_wrong':(x['selected_crossing']>0)&(x['met']==0)&(x['step']>9)}
    for key,mask in categories.items():
        top5=np.asarray([[bool(n) for n in g['top5']] for g in games])[x['game'],x['seat']]
        eligible=np.where(mask & top5 & (x['step']>=10))[0]
        if not len(eligible):continue
        def ranking(i):
            r=x[i]
            return float(r['banked'] if key=='banked' else r['enemy_base_refund'] if key=='refunded' else r['their_cargo'])
        ids=sorted(eligible,key=ranking,reverse=True);picked=set();out=[]
        for i in ids:
            r=x[i];g=games[int(r['game'])]
            if g['id'] in picked:continue
            picked.add(g['id'])
            frame=json.loads(gzip.decompress((root/'replays'/(g['id']+'.json.gz')).read_bytes()))['steps']
            seat=int(r['seat']);wid=f"player-{seat}-worker-{r['hunter']}";vid=f"player-{1-seat}-worker-{r['victim']}";t=int(r['step'])
            traj=[]
            for d in (-10,-5,-3,-2,-1,0,1,3,5,10):
                k=t+1+d
                if not 0<=k<len(frame):continue
                players=frame[k][0]['observation']['players'];cmd=[a['action'] or {} for a in frame[k+1]] if k+1<len(frame) else [{},{}]
                hw=players[seat][2].get(wid);vw=players[1-seat][2].get(vid)
                traj.append({'relative_contact':d,'frame':k,'hunter':hw,'victim':vw,'hunter_next':cmd[seat].get(wid),'victim_next':cmd[1-seat].get(vid),
                             'distance':DIST[hw[0]][vw[0]] if hw and vw else None,'cash':[p[0] for p in players]})
            out.append({'id':g['id'],'hunter_name':g['seat_names'][seat],'victim_name':g['seat_names'][1-seat],'cohort':g['cohort'],
                        'root':{n:r[n].item() for n in x.dtype.names},'trace':traj})
            if len(out)==3:break
        choices[key]=out
    return choices


if __name__=='__main__':main()

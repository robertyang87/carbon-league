"""Read the official actions. Compare the two archived Fable wins without simulating new games."""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('raw',type=Path);p.add_argument('scorecard',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
sys.path.insert(0,str(a.repo/'reports/sapling_1200k/all_top5'));import economy_extract as E
env=json.loads(gzip.decompress(a.raw.read_bytes()));card=json.loads(a.scorecard.read_bytes());rr=card['result'][0];f=env['steps'];cfg=env['configuration'];ps=[x[0]['observation']['players'] for x in f]
me=rr['order'].index('a');them=1-me;parties=[me,them]
assert card['name_a']=='ds-pro-v1' and card['name_b']=='ClaudeFable5.2-v8' and len(f)==300 and rr['scores']=={k:env['rewards'][s] for s,k in enumerate(rr['order'])}
ledger,_,banks,_=E.reconstruct(f,cfg);dists=E.distances(cfg['size'])
m=[round(v[me][0]-v[them][0],2) for v in ps]
def signed(x):return round(x[me]-x[them],3)
def path(s,uid,t):
    res=[]
    for j in sorted({max(0,t-k) for k in (10,5,3,1,0)}):
        q=ps[j][s][2].get(uid)
        if not q:res.append(dict(j=j));continue
        near=[{'uid':u,'pos':w[0],'cargo':w[1],'dist':dists[q[0]][w[0]]} for u,w in ps[j][1-s][2].items() if w[2]=='COLLECTOR' and w[1]<=q[1]]
        res.append(dict(j=j,pos=q[0],cargo=round(q[1],2),dist_home=min(dists[q[0]][b] for b in ps[j][s][1].values()),nearest_lighter=min(near,key=lambda v:v['dist'],default=None),issued=(f[j+1][s]['action'] or {}).get(uid,'NONE') if j<299 else None))
    return res
def window(left,right):
    cats={k:[round(sum(v[s][c] for v in ledger[left:right]),2) for s in parties] for c,k in enumerate(E.KEYS)}
    cats['tree_net']=[round(sum(cats[k][s] for k in ('tree','plant','seize')),2) for s in (0,1)]
    deposits=[dict(arrival=j+1,seat=s,uid=u,wallet=v,path=path(s,u,j+1)) for j in range(left,right) for s,u,v in banks[j]]
    origins=[{'t':j,'profiles':[E.profile(ps[j],s,dists) for s in parties],'issued':[f[j+1][s]['action'] if j<299 else None for s in parties]} for j in sorted({max(0,left-k) for k in (10,5,3,0)}|{right})]
    return dict(left=left,right=right,margin_before=m[left],margin_after=m[right],change=round(m[right]-m[left],2),accounts=cats,profiles=origins,deposits=deposits)
sim=E.FastCarbon(cfg).load_observation(f[0][0]['observation']);sim.cash=E.CashEvents(sim.cash);tenures=[];cur={};loot=[]
for t in range(299):
    sim.cash.events.clear();sim.step(*[x['action'] or {} for x in f[t+1]])
    post={v[0]:(s,uid) for s in (0,1) for uid,v in ps[t+1][s][3].items()}
    for kind,s,val,*detail in sim.cash.events:
        if kind in ('plant','seize'):
            uid,cell=detail
            owner,tid=post[cell];assert owner==s
            q=dict(start=t+1,seat=s,uid=uid,tree=tid,pos=cell,cost=-val,earned=0,kind=kind)
            tenures.append(q);cur[tid]=q
        elif kind=='tree_absorption':
            tid,cell,_=detail
            assert tid in cur and cur[tid]['seat']==s;cur[tid]['earned']+=val
    for uid,victim,amt,rs,collector in sim.last_collisions:
        if collector==1-victim and amt>0:loot.append(dict(state=t+1,victim=victim,uid=uid,winner=rs,amount=amt))
for q in tenures:
    q['net']=q['earned']-q['cost']
    end=next((j for j in range(q['start']+1,300) if q['tree'] not in ps[j][q['seat']][3]),300)
    q['end_exclusive']=end
    q['exit']='terminal' if end==300 else 'enemy_seize' if q['tree'] in ps[end][1-q['seat']][3] else 'wither'
    q['after_enemy_income']=None
    if q['exit']=='enemy_seize':
        later=next(z for z in tenures if z['tree']==q['tree'] and z['start']==end)
        q['after_enemy_income']=later['earned']
summary=window(0,299)
whole={k:v for k,v in summary.items() if k not in ('profiles','deposits')}
for idx,s in enumerate(parties):
    assert abs(sum(q['net'] for q in tenures if q['seat']==s)-whole['accounts']['tree_net'][idx])<0.025
a10=min(range(290),key=lambda i:m[i+10]-m[i]);peak_gap=max(m);max_draw=max((m[j]-m[i],j,i) for i in range(300) for j in range(i+1))
lastpos=max(i for i,x in enumerate(m) if x>0);lastnonneg=max(i for i,x in enumerate(m) if x>=0)
idx=max(range(299),key=lambda t:m[t]-m[t+1])
nc=card['id'];out=dict(match=nc,source_sha256=hashlib.sha256(a.raw.read_bytes()).hexdigest(),seat=me,official_scores=[rr['scores'][k] for k in ('a','b')],official_margin=m[-1],whole=whole,loot=loot,tree_tenures=tenures,largest_lead=(m.index(peak_gap),peak_gap),largest_deficit=(m.index(min(m)),min(m)),last_lead=(lastpos,m[lastpos]),last_nonnegative=lastnonneg,peak_to_later_trough=max_draw,largest_single_step_loss=window(idx,idx+1),final_loss_of_lead=window(lastnonneg,lastnonneg+1),
         worst10=window(a10,a10+10),last30=window(269,299),last10=window(289,299),every_state_margin=m)
v6=json.loads(gzip.decompress((a.repo/'reports/master3s2_derivative/ladder/observation_nov20/distinct_opponents.json.gz').read_bytes()))['games']
six=next(g for g in v6 if g['opponent']=='Fable5.2-Max-V6')
nine=json.loads(gzip.decompress((a.repo/'reports/master3s2_derivative/ladder/hajimi_v3pro/rematch/hajimi_and_fable_economies.json.gz').read_bytes()))['cc9d17e7f3c77324526c9f2ac5e3e7cc']
z=nine['whole'];out['archived_comparators']={'v6':dict(match=six['match'],seat=six['seat'],score=six['score'],whole=six['whole']),'v9':dict(match=nine['id'],seat=nine['seat'],score=[nine['my_score'],nine['opponent_score']],whole=z)}
def midband(l,r):
    w=window(l,r); summary={}
    for s in parties:
        per_state=[E.profile(v,s,dists) for v in ps[l:r]]
        summary[s]=dict(worker_turns=sum(z['workers'] for z in per_state),collector_turns=sum(z['collectors'] for z in per_state),planter_turns=sum(z['workers']-z['collectors'] for z in per_state),tree_turns=sum(z['trees'] for z in per_state),carried_sum=sum(z['cargo'] for z in per_state),mean_carried=sum(z['cargo'] for z in per_state)/(r-l),mean_safe_close=sum(z['safe_close_load'] for z in per_state)/(r-l))
    return dict(left=l,right=r,accounts=w['accounts'],margin_before=w['margin_before'],margin_after=w['margin_after'],cash_and_workers_from_to=[dict(t=t,states=[E.profile(ps[t],s,dists) for s in parties]) for t in (l,r)],live_summary_by_physical_seat=summary,deposits=w['deposits'])
middle=midband(100,200)
g0=min(range(100,191),key=lambda t:m[t+10]-m[t]);g1=max(range(100,191),key=lambda t:m[t+10]-m[t])
out['midgame']=dict(whole=middle,behind_states_100_through_199=sum(z<0 for z in m[100:200]),ahead_states_100_through_199=sum(z>0 for z in m[100:200]),mean_margin=sum(m[100:200])/100,split=[midband(t,t+10) for t in range(100,200,10)],worst10=window(g0,g0+10),best10=window(g1,g1+10),first_long_drop=window(122,123),turning_back_up=window(106,107),next10_after_long_drop=window(122,132))
with gzip.GzipFile(filename=str(a.out),mode='wb',mtime=0) as fo: fo.write(json.dumps(out,ensure_ascii=False,separators=(',',':')).encode())
print('extrema',out['largest_lead'],out['largest_deficit'],out['last_lead'],out['peak_to_later_trough'],out['largest_single_step_loss']['left']);print('last crossing',out['final_loss_of_lead']['accounts'])
print('worst10',out['worst10']['left'],out['worst10']['right'],[(x['arrival'],x['seat'],x['uid'],round(x['wallet'],2)) for x in out['worst10']['deposits']])
print('worsttree',sorted((q['net'],q['uid'],q['start'],q['end_exclusive'],q['cost'],q['exit'],q['after_enemy_income'],q['pos']) for q in tenures if q['seat']==me)[:10]);print('loot',loot)

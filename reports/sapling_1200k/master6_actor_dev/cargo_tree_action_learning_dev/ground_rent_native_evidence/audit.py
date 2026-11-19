"""Replay every signed planter pair using the same split and a fixed, observable stratification."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ACTOR=HERE.parents[1]
BASE=ACTOR.parent
ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from turning_point_ledger import CashEvents,FastCarbon,write

def read(p):
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def distance(a,b,n):
    dx=abs(a%n-b%n);dy=abs(a//n-b//n)
    return min(dx,n-dx)+min(dy,n-dy)

def context(sim,s,uid):
    o=sim.obs(s); teams=o['players'];f=teams[s][2].get(uid)
    v=dict(step=sim.step_count,cash=[float(k[0]) for k in teams],workers=[k[2] for k in teams],
           trees=[k[3] for k in teams],bases=[k[1] for k in teams])
    if f:
        cell=f[0];nb=sim._nbr4[cell];size=int(sim.config['size'])
        en=[distance(cell,w[0],size) for w in teams[1-s][2].values()]
        tile='empty'
        for i,team in enumerate(teams):
            if cell in team[1].values():tile='base'
            if any(pos==cell for pos,age in team[3].values()):tile='own' if s==i else 'enemy'
        v.update(focal=f,adjacent_presumed_visible=bool(f[2]=='PLANTER'),adjacent=[float(o['carbon'][i]) for i in nb],
                 min_enemy_distance=min(en,default=99),surface=tile)
    return v

def delta(r1,r0,key,s):
    a,b=r1[key],r0[key];return (a[s]-a[1-s])-(b[s]-b[1-s])

def settle(r,signed,arm,snap,cfg):
    a=r['arms'][arm]; counterpart=signed['arms'][a['native_arm']]
    path=ROOT/counterpart['action_file'];rp=ROOT/counterpart['receipt']
    if not (sha(path)==counterpart['action_sha256']==a['archived_tape_sha256'] and sha(rp)==counterpart['receipt_sha256']):
        raise ValueError('archive digest')
    tape=read(path);receipt=read(rp);s=r['seat']; end=a['terminal_observation_t']
    if not (tape[0][s]==r['own_root_joint'][arm] and tape[0][1-s]==signed['root_opponent']
        and receipt['score']==a['official_physical_scores'] and receipt['statuses']==['DONE','DONE']
        and len(tape)-1==counterpart['calls'] and end==r['t']+len(tape)):
        raise ValueError('mismatched signed commands, opponent response or score')
    sim=FastCarbon(cfg);sim.set_state(snap);sim.cash=CashEvents(sim.cash)
    vals={k:[0.,0.] for k in ('ground','mid_ground','tax','mid_tax','plant_cost','seize_cost','own_delivery')}
    paid=None;focal_g=focal_tax=0.
    for joint in tape:
        t=sim.step_count;sim.cash.events.clear();sim.step(*joint)
        for kind,seat,amount,*fields in sim.cash.events:
            if kind=='tree_absorption':
                disk=amount-fields[2];tariff=fields[2]
                vals['ground'][seat]+=disk;vals['tax'][seat]+=tariff
                if 100<=t<200:vals['mid_ground'][seat]+=disk;vals['mid_tax'][seat]+=tariff
                if (seat==s and fields[0]==a['tree_id'] and a['purchase_action_t'] is not None and t>=a['purchase_action_t']
                    and (a['end_first_ownership_observation_t'] is None or t+1<=a['end_first_ownership_observation_t'])):
                    focal_g+=disk;focal_tax+=tariff
            elif kind in ('plant','seize'):
                vals[kind+'_cost'][seat]-=amount
                if t==r['t'] and seat==s and fields[0]==r['uid']: paid={'kind':kind,'amount':-amount}
            elif kind=='own_delivery':vals[kind][seat]+=amount
    if (not np.allclose(sim.cash,a['official_physical_scores'],atol=.011,rtol=0) or sim.step_count!=end or
        abs(a['absorption']-focal_g-focal_tax)>.001):
        raise ValueError('economic or official terminal replay mismatch')
    return dict(**vals,first_tenure_ground=focal_g,first_tenure_tax=focal_tax,first_tenure_cost=a['debit'],
                terminal_score=a['official_physical_scores'],official_margin=a['official_margin'],immediate_payment=paid,
                sequence_sha256=sha(path),receipt_sha256=sha(rp),opponent_calls_after_root=counterpart['calls'])

def classes(r,c):
    q=['all'];mid=100<=r['t']<200
    if not mid:return q
    q.append('midgame_100_199');surface=c['surface'];safe=c['min_enemy_distance']>=2
    if surface=='empty':klass='empty_rich_safe_midgame' if safe and sum(c['adjacent'])>=120 else 'empty_other_midgame'
    elif surface=='enemy':klass='enemy_tree_safe_midgame' if safe else 'enemy_tree_contested_midgame'
    else:klass='other_surface_midgame'
    q.append(klass);return q

def stats(rows):
    by=defaultdict(list)
    for r in rows:by[r['map']].append(r)
    keys=('mid_ground','ground','margin','own_delivery')
    maps={k:{m:float(np.mean([v['change'][m] for v in group])) for m in keys} for k,group in by.items()}
    def signs(field):
        up=lambda row: all(row[k]>1e-7 for k in (field,'margin'))
        dn=lambda row: all(row[k]<-1e-7 for k in (field,'margin'))
        return dict(positive=sum(up(r['change']) for r in rows),negative=sum(dn(r['change']) for r in rows),
                    positive_maps=sum(up(m) for m in maps.values()),negative_maps=sum(dn(m) for m in maps.values()))
    seats={}
    for s in (0,1):
        group=[r for r in rows if r['seat']==s];permap=defaultdict(list)
        for r in group:permap[r['map']].append(r)
        if group:seats[s]={k:float(np.mean([np.mean([v['change'][k] for v in rr]) for rr in permap.values()])) for k in ('mid_ground','margin')}
    return dict(n=len(rows),n_maps=len(maps),by_map=maps,mid_signs=signs('mid_ground'),full_signs=signs('ground'),
                map_mean={k:float(np.mean([m[k] for m in maps.values()])) for k in keys} if maps else {},seat_map_mean=seats,
                immediate_payment=sum(r['arms']['stay']['immediate_payment'] is not None for r in rows))

def run():
    q=read(HERE/'protocol.json')
    registry=ACTOR/'joint_response_critic_dev/signed_roots.json.gz'
    rowsfile=ACTOR/'two_arm_plant_economy_dev/native_eval_rows.json.gz'
    signed=read(registry); primary={r['case']:r for r in signed['rows'] if r['role']=='planter'}
    source=read(rowsfile)
    if len(signed['rows'])!=108 or len(primary)!=len(source) or len(primary)!=66 or set(primary)!={r['case'] for r in source}:
        raise ValueError('population drift')
    output=[]; groups=defaultdict(list)
    for r in source:groups[primary[r['case']]['source'],r['seat']].append(r)
    for (file,s),rows in groups.items():
        path=ROOT/file;raw=read(path);sim=FastCarbon(raw['game_configuration']).reset(seed=raw['seed']);dig=sha(path)
        rows.sort(key=lambda r:r['t'])
        for r in rows:
            a=primary[r['case']];uid=r['uid'];t0=r['t']
            if not(a['source_sha256']==dig==r['source_sha256'] and a['changed_uids']==[uid] and
                a['t']==t0 and a['seat']==s and a['seed']==r['seed'] and a['terrain']==r['map'] and not a['repeat_stream']):
                raise ValueError('pair identity')
            sim=FastCarbon(raw['game_configuration']).reset(seed=raw['seed'])
            history=[]
            while sim.step_count<t0:
                t=sim.step_count; order=raw['actions'][t]
                if t>=t0-10:history.append(dict(public=context(sim,s,uid),archival_joint_order=order))
                sim.step(*order)
            ctx=context(sim,s,uid)
            if not ctx.get('adjacent_presumed_visible'):raise ValueError('missing planter')
            arms={k:settle(r,a,k,sim.get_state(),raw['game_configuration']) for k in ('stay','move')}
            change={k:delta(arms['stay'],arms['move'],k,s) for k in ('ground','mid_ground','tax','own_delivery','plant_cost','seize_cost')}
            change['margin']=arms['stay']['official_margin']-arms['move']['official_margin']
            change['focal_net']=sum(sign*(u['first_tenure_ground']-u['first_tenure_cost']) for sign,u in ((1,arms['stay']),(-1,arms['move'])))
            keys=classes(r,ctx);output.append(dict(case=r['case'],signed_root=a['identity'],seed=r['seed'],map=r['map'],seat=s,t=t0,
                source=file,source_sha256=dig,historical_parent=r['parent_pick'],split=r['historical_use'],uid=uid,
                classes=keys,visible_root=ctx,preceding_ten=history,own_commands=r['own_root_joint'],root_enemy_for_audit=a['root_opponent'],
                arms=arms,change=change))
    out=HERE/'roots.json.gz';write(out,output,True)
    sums={}
    for klass in q['strata']:
        selected=[r for r in output if klass in r['classes']]
        totals={g:stats([r for r in selected if g=='all' or g==r['split']]) for g in ('train','hold','outside','all')}
        verdict=True
        for split in ('train','hold'):
            v=totals[split];m=v['map_mean']
            verdict&=v['n_maps']>=2 and v['mid_signs']['positive_maps']*3>=2*v['n_maps'] and set(v['seat_map_mean'])=={0,1} and all(m.get(k,0)>0 for k in ('mid_ground','margin'))
        verdict&=all(all(a[k]>0 for k in ('mid_ground','margin')) for a in totals['hold']['seat_map_mean'].values())
        sums[klass]=dict(totals=totals,provisional_replication=bool(verdict))
    write(HERE/'counts.json',dict(scope=sha(HERE/'protocol.json'),program=sha(Path(__file__)),registry=sha(registry),inputs=sha(rowsfile),economic_audit=sha(out),strata=sums))
    for k,z in sums.items():print(k,z['provisional_replication'],[(s,v['n'],v['n_maps'],v['mid_signs']['positive_maps'], v['map_mean']) for s,v in z['totals'].items() if s in ('train','hold')],flush=True)
if __name__=='__main__':run()

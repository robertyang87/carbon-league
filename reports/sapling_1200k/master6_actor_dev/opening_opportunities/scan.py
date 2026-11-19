"""Opening-only audit of original Master3s2 vs Sapling. No agent is called."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
from turning_point_ledger import FastCarbon, CashEvents, compact
import tree_win_gate as gate

FROZEN = '4a400100aa53be4c3d7575a3baa4ded32cf60451'
DIRECTIONS = ('attack', 'reverse')
NAMES = ['roots', 'low', 'near_or_low', 'under_two_fees', 'one_two_live', 'bankable', 'scarce',
         'one_two_loaded', 'broad_geometry', 'direct', 'two_only', 'home_geometry', 'low_scarce',
         'low_geometry', 'scarce_geometry', 'joint_direct', 'joint_two_only', 'joint',
         'joint_near_or_low', 'joint_under_two_fees', 'joint_broad_geometry', 'joint_loaded_only_definition',
         'broad_direct_joint','broad_two_only_joint','joint_light_le_30', 'joint_no_tree', 'joint_all_carriers_covered', 'joint_with_recruit_slot']


def sha(data):
    return hashlib.sha256(data).hexdigest()


@lru_cache(None)
def neighbors(p, n):
    y, x = divmod(p, n)
    return (p, ((y-1)%n)*n+x, y*n+(x+1)%n, ((y+1)%n)*n+x, y*n+(x-1)%n)


@lru_cache(None)
def dist(a, b, n):
    ay, ax = divmod(a, n)
    by, bx = divmod(b, n)
    dx, dy = abs(ax-bx), abs(ay-by)
    return min(dx, n-dx) + min(dy, n-dy)


def home_dist(p, bases, n):
    return min(dist(p, b, n) for b in bases)


@lru_cache(None)
def routes(h, v, bases, n):
    """Hypothetical legal movement pairs. Truncate each target route at its first deposit."""
    d = home_dist(v, bases, n)
    if not d:
        return None
    home_steps = [q for q in neighbors(v, n)[1:] if home_dist(q, bases, n) < d]
    hsteps = neighbors(h, n)
    for e in [v] + home_steps:
        if e in hsteps:
            return (1, [h,e], [v,e])
    for e1 in home_steps:
        if e1 in bases:
            continue
        for e2 in neighbors(e1, n)[1:]:
            if home_dist(e2,bases,n) >= d-1:
                continue
            for h1 in hsteps[1:]:  # no wait/harvest in the first transition
                if e2 in neighbors(h1,n):
                    return (2,[h,h1,e2],[v,e1,e2])
    return None


@lru_cache(None)
def open_routes(h,v,bases,n):
    """Existence of simultaneous legal paths, target may also travel away from home."""
    if v in bases:  # remaining positive cargo there is not a future return from the field
        return None
    for e in neighbors(v,n):
        if e in neighbors(h,n):
            return (1,[h,e],[v,e])
    for e1 in neighbors(v,n):
        if e1 in bases:
            continue
        for e2 in neighbors(e1,n):
            for h1 in neighbors(h,n)[1:]:
                if e2 in neighbors(h1,n):
                    return (2,[h,h1,e2],[v,e1,e2])
    return None


def classify(obs, cfg, target):
    """Only public players/configuration are read. Future actions have no access path here."""
    players = obs['players']
    cash, bases_dict, workers, trees = players[target]
    bases = tuple(sorted(bases_dict.values()))
    all_roles = Counter(v[2] for player in players for v in player[2].values())
    fees = [cfg['recCollectorCost'] + cfg['recCollectorIncreaseCost']*all_roles['COLLECTOR'],
            cfg['recPlanterCost'] + cfg['recPlanterIncreaseCost']*all_roles['PLANTER']]
    fee = min(fees)
    n = cfg['size']
    remaining = cfg['episodeSteps']-1-obs['step']
    fleet = {k:v for k,v in workers.items() if v[2]=='COLLECTOR'}
    loaded = {k:v for k,v in fleet.items() if v[1]>cfg['smeltCost'] and bases and home_dist(v[0],bases,n)<=remaining}
    hunters = {k:v for k,v in players[1-target][2].items() if v[2]=='COLLECTOR'}
    witnesses = []
    free_witnesses = []
    covered = set()
    for victim,(v,vg,_) in loaded.items():
        for hunter,(h,hg,_) in hunters.items():
            if hg >= vg:
                continue
            delta = dist(h,v,n)
            if delta>4:
                continue
            free = open_routes(h,v,bases,n)
            if free:
                free_witnesses.append((free[0],hg,hunter,victim,h,vg,v,free[1],free[2]))
            r = routes(h,v,bases,n)
            if r:
                witnesses.append((r[0],hg,hunter,victim,h,vg,v,r[1],r[2]))
                covered.add(victim)
    broad = bool(free_witnesses)
    broad_direct = broad and min(w[0] for w in free_witnesses)==1
    low = cash < fee
    near = cash < 1.5*fee
    twice = cash < 2*fee
    scarce = 1<=len(fleet)<=2 and bool(loaded)
    geom = bool(witnesses)
    direct = geom and min(w[0] for w in witnesses)==1
    joint = low and scarce and geom
    flags = [True,low,near,twice,1<=len(fleet)<=2,bool(loaded),scarce,1<=len(loaded)<=2,broad,direct,
             geom and not direct,geom,low and scarce,low and geom,scarce and geom,
             joint and direct,joint and not direct,joint,near and scarce and geom,twice and scarce and geom,
             low and scarce and broad,low and 1<=len(loaded)<=2 and geom,
             low and scarce and broad_direct,low and scarce and broad and not broad_direct,
             joint and any(w[1]<=30 for w in witnesses),joint and not trees,
             joint and covered==set(loaded),joint and bool(bases) and len(workers)<cfg['workerLimit']]
    assert len(flags)==len(NAMES)
    counters = {k:int(x) for k,x in zip(NAMES, flags)}
    witness = min(witnesses) if geom else None
    return counters, {'cash':cash,'fees':fees,'live_collectors':len(fleet),'loaded_collectors':len(loaded),
               'trees':len(trees),'workers':len(workers),'bases':bases,'visible_witness':witness,
               'covered_loaded_uids':sorted(covered),'open_witness':min(free_witnesses) if broad else None,
               'direct_pairs':[(w[2],w[3]) for w in free_witnesses if w[0]==1] if low and scarce else []}


def future(states, logs, events, seat, t, uid, horizon=5):
    seq=[]
    for k in range(t,min(t+horizon,len(logs))):
        pre=states[k][seat][2].get(uid)
        if pre is None:
            break
        post=states[k+1][seat][2].get(uid)
        deposited=sum(e[2] for e in events[k] if e[0]=='own_delivery' and e[1]==seat and e[3]==uid)
        seq.append({'t':k,'cmd':logs[k][seat].get(uid,'NONE'),'from':pre[0],'to':post[0] if post else None,
                    'cargo_after':round(post[1],4) if post else None,'own_deposit':round(deposited,4)})
    return seq


def direct_replays(sim, cfg, commands, target, pairs):
    """Post-selection verification against the real submitted opponent command; not an opponent model."""
    if not pairs:
        return []
    snapshot=sim.get_state()
    probe=FastCarbon(cfg)
    probe.set_state(snapshot)
    probe.step(*commands)
    alive=set(probe.workers)
    choices=('NONE','UP','RIGHT','DOWN','LEFT')
    hits=[]
    for hunter, victim in sorted(set(pairs)):
        if victim not in alive:
            continue
        origin=sim.workers[hunter].pos
        v=sim.workers[victim]
        actual_target_cmd=commands[target].get(victim,'NONE')
        dest=neighbors(v.pos,cfg['size'])[choices.index(actual_target_cmd)]
        positions=neighbors(origin,cfg['size'])
        if dest not in positions:
            continue
        alt=choices[positions.index(dest)]
        if commands[1-target].get(hunter,'NONE')==alt:
            continue
        modified=[cmd.copy() for cmd in commands]
        if alt=='NONE':
            modified[1-target].pop(hunter,None)
        else:
            modified[1-target][hunter]=alt
        probe.set_state(snapshot)
        probe.step(*modified)
        if victim not in probe.workers and hunter in probe.workers and probe.workers[hunter].pos==dest:
            hits.append({'hunter':hunter,'carrier':victim,'actual_hunter':commands[1-target].get(hunter,'NONE'),
                         'alternative_hunter':alt,'actual_victim':actual_target_cmd,'cell':dest,
                         'victim_pre_cargo':v.carbon,'hunter_post_cargo':probe.workers[hunter].carbon,
                         'actual_post_cash':None,'alternative_post_cash':list(probe.cash)})
    return hits


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--revision', default=FROZEN)
    args=ap.parse_args()
    files=subprocess.check_output(['git','ls-tree','-r','--name-only',args.revision,'reports'],cwd=ROOT,text=True).splitlines()
    files=[ROOT/f for f in files if re.search(r'/raw_\d+_[01]\.json\.gz$',f)]
    source=subprocess.check_output(['git','show',args.revision+':sources/master3s2/master3s2.py'],cwd=ROOT)
    assert hashlib.md5(source).hexdigest()==gate.EXAMINER
    seen, excluded, details, all_examples = {},[],[],[]
    totals={d:Counter() for d in DIRECTIONS}
    groups=defaultdict(lambda:{d:Counter() for d in DIRECTIONS})
    fee_ranges=set()
    public_roots={d:defaultdict(set) for d in DIRECTIONS}
    for path in files:
        folder=path.parent
        rel=str(path.relative_to(BASE))
        raw=json.loads(gzip.decompress(path.read_bytes()))
        ledger=json.loads((folder/'results.json').read_text())
        if folder.name=='tree_exam_it50':
            excluded.append([rel,'pure it50 guard baseline']); continue
        if raw['errors']:
            excluded.append([rel,'execution error',raw['errors'][0]]); continue
        row=next(r for r in ledger['games'] if r['raw_file']==path.name)
        assert ledger['status']=='complete'
        gate.validate_game(row)
        gate.validate_examiner_configuration(raw['configuration'])
        assert not raw.get('audit_messages')
        header=ledger['header']
        assert header['examiner_md5']==gate.EXAMINER
        weight_identity=header.get('weight_sha256') or header.get('component_label')
        gate.verify_raw(folder,row,header['pack_sha256'],weight_identity)
        sim=FastCarbon(raw['game_configuration']).reset(seed=raw['seed'])
        board=gate.map_identity.terrain_sha256(sim.carbon)
        tape_id=sha(compact([board,raw['game_configuration'],raw['actions']]))
        if tape_id in seen:
            excluded.append([rel,'duplicate physical trajectory',seen[tape_id]]);continue
        seen[tape_id]=rel
        sim.cash=CashEvents(sim.cash)
        seat=raw['candidate_seat']
        actions=raw['actions']
        cap=min(len(actions),81)
        states=[]; events=[]; hits={d:Counter() for d in DIRECTIONS}; examples=[]
        for t in range(cap+1):
            obs=sim.obs(seat)
            assert obs['step']==t
            states.append(obs['players'])
            if t<min(76,len(actions)):
                public_id=(board, seat, t, sha(compact([obs['players'],obs['carbon'],obs['trees']])))
                for direction,victim_seat in zip(DIRECTIONS,(1-seat,seat)):
                    flags, info=classify(obs,raw['game_configuration'],victim_seat)
                    hits[direction].update(flags)
                    fee_ranges.add(tuple(info['fees']))
                    for k in ('roots','joint','low_scarce','joint_broad_geometry','joint_no_tree','joint_all_carriers_covered'):
                        if flags[k]:
                            public_roots[direction][k].add(public_id)
                    if flags['scarce_geometry'] or flags['joint_broad_geometry']:
                        direct=direct_replays(sim,raw['game_configuration'],actions[t],victim_seat,info.pop('direct_pairs'))
                        info['fixed_other_commands_direct_kills']=direct
                        examples.append({'direction':direction,'t':t,'target_seat':victim_seat,'joint':bool(flags['joint']),'broad_joint':bool(flags['joint_broad_geometry']),
                                         **info})
            if t<cap:
                sim.cash.events.clear()
                sim.step(*actions[t])
                events.append(sim.cash.events.copy())
        near_rank={}
        for ex in examples:
            k=ex['direction']
            if not ex['visible_witness']:
                continue
            rank=(ex['cash']-min(ex['fees']),ex['t'])
            if k not in near_rank or rank<near_rank[k][0]:
                near_rank[k]=(rank,ex)
        chosen=[x for x in examples if x['broad_joint']]
        for _,ex in near_rank.values():
            if not ex['broad_joint']:
                chosen.append(ex)
        for ex in chosen:
            t=ex['t']; v=ex['target_seat']; w=ex['visible_witness'] or ex['open_witness']
            for alt in ex['fixed_other_commands_direct_kills']:
                alt['actual_post_cash']=[p[0] for p in states[t+1]]
            ex['victim_five']=future(states,actions,events,v,t,w[3])
            ex['victim_pre_five']=[states[k][v][2].get(w[3]) for k in range(max(0,t-5),t)]
            ex['hunter_pre_five']=[states[k][1-v][2].get(w[2]) for k in range(max(0,t-5),t)]
            ex['hunter_five']=future(states,actions,events,1-v,t,w[2])
            ex['previous_target_cash']=[round(states[k][v][0],3) for k in range(max(0,t-5),t+1)]
            ex['strict_joint_or_nearest_by_cash']='strict' if ex['joint'] else ('free_route_only' if ex['broad_joint'] else 'near_miss')
        game={'source':rel,'seed':raw['seed'],'sapling_seat':seat,'physical_tape_sha256':tape_id,
              'raw_sha256':row['raw_sha256'],'terrain_sha256':board,'roots':hits['attack']['roots'],
              'counts':hits,'examples':chosen}
        details.append(game)
        for d in DIRECTIONS:
            totals[d].update(hits[d]);groups[raw['seed']][d].update(hits[d])
        for ex in chosen:
            all_examples.append((game['seed'],seat,rel,ex))
    summary={'revision':args.revision,'examiner_md5':gate.EXAMINER,'full_raw_files':len(files),
             'eligible_distinct_tapes':len(details),'distinct_maps':len(groups),'excluded':excluded,
             'config_fees_encountered':sorted(fee_ranges),'root_counts':totals,'map_root_counts':groups}
    summary['distinct_public_roots']={d:{k:len(v) for k,v in vals.items()} for d,vals in public_roots.items()}
    summary['equal_map_rates']={d:{k:sum(g[d][k]/g[d]['roots'] for g in groups.values())/len(groups)
                                  for k in NAMES} for d in DIRECTIONS}
    for d in DIRECTIONS:
      for tag,flag in [('strict_five_actual','joint'),('all_legal_five_actual','broad_joint')]:
        exs=[ex for s,i,r,ex in all_examples if ex['direction']==d and ex[flag]]
        summary.setdefault(tag,{})[d]={'one_selected_witness_per_root':len(exs),
            'roots_with_at_least_one_replayed_direct_kill':sum(bool(ex['fixed_other_commands_direct_kills']) for ex in exs),
            'victim_uid_made_own_deposit':sum(any(x['own_deposit']>0 for x in ex['victim_five']) for ex in exs),
            'victim_uid_disappeared':sum(any(x['to'] is None for x in ex['victim_five']) for ex in exs),
            'victim_uid_received_own_bank_total_across_overlapping_windows':round(sum(x['own_deposit'] for ex in exs for x in ex['victim_five']),3)}
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (HERE/'games.jsonl.gz').write_bytes(gzip.compress(b''.join(compact(g)+b'\n' for g in details),mtime=0))
    print(json.dumps({k:summary[k] for k in ('eligible_distinct_tapes','distinct_maps','root_counts','strict_five_actual')}))

if __name__=='__main__':
    main()

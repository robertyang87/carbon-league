"""Five-step collector ambushes on archived genuine Sapling–Master3s2 games."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import tree_audited_game as arena
from interception_cash import collision_events
from interception_fit import intent
from tree_win_gate import validate_game, verify_raw

HERE = Path(__file__).resolve().parent
DIST = intent.DIST.tolist()
DEST = intent.DEST.tolist()
MOVES = tuple(intent.MOVES)
H = 5
RUNS = ("tree_exam_soft_it50_v5", "tree_exam_interception", "tree_exam_tree_raid")


def pairs(obs):
    me = obs["player"]
    own, foe = obs["players"][me], obs["players"][1-me]
    for wid, (wp, wc, wt) in sorted(own[2].items()):
        if wt != "COLLECTOR" or wc > 15:
            continue
        for uid, (ep, ec, et) in sorted(foe[2].items()):
            if et != "COLLECTOR" or ec < 50 or not 3 <= DIST[wp][ep] <= 10:
                continue
            if not 2 <= min(DIST[ep][b] for b in foe[1].values()) <= 9:
                continue
            yield wid, wp, uid, ep, ec


def public_crossings(pos, homes, heading):
    """Returning by shortest routes; weight nearer bases and persistence. No future commands."""
    home_weights = [0.4 ** (DIST[pos][b]-min(DIST[pos][v] for v in homes)) for b in homes]
    result = defaultdict(float)
    for base, home_weight in zip(homes, home_weights):
        paths = {(pos, heading): home_weight / sum(home_weights)}
        for depth in range(1,H+1):
            nxt = defaultdict(float)
            for (p, previous), mass in paths.items():
                dirs = [m for m in range(1,5) if DIST[DEST[p][m]][base] < DIST[p][base]]
                weights = [2 if m == previous else 1 for m in dirs]
                for m, weight in zip(dirs, weights):
                    dest = DEST[p][m]
                    if dest in homes:
                        continue
                    prob = mass * weight / sum(weights)
                    nxt[dest,m] += prob
                    if depth >= 3:
                        result[depth,dest] += prob
            paths = nxt
    return result


def approach(origin, target, ticks, bans, first_bans, root_command):
    """Commit to a short crossing route, staying only at rendezvous if early."""
    if DIST[origin][target] > ticks:
        return None
    pos = origin
    route = []
    for turn in range(ticks):
        if pos == target:
            choices = [0]
        else:
            choices = sorted((m for m in range(1,5) if DIST[DEST[pos][m]][target] < DIST[pos][target]),
                             key=lambda m: (turn == 0 and MOVES[m] != (root_command or 'NONE'),m))
        m = next((m for m in choices if DEST[pos][m] not in bans and
                  (turn > 0 or DEST[pos][m] not in first_bans)), None)
        if m is None:
            return None
        route.append(m)
        pos = DEST[pos][m]
    return route if route[0] and pos == target else None


def choose(obs, previous, own_orders, future=None, forecaster=None):
    """Only the oracle variant is passed realized future states."""
    me = obs["player"]
    own, foe = obs["players"][me], obs["players"][1-me]
    homes = set(foe[1].values())
    booked = Counter(DEST[p][intent.MOVE_ID[own_orders.get(w) or "NONE"]]
                     for w,(p,_,_) in own[2].items())
    recruits = {p for b,p in own[1].items() if own_orders.get(b) in ('RECCOLLECTOR','RECPLANTER')}
    prediction = {}
    best = None
    rank = None
    for wid, wp, uid, ep, ec in pairs(obs):
        if uid not in prediction:
            if forecaster is not None:
                prediction[uid] = forecaster.crossings(obs, previous, uid)
            elif future is None:
                past = previous.get(uid)
                heading = DEST[past[0]].index(ep) if past is not None and ep in DEST[past[0]] else 0
                prediction[uid] = public_crossings(ep, sorted(homes), heading)
            else:
                prediction[uid] = {}
                for d,step in enumerate(future,1):
                    w = step['workers'].get(uid)
                    if not w or w['pos'] in homes or w['carbon'] < 50:
                        break
                    if d >= 3:
                        prediction[uid][d,w['pos']] = 1.
        original = own_orders.get(wid) or 'NONE'
        own_dest = DEST[wp][intent.MOVE_ID[original]]
        banned = recruits | {p for p,n in booked.items() if n > int(p == own_dest)}
        for (depth, destination), prob in prediction[uid].items():
            if forecaster is not None and (prob < forecaster.min_probability or prob*ec < forecaster.min_expected_cargo):
                continue
            distance = DIST[wp][destination]
            if distance < 2 or distance > depth:
                continue
            route = approach(wp, destination, depth, homes, banned, original)
            if not route or MOVES[route[0]] == original:
                continue
            score = prob * min(ec,700) / (1+.20*(depth-3)+.10*(depth-distance))
            key = (score,-depth,-distance,wid,uid,-destination)
            if rank is None or key > rank:
                rank = key
                best = dict(attacker=wid, target=uid, origin=wp, target_origin=ep, target_cargo=ec,
                            depth=depth,destination=destination,moves=[MOVES[m] for m in route],
                            forecast_probability=prob)
    return best


def branch(sim, states, trail, me, plan, apply=True, controller=None):
    sim.set_state(states[0])
    totals = Counter()
    tagged = Counter()
    banked = 0.
    caught = False
    for d, orders in enumerate(trail):
        before = sim.get_state()
        joint = [cmd.copy() for cmd in orders]
        if controller is not None:
            joint[me] = controller(sim.obs(me),joint[me])
        elif apply and not caught and d < plan['depth'] and plan['attacker'] in sim.workers:
            joint[me][plan['attacker']] = plan['moves'][d] if plan['moves'][d]!='NONE' else None
        for p,cmd in enumerate(joint):
            available = set(sim.player_workers[p]) | {b for b,_ in sim.bases[p]}
            vanished = set(cmd) - available
            totals['missing_command_ids'] += len(vanished)
            if not caught:
                totals['missing_before_first_target_capture'] += len(vanished)
            for uid in vanished:
                del cmd[uid]
        totals['roster_divergence_steps'] += before['workers'].keys() != states[d]['workers'].keys()
        sim.step(*joint)
        events = collision_events(before,sim,joint)
        for p in (0,1):
            for w in sim.workers.values():
                if not w.is_collector or sim.base_at.get(w.pos,(None,))[0] != p:
                    continue
                original = before['workers'].get(w.id)
                direct = original['carbon'] if original else 0.
                additions = sum(mass for victim,(owner,credit,mass,route) in events.items()
                                if route == ('cash',p) and credit == w.player
                                and DEST[before['workers'][victim]['pos']][intent.MOVE_ID[joint[owner].get(victim) or 'NONE']] == w.pos)
                totals['own_worker_deposits' if p == me else 'enemy_worker_deposits'] += direct+additions
        for uid,(owner,credit,mass,route) in events.items():
            if owner == me or credit != me:
                continue
            if route[0] == 'worker' or route == ('cash',me):
                totals['enemy_cargo_acquired'] += mass
            if uid == plan['target'] and (route[0] == 'worker' or route == ('cash',me)):
                totals['target_acquired_by_any_ally'] += mass
            if uid == plan['target'] and plan['attacker'] in sim.workers:
                w = sim.workers[plan['attacker']]
                target_pos = before['workers'][uid]['pos']
                target_move = intent.MOVE_ID[joint[1-me].get(uid) or 'NONE']
                if w.pos == DEST[target_pos][target_move] and (route == ('worker',w.id) or route == ('cash',me)):
                    caught = True
                    totals['target_captured'] += mass
                    if route[0] == 'worker':
                        tagged[w.id] += mass
                    else:
                        banked += mass
        surviving = Counter()
        for uid,mass in tagged.items():
            route = events.get(uid,(0,0,0,None))[3]
            if route is None and uid in sim.workers:
                w = sim.workers[uid]
                route = ('cash',sim.base_at[w.pos][0]) if w.pos in sim.base_at else ('worker',uid)
            if route == ('cash',me):
                banked += mass
            elif route and route[0] == 'worker' and sim.workers[route[1]].player == me:
                surviving[route[1]] += mass
        tagged = surviving
        if d < len(trail)-1 and any(arena.exam._extinct(sim.get_state(), p) for p in (0,1)):
            totals['early_terminal'] = 1
            break
    totals.update(own_cash=sim.cash[me],enemy_cash=sim.cash[1-me],
                  own_cargo=sum(w.carbon for w in sim.workers.values() if w.player==me),
                  enemy_cargo=sum(w.carbon for w in sim.workers.values() if w.player!=me),
                  target_cargo_still_carried=sum(tagged.values()),target_cargo_banked=banked)
    return totals


def evaluate_roots(sim,states,observations,actions,me,roots,modes=('public','hindsight'),forecaster=None):
    results = {}
    for mode in modes:
        triggered = []
        for tick in roots:
            trail = actions[tick:tick+H]
            segment = states[tick:tick+H+1]
            previous = observations[tick-1]['players'][1-me][2] if tick else {}
            plan = choose(observations[tick],previous,trail[0][me],segment[1:] if mode=='hindsight' else None,forecaster=forecaster)
            if not plan:
                continue
            base = branch(sim,segment,trail,me,plan,apply=False)
            if sim.get_state() != segment[-1]:
                raise ValueError('branch control diverged')
            alternative = branch(sim,segment,trail,me,plan)
            delta = {k:round(alternative[k]-base[k],4) for k in
                     ('enemy_cargo_acquired','own_worker_deposits','enemy_worker_deposits','own_cash','enemy_cash','own_cargo','enemy_cargo')}
            real = segment[plan['depth']]['workers'].get(plan['target'])
            on_target = real is not None and real['pos'] == plan['destination'] and real['carbon'] >= 50
            triggered.append(dict(tick=tick,forecast_crossing_observed=on_target,parent_target_receipt=round(base['target_acquired_by_any_ally'],4),**plan,**{k:round(alternative[k],4) for k in
                                   ('target_captured','target_cargo_still_carried','target_cargo_banked',
                                    'missing_command_ids','missing_before_first_target_capture','roster_divergence_steps','early_terminal')},delta=delta))
        results[mode] = triggered
    return results


def run_game(run, row, header, forecaster=None):
    verify_raw(run,row,header['pack_sha256'],header['weight_sha256'])
    validate_game(row)
    raw = json.loads(gzip.decompress((run/row['raw_file']).read_bytes()))
    me = row['candidate_seat']
    sim = arena.exam.FastCarbon(raw['game_configuration']).reset(seed=row['seed'])
    if sim.config['smeltCost'] or sim.config['moveCost']:
        raise ValueError('zero-cost transfer required')
    states, observations = [sim.get_state()], [sim.obs(me)]
    for joint in raw['actions']:
        sim.step(*joint)
        states.append(sim.get_state())
        observations.append(sim.obs(me))
    if arena.prior.hashed(states[-1]) != row['final_state_sha256']:
        raise ValueError('terminal hash mismatch')
    eligible = len(raw['actions'])-H+1
    modes = ('public','hindsight') if forecaster is None else ('learned',)
    results = evaluate_roots(sim,states,observations,raw['actions'],me,range(eligible),modes,forecaster)
    return dict(run=run.name,seed=row['seed'],seat=me,eligible_roots=eligible,
                raw_sha256=row['raw_sha256'],final_state_sha256=row['final_state_sha256'],**results)


def write_archive(path, result):
    payload=(json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
    path.write_bytes(gzip.compress(payload,mtime=0) if path.suffix=='.gz' else payload)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',type=Path,required=True)
    args = cli.parse_args()
    games=[]
    for name in RUNS:
        run = HERE/name
        ledger=json.loads((run/'results.json').read_bytes())
        for row in ledger['games']:
            game=run_game(run,row,ledger['header'])
            games.append(game)
            print(name,game['seed'],game['seat'],[(m,len(game[m]),sum(x['target_captured']>0 for x in game[m])) for m in ('public','hindsight')],flush=True)
    summary={}
    for mode in ('public','hindsight'):
        rows=[r for g in games for r in g[mode]]
        summary[mode]={'all_roots':sum(g['eligible_roots'] for g in games),'attempted':len(rows),
                       'caught_target_roots':sum(r['target_captured']>0 for r in rows),
                       'caught_maps':len({g['seed'] for g in games if any(r['target_captured']>0 for r in g[mode])}),
                       'missing_future_commands':sum(r['missing_command_ids']>0 for r in rows),
                       'changed_roster':sum(r['roster_divergence_steps']>0 for r in rows),
                       'premature_terminal':sum(r['early_terminal']>0 for r in rows),
                       'target_cargo':{k:round(sum(r[k] for r in rows),4) for k in ('target_captured','target_cargo_banked','target_cargo_still_carried')},
                       'delta_on_all_roots':{k:round(sum(r['delta'][k] for r in rows),4) for k in ('enemy_cargo_acquired','own_worker_deposits','enemy_worker_deposits','own_cash','enemy_cash','own_cargo','enemy_cargo')}}
    result=dict(horizon=H,scope='Overlapping local interventions; unchanged roots have zero delta. Economic sums are branch totals, never a cumulative game gain. Changed states can change actual policies even when all remaining recorded commands refer to live actors.',games=games,summary=summary)
    write_archive(args.out,result)
    print(json.dumps(summary,indent=2))

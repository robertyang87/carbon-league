"""Reconcile a two-order tree takeover and a visible reply against one archived public game."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / '.git').exists())
sys.path[:0] = [str(ROOT / 'reports/sapling_1200k'), str(ROOT / 'code/env_src/carbon_challenge-master')]
from turning_point_ledger import CashEvents, FastCarbon, distances
from zerosum_env.envs.carbon import idgen
from zerosum_env.envs.carbon.helpers import Board

T = 46
FOCUS = 'player-0-worker-4'
DEFENDER = 'player-1-worker-9'
TREE = 'player-1-tree-4'
CELL = 15
SOURCE = HERE.parents[1] / 'replays/d43d676041c83c5d9380f9daf394d2ce-1.json.gz'
SCENARIOS = {'observed': ({}, 299),
             'right_stay': ({T: [(0, FOCUS, 'RIGHT')], T+1: [(0, FOCUS, 'NONE')]}, T+10),
             'stay_on_tree_29': ({T: [(0, FOCUS, 'NONE')]}, T+2),
             'right_stay_with_defender': ({T: [(0, FOCUS, 'RIGHT')], T+1: [(0, FOCUS, 'NONE'), (1, DEFENDER, 'RIGHT')]}, T+2),
             'original_with_defender': ({T+1: [(1, DEFENDER, 'RIGHT')]}, T+2)}


def check(ok, message):
    if not ok:
        raise ValueError(message)


def owner(players):
    return next((s for s in (0, 1) if TREE in players[s][3]), None)


def legal_root(view, cfg):
    players = view['players']
    own, other = players
    dist = distances(cfg['size'])
    check(own[2][FOCUS] == [29, 0, 'PLANTER'] and other[3][TREE] == [15, 9], 'Different visible root')
    near = [{uid: (w[0], w[2], dist[w[0]][CELL]) for uid, w in seat[2].items() if dist[w[0]][CELL] <= 3}
            for seat in players]
    return dict(step=T, player=0, wallet=own[0], unit=FOCUS, position=29, role='PLANTER', cargo=0,
                current_cell_enemy_tree=next({k: v} for k, v in other[3].items() if v[0] == 29),
                target=TREE, target_cell=CELL, target_age=other[3][TREE][1], target_owner=1,
                toroidal_steps_to_target=dist[29][CELL], nearest_own_base_steps=min(dist[CELL][v] for v in own[1].values()),
                nearby_public_workers_by_seat=near, target_public_last_absorption=view['trees'][TREE][1],
                seize_cost=cfg['seizeCost'], tree_lifespan=cfg['treeLifespan'], hire_planter_price=cfg['recPlanterCost'],
                observations_used=['step', 'players', 'trees'], own_base_orders_recorded_on_entry={})


def run(frames, cfg, name, overrides, stop):
    idgen.reset()
    initial = frames[0][0]['observation']
    state = copy.deepcopy(initial)
    state['carbon'] = state['full_carbon']
    sim = FastCarbon(cfg).load_observation(initial)
    sim.cash = CashEvents(sim.cash)
    snapshots = []
    owners = []
    target_abs = [0., 0.]
    target_cost = [0., 0.]
    paid_back = None
    matched_recorded = 0
    all_ignored = []
    for step in range(stop):
        joints = [dict(side['action'] or {}) for side in frames[step+1]]
        for seat, uid, action in overrides.get(step, []):
            check(uid in state['players'][seat][2], 'Missing counterfactual actor')
            if action == 'NONE':
                joints[seat].pop(uid, None)
            else:
                joints[seat][uid] = action
        pre = state['players']
        ignored = [(s, u) for s in (0, 1) for u in joints[s] if u not in pre[s][2] and u not in pre[s][1]]
        if ignored and step >= T:
            all_ignored.append(dict(step=step, commands=ignored))
        before_owner = owner(pre)
        sim.cash.events.clear()
        sim.step(*joints)
        state = Board(state, cfg, joints).next().observation
        from_fast = sim.obs(0)
        check(from_fast['players'] == state['players'] and from_fast['trees'] == state['trees'] and
              list(sim.carbon) == state['carbon'], f'{name} engine mismatch at turn {step}')
        if name == 'observed' or step < T:
            original = frames[step+1][0]['observation']
            check(state['players'] == original['players'] and state['trees'] == original['trees'] and
                  state['carbon'] == original['full_carbon'], f'{name} archive mismatch at turn {step}')
            matched_recorded += 1
        if step < T:
            continue
        pl = state['players']
        new_owner = owner(pl)
        if new_owner != before_owner:
            owners.append(dict(action_step=step, before=before_owner, after=new_owner))
        receipt = [0., 0.]
        costs = [0., 0.]
        for origin, s, amount, *info in sim.cash.events:
            if origin == 'tree_absorption' and info[0] == TREE:
                receipt[s] += amount
            if origin == 'seize' and info[1] == CELL and before_owner is not None and new_owner == s and before_owner != s:
                costs[s] -= amount
        for s in (0, 1):
            target_abs[s] += receipt[s]
            target_cost[s] += costs[s]
        if paid_back is None and target_cost[0] > 0 and target_abs[0] >= target_cost[0]:
            paid_back = step + 1
        if step < T+10 or before_owner != new_owner or costs[0] or costs[1]:
            snapshots.append(dict(action_step=step, state=step+1, wallet=[x[0] for x in pl],
                                  margin=round(pl[0][0]-pl[1][0],2), target_owner=new_owner,
                                  target_credit=receipt, target_purchase=costs,
                                  cumulative_target_credit=target_abs.copy(), cumulative_target_cost=target_cost.copy(),
                                  focus=pl[0][2].get(FOCUS), defender=pl[1][2].get(DEFENDER),
                                  focus_order=joints[0].get(FOCUS, 'NONE'), defender_order=joints[1].get(DEFENDER, 'NONE'),
                                  collisions=list(sim.last_collisions)))
    return dict(scenario=name, archived_transitions_reproduced=matched_recorded, double_engine_transitions=stop,
                first_target_payback_state=paid_back, ownership_changes_after_root=owners,
                target_credits_by_side_since_root=target_abs, target_cost_by_side_since_root=target_cost,
                snapshots=snapshots, ignored_orders_after_divergence=all_ignored)


def main():
    raw = SOURCE.read_bytes()
    game = json.loads(gzip.decompress(raw))
    f, cfg = game['steps'], game['configuration']
    v = f[T][0]['observation']
    before = legal_root({k: v[k] for k in ('step', 'players', 'trees')}, cfg)
    actual = f[T+1][0]['action'] or {}
    check(not (set(actual) & set(v['players'][0][1])) and actual[FOCUS] == 'DOWN', 'Different factual budget or order')
    results = {k: run(f, cfg, k, changes, end) for k,(changes,end) in SCENARIOS.items()}
    expected = results['observed']['snapshots']
    series = []
    for i, alt in enumerate(results['right_stay']['snapshots']):
        fact = expected[i]
        series.append(dict(state=alt['state'], actual_margin=fact['margin'], conditional_margin=alt['margin'],
                           cash_margin_shift=round(alt['margin']-fact['margin'],2),
                           own_wallet_shift=round(alt['wallet'][0]-fact['wallet'][0],2),
                           opponent_wallet_shift=round(alt['wallet'][1]-fact['wallet'][1],2)))
    evidence = dict(schema=1, match=SOURCE.name, sha256=hashlib.sha256(raw).hexdigest(), official_ending=game['rewards'],
                    selection='Single exposed observational root chosen after inspecting the early loss of tree ownership. Conditions in root use actor-visible fields alone; opponents and archived orders after the root are used only for conditional outcome checks.',
                    root=before, arms=results, paired_wallets=series)
    dest = HERE / 'tree_response.json'
    dest.write_text(json.dumps(evidence, sort_keys=True, indent=2, allow_nan=False)+'\n')
    print('actors', before['nearby_public_workers_by_seat'], 'wallet', before['wallet'])
    print('paired',series)
    print('recoup',results['right_stay']['first_target_payback_state'], 'changes',results['observed']['ownership_changes_after_root'])
    for k,b in results.items():
        s=b['snapshots'][1]; print(k,s['state'],s['focus'],s['defender'],s['target_owner'],s['margin'],b['target_credits_by_side_since_root'],b['ignored_orders_after_divergence'][:2])


if __name__ == '__main__':
    main()

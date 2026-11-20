"""Offline targets for a planter's first continuous ownership interval after each actual acquisition."""
from collections import Counter
import importlib.util
from pathlib import Path


def _prior_module():
    path = Path(__file__).resolve().parents[1] / 'ladder/tree_payback/labels.py'
    spec = importlib.util.spec_from_file_location('continuous_tree_ledger', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_prior = _prior_module()
_econ = _prior.economy
check = _prior.require


def labels(replay, seat):
    """Map (decision t, acting planter ID) to realized own cash, including the purchase debit.

    Action t changes state t into state t+1. The incoming owner receives that turn's tree
    absorption. A collision earlier on the same turn pays the outgoing owner. A subsequent
    recapture is a separate target. These fields depend on future observations and belong
    exclusively on the target side of training. Failed attempts and other actions have no row.
    """
    check(seat in (0, 1), 'Unknown physical seat')
    data = _prior.load_replay(replay)
    frames, cfg = data['steps'], data['configuration']
    check(len(frames) >= 2 and data['statuses'] == ['DONE', 'DONE'], 'Expected a completed ladder replay')
    check(frames[0][0]['observation']['step'] == 0 and all(s['status'] == 'DONE' for s in frames[-1]),
          'Incomplete opening or terminal observations')
    # Reuse the existing validator of both balances and every worker and tree at every transition.
    ledger, _, _, _ = _econ.reconstruct(frames, cfg)
    sim = _econ.FastCarbon(cfg).load_observation(frames[0][0]['observation'])
    sim.cash = _econ.CashEvents(sim.cash)
    output = {}
    active = {}
    for t, after in enumerate(frames[1:]):
        before = frames[t][0]['observation']
        obs = after[0]['observation']
        check(obs['step'] == t+1, 'Skipped or duplicated frame')
        old_owners = {uid:s for s,p in enumerate(before['players']) for uid in p[3]}
        new_trees = {uid:(s,tree[0]) for s,p in enumerate(obs['players']) for uid,tree in p[3].items()}
        orders = [s['action'] or {} for s in after]
        sim.cash.events.clear()
        sim.step(*orders)
        payments = {}
        per_tree_absorption = Counter()
        for kind, owner, value, *detail in sim.cash.events:
            if kind in ('plant','seize') and owner == seat:
                worker, cell = detail
                check((kind,cell) not in payments, 'Duplicate acquisition debit')
                payments[kind,cell] = (worker,-value)
            if kind == 'tree_collision_cash' and owner == seat:
                uid = detail[2]
                check(old_owners.get(uid) == owner, 'Collision paid to an unexpected pre-transfer owner')
                if uid in active:
                    active[uid]['collision'] += value
            if kind == 'tree_absorption':
                uid, cell, tax = detail
                check(new_trees[uid] == (owner,cell), 'Tree absorption attributed to the wrong owner or cell')
                per_tree_absorption[uid] += value
        for uid in list(active):
            if uid not in new_trees or new_trees[uid][0] != seat:
                active.pop(uid).update(end_t=t,exit='seized' if uid in new_trees else 'withered')
        for uid,(owner,cell) in new_trees.items():
            check(abs(per_tree_absorption[uid]-obs['trees'][uid][1])<1e-6, 'Tree receipt differs from official observation')
            if owner!=seat or old_owners.get(uid)==seat:
                continue
            kind='seize' if uid in old_owners else 'plant'
            planter=obs['trees'][uid][0]
            check(planter in before['players'][seat][2] and before['players'][seat][2][planter][::2]==[cell,'PLANTER'],
                  'Purchase not attributable to a pre-action planter on that cell')
            check(orders[seat].get(planter) in (None,'NONE'), 'Acquisition while moving')
            payment=payments.pop((kind,cell),None)
            if payment is None:
                # CashEvents suppresses actual zero writes; an observed ownership transition
                # and the official post-action tree actor still identify free transactions.
                free_cfg='seizeCost' if kind=='seize' else 'plantCost'
                check(cfg[free_cfg]==0, 'Acquisition has no corresponding debit')
                cost=0.
            else:
                actor,cost=payment
                check(actor==planter and cost>=0, 'Debit actor or sign inconsistent with tree event')
            check(uid not in active and (t,planter) not in output, 'Duplicate tenure or planter action')
            r=dict(kind=kind,t=t,seat=seat,planter=planter,tree=uid,cell=cell,cost=float(cost),
                   ground=0.,enemy_tax=0.,collision=0.)
            output[t,planter]=active[uid]=r
        check(not payments, 'Purchase debit without official owner change')
        for kind, owner, value, *detail in sim.cash.events:
            if kind=='tree_absorption' and owner==seat:
                uid, _, tax=detail
                if uid in active:
                    active[uid]['ground']+=value-tax
                    active[uid]['enemy_tax']+=tax
    for r in active.values():
        r.update(end_t=len(frames)-2,exit='horizon' if len(frames)==cfg['episodeSteps'] else 'early_terminal')
    for r in output.values():
        r['gross']=r['ground']+r['enemy_tax']+r['collision']
        r['net']=r['gross']-r['cost']
    for idx,kind in ((5,'plant'),(6,'seize')):
        check(abs(sum(r['cost'] for r in output.values() if r['kind']==kind)+sum(row[seat][idx] for row in ledger))<1e-5,
              f'Summed {kind} charges differ from the validated cash ledger')
    return output

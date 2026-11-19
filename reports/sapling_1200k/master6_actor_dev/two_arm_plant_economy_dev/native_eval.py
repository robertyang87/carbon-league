"""Re-score archived real *paired* response tapes at preselected roots, conditional on each joint order."""
import argparse
from collections import Counter, defaultdict, deque
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parent
SAP = ACTOR.parent
ROOT = SAP.parents[1]
sys.path[:0] = [str(SAP), str(ACTOR / 'lot_settlement_dev'), str(SAP / 'root_recovery_branches')]
from lots import Board, CashEvents, FastCarbon, idgen
from branches import clean, destination, utility
import economic

NATIVE = ACTOR / 'tree_tenure_dev/native.json.gz'
PUBLIC = ('player', 'step', 'players', 'trees', 'carbon')
PARENT = '9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0'
EXAMINER = ROOT / 'sources/master3s2/master3s2.py'
REL = lambda path: str(path.relative_to(ROOT))


def load(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return json.load(stream)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def check(condition, issue):
    if not condition:
        raise ValueError(issue)


def write(path, val):
    data = json.dumps(val, indent=2, ensure_ascii=False, allow_nan=False).encode() + b'\n'
    if path.suffix == '.gz':
        data = gzip.compress(data, compresslevel=6, mtime=0)
    new = path.with_name(path.name + '.new')
    new.write_bytes(data)
    new.replace(path)


def inventory():
    hist = load(NATIVE)['rows']
    check(len(hist) == 64, 'historical cohort changed')
    seen = set()
    group = defaultdict(list)
    for row in hist:
        land = row['map']
        key = (land, row['seat'], row['t'], row['uid'])
        check(key not in seen, key)
        seen.add(key)
        group[(row['split'], land, row['seed'])].append({
            'seat': row['seat'], 't': row['t'], 'uid': row['uid'], 'case': row['case']})
    lands = [{'map_sha256': land, 'seed': seed, 'historical_use': split, 'roots': roots}
             for (split, land, seed), roots in sorted(group.items(), key=lambda x: x[0][2])]
    check(len(lands) == 16 and all(len(x['roots']) == 4 for x in lands), 'denominator changed')
    extras = []
    same = []
    scope = Counter()
    scope_arms = defaultdict(Counter)
    for manifest in sorted(SAP.rglob('live_protocol.json')):
        doc = load(manifest)
        family = doc.get('family')
        scope[family] += 1
        proofs = [manifest.parent / f'live_{arm}.json' for arm in ('parent','alternative')]
        parent_hashes = sorted({load(p)['pack_sha256'] for p in proofs if p.is_file()})
        scope_arms[family][('complete_pair' if all(p.is_file() for p in proofs) else 'incomplete')+
                           '/' + ','.join(parent_hashes)] += 1
        if family not in ('two_turn_planter_exercise', 'frozen_tree_pay_repeatability'):
            continue
        raw = load(ROOT / doc['record']['source'])
        card = {'case': REL(manifest.parent), 'seed': doc['seed'], 'seat': doc['seat'], 't': doc['root_t'],
                'uid': doc['root']['worker'], 'map': raw['initial_terrain_u8_sha256'],
                'pack': raw['pack_sha256']}
        check(card['pack'] == PARENT, 'mixed native actor')
        if family == 'two_turn_planter_exercise':
            check(card['map'] not in {x['map_sha256'] for x in lands}, 'external maps overlap')
            extras.append(card)
        else:
            check((card['map'], card['seat'], card['t'], card['uid']) in seen, 'repeat identity')
            same.append(card)
    check(len(extras) == 2 and len(same) == 4, 'external/repeat inventory changed')
    return hist, extras, {
        'source_native': REL(NATIVE), 'source_native_sha256': sha(NATIVE), 'maps': lands,
        'population': {'previously_used_training_roots': 48, 'previously_published_historical_hold_roots': 16,
                       'actually_fresh_blind_native_pairs': 0, 'already_published_additional_other_selector_pairs': 2},
        'alternative_stream_repeats_same_old_roots': same,
        'outside_original_64_different_terrain_already_published': extras,
        'response_manifest_counts_under_sapling_1200k': dict(scope),
        'response_receipt_counts_by_family_and_frozen_actor_identity':scope_arms,
        'scope': 'Checked archived exact-v1 responding stay/move pairs under reports/sapling_1200k. Other interventions and other actor identities omitted. Unrelated active full-policy matches excluded.',
        'reusable': ['Root public observation and last three masked historical views from source tape.',
                     'Both complete own joint orders, common recorded simultaneous opponent command and separate genuine later opponent responses.',
                     'Terminal official physical scores and root worker survival.',
                     'Original labelled root already-charged first tenure. A new focal purchase at root+1, including a moving arm arriving and then buying there, requires re-reading both tapes.'],
        'prior_methods': {
            'tree_response_training/planter_survival': 'Both destinations, public reach geometry and next-frame death; trained on batch01, validation batch02, veto replication batch03, public-v1 replay also uses batch04. Does not jointly score first-tenure net debt/income for two destinations.',
            'tree_tenure_dev': 'Pooled rooting, charges and net receipts for immediate stay exercise; the four named historical held maps have already had endpoint results published.',
            'planter_threat_transfer_dev': 'Prospective enemy one-step reach at a pay root and fixed recorded enemy one-step forks. No separate responding terminal rollout at these roots.',
            'planter_option': 'Two previously published roots from another pay screen on two older maps are a separate sensitivity denominator.'}}


def check_arms(case, raw, s, t, uid, examiner_sha, record_sha):
    proto = case / 'live_protocol.json'
    shared = case / 'live_shared_init.json'
    arms = {}
    for native in ('parent', 'alternative'):
        p = case / f'live_{native}.json'
        rec = load(p)
        path = case / rec['action_tape']
        tape = load(path)
        check(sha(proto) == rec['protocol_sha256'] and sha(shared) == rec['shared_init_sha256']
              and rec['pack_sha256'] == PARENT and rec['source_sha256'] == examiner_sha
              and rec['original_trajectory_sha256'] == record_sha and rec['action_sha256'] == sha(path)
              and rec['seed'] == raw['seed'] and rec['seat'] == s and rec['arm'] == native
              and rec['search_calls'] == len(tape)-1 and rec['terminal_t'] == t+len(tape),
              f'archival identity: {p}')
        pair = tape[0]
        check([clean(x) for x in pair] == [clean(x) for x in rec['root_actions']] and
              clean(pair[1-s]) == clean(raw['actions'][t][1-s]), f'enemy/physical order {p}')
        joint = clean(pair[s])
        if native == 'parent':
            check(joint == clean(raw['actions'][t][s]), 'parent order changed')
        key = 'move' if uid in joint else 'stay'
        check(key not in arms, 'stay/move pair')
        arms[key] = (rec, tape, joint, native)
    a, b = (arms[k][2] for k in ('stay', 'move'))
    check(all(a.get(k) == b.get(k) for k in set(a) | set(b) if k != uid), 'other worker modified')
    return arms


def extinct(obs, cfg):
    floor = min(cfg['recCollectorCost'], cfg['recPlanterCost'])
    return [not (workers or trees) and (not bases or cash < floor) for cash,bases,workers,trees in obs['players']]


def verify_step(sim, pair):
    before = sim.full_observation()
    idgen._id_dict.clear()
    for side in (0, 1):
        idgen._id_dict[f'player-{side}-worker-'] = sim.worker_counters[side]
        idgen._id_dict[f'player-{side}-tree-'] = sim.tree_counters[side]
    official = Board(before, sim.config, pair).next().observation
    sim.step(*pair)
    after = sim.full_observation()
    check(all(after[k] == official[k] for k in ('players','trees','carbon')), ('official fork mismatch', before['step']))
    return after


def replay_arm(snap, cfg, s, uid, end_cell, arm):
    rec,tape,_,native = arm
    sim = FastCarbon(cfg)
    sim.set_state(snap)
    sim.cash = CashEvents(sim.cash)
    t = sim.step_count
    start_cash = list(sim.cash)
    cashflow = [0.,0.]
    buy_t = tree_id = until = purchase_kind = None
    debit = absorption = collision = 0.
    survived = {}
    for j, pair in enumerate(tape):
        pre = sim.full_observation()
        check(not any(extinct(pre,cfg)) and pre['step'] < cfg['episodeSteps']-1, 'step after terminal')
        sim.cash.events.clear()
        post = verify_step(sim,pair)
        ev = sim.cash.events
        for e in ev:
            cashflow[e[1]] += e[2]
        if j <= 1:
            survived[str(j+1)] = uid in sim.workers
            pays = [e for e in ev if e[0] in ('plant','seize') and e[1]==s and e[3]==uid and e[4]==end_cell]
            check(len(pays) <= 1, 'two charges at one destination')
            if pays and buy_t is None:
                e = pays[0]
                buy_t, purchase_kind, debit = t+j, e[0], -e[2]
                tree_id = next(x.id for x in sim.trees.values() if x.pos==end_cell and x.player==s)
        if tree_id is not None and until is None:
            absorption += sum(e[2] for e in ev if e[1]==s and e[0]=='tree_absorption' and e[3]==tree_id)
            collision += sum(e[2] for e in ev if e[1]==s and e[0]=='tree_collision_cash' and e[5]==tree_id)
            if tree_id not in sim.trees or sim.trees[tree_id].player != s:
                until = sim.step_count
    ended = extinct(post,cfg)
    check(any(ended) or sim.step_count == cfg['episodeSteps']-1, 'partial game')
    scores = [(sim.step_count-cfg['episodeSteps']-1 if ended[p] else (pre if any(ended) else post)['players'][p][0])
              for p in (0,1)]
    check(scores == rec['score'] and rec['statuses'] == ['DONE','DONE'] and sim.step_count == rec['terminal_t']
          and rec['margin'] == scores[s]-scores[1-s] and math.isclose(utility(scores,s), rec['u']),
          f'official terminal mismatch {rec["seed"]}')
    check(all(math.isclose(cashflow[p]+start_cash[p],sim.cash[p],abs_tol=1e-6) for p in (0,1)), 'cash conservation')
    stayed = uid not in clean(tape[0][s])
    required_buy_t = t if stayed else t+1
    broad_diagnostic = {'purchase_action_t': buy_t, 'net':absorption+collision-debit}
    if buy_t != required_buy_t:
        buy_t = tree_id = until = purchase_kind = None
        debit = absorption = collision = 0.
    return {'native_arm':native,'root_order_destination':end_cell,'survived_root':survived['1'],
            'strict_eligible_purchase_action_t':required_buy_t,'both_transitions_sensitivity_only':broad_diagnostic,
            'survived_through_next_transition':survived.get('2'),'purchase_action_t':buy_t,'debit':debit,
            'purchase_kind':purchase_kind,'tree_id':tree_id,'end_first_ownership_observation_t':until,
            'through_terminal':tree_id is not None and until is None,
            'absorption':absorption,'tree_collision':collision,'first_tenure_net':absorption+collision-debit,
            'official_physical_scores':scores,'official_margin':rec['margin'],'terminal_observation_t':sim.step_count,
            'win':scores[s]>scores[1-s],'opponent_later_live_calls':rec['search_calls'],'archived_tape_sha256':rec['action_sha256']}


def evaluate_case(row, model, examiner_sha, cache):
    case = ROOT / row['case']
    protocol = load(case/'live_protocol.json')
    s,t,uid = row['seat'],row['t'],row['uid']
    p = ROOT / protocol['record']['source']
    if p not in cache:
        raw = load(p)
        sim = FastCarbon(raw['game_configuration']).reset(seed=raw['seed'])
        check(hashlib.sha256(bytes(map(int, sim.carbon))).hexdigest() == raw['initial_terrain_u8_sha256'], 'map initial state')
        check(raw['pack_sha256']==PARENT, 'non-v1 source')
        cache[p] = raw,sha(p),sim,deque(maxlen=3)
    raw,record_sha,sim,prior = cache[p]
    check(record_sha == protocol['record']['source_sha256'] and raw['initial_terrain_u8_sha256']==row['map']
          and raw['candidate_seat']==s and raw['seed']==row['seed'], 'original root/source mismatch')
    while sim.step_count < t:
        view = sim.obs(s)
        prior.append({k:view[k] for k in PUBLIC})
        sim.step(*raw['actions'][sim.step_count])
    check(t==sim.step_count and [o['step'] for o in prior] == list(range(t-3,t)), 'history order')
    view = sim.obs(s)
    obs = {k:view[k] for k in PUBLIC}
    saved = case/'root.json.gz'
    if saved.exists():
        old = load(saved)['public_at_root']
        check(all(obs[k]==old[k] for k in PUBLIC), f'public info mismatch {case}')
    arms = check_arms(case,raw,s,t,uid,examiner_sha,record_sha)
    cfg = {k:v for k,v in raw['game_configuration'].items() if k!='randomSeed'}
    pred = economic.decide(model,obs,list(prior),arms['stay'][2],arms['move'][2],uid,cfg)
    check(pred['pick'] in arms and all(math.isfinite(pred[k][metric]) for k in arms
          for metric in ('p_death','net_tree','utility')), 'invalid prediction')
    start = obs['players'][s][2][uid][0]
    snap = sim.get_state()
    out = {k:replay_arm(snap, cfg,s,uid,destination(start,a[2].get(uid,'NONE'),cfg['size']),a)
           for k,a in arms.items()}
    parent = next(k for k,a in arms.items() if a[3]=='parent')
    if 'arms' in row:
        check(parent == {'pay':'stay','defer':'move'}[row['parent_arm']], 'parent inversion')
        for k,old_k in [('stay','pay'),('move','defer')]:
            prev,now=row['arms'][old_k],out[k]
            check(prev['worker_survived_first']==now['survived_root'] and prev['margin']==now['official_margin']
                  and prev['action_sha256']==now['archived_tape_sha256'] and math.isclose(
                      prev['debit'],now['debit'] if now['purchase_action_t']==t else 0.,abs_tol=1e-8), 'old labels')
            if now['purchase_action_t']==t:
                check(math.isclose(prev['first_tenure_receipts'],now['absorption']+now['tree_collision'],abs_tol=1e-6),
                      'historical root tenure receipt mismatch')
    return {'case':row['case'],'seed':row['seed'],'map':row['map'],'seat':s,'t':t,'uid':uid,'historical_use':row.get('split','outside'),
            'parent_pick':parent,'C_prediction':pred,'arms':out,'own_root_joint':{k:v[2] for k,v in arms.items()},
            'history_and_public_view_sha256':digest([*prior,obs]),'source_sha256':record_sha,
            'common_opponent_root_sha256':digest(clean(raw['actions'][t][1-s]))}


def summarize(rows):
    def result(key):
        table = [r['arms'][key(r)] for r in rows]
        return {'root_survivals':sum(v['survived_root'] for v in table),
                'root_or_next_purchases':sum(v['purchase_action_t'] is not None for v in table),
                'mean_first_tenure_net':mean(v['first_tenure_net'] for v in table),
                'wins':sum(v['win'] for v in table),
                'mean_official_margin':mean(v['official_margin'] for v in table)}
    policy={'C':result(lambda r:r['C_prediction']['pick']),'always_stay_A_at_these_roots':result(lambda r:'stay'),
            'actual_parent':result(lambda r:r['parent_pick'])}
    def improvements(reference):
        delta=[r['arms'][r['C_prediction']['pick']]['official_margin']-r['arms'][reference(r)]['official_margin'] for r in rows]
        return {'delta_mean_margin':mean(delta),'positive':sum(v>0 for v in delta),'negative':sum(v<0 for v in delta),
                'zero':sum(v==0 for v in delta)}
    factual = {}
    for arm in ('stay','move'):
        observed = [r['arms'][arm] for r in rows]
        predictions = [r['C_prediction'][arm] for r in rows]
        factual[arm] = {
            'observed_death_fraction':mean(int(not v['survived_root']) for v in observed),
            'predicted_death_mean':mean(v['p_death'] for v in predictions),
            'death_brier':mean((p['p_death'] - int(not v['survived_root']))**2 for p,v in zip(predictions,observed)),
            'predicted_first_tenure_net_mean':mean(v['net_tree'] for v in predictions),
            'observed_first_tenure_net_mean':mean(v['first_tenure_net'] for v in observed),
            'first_tenure_net_mae':mean(abs(p['net_tree']-v['first_tenure_net']) for p,v in zip(predictions,observed)),
            'late_or_early_purchases_excluded_by_frozen_timing':sum(
                v['purchase_action_t'] != v['both_transitions_sensitivity_only']['purchase_action_t'] for v in observed)}
    return {'fixed_n':len(rows),'maps':len({r['map'] for r in rows}),'factual_prediction_check':factual,
            'C_stays':sum(r['C_prediction']['pick']=='stay' for r in rows),
            'C_moves':sum(r['C_prediction']['pick']=='move' for r in rows),
            'movement_arm_next_purchases':sum(r['arms']['move']['purchase_action_t']==r['t']+1 for r in rows),
            'policy':policy,'C_minus_A':improvements(lambda r:'stay'),'C_minus_parent':improvements(lambda r:r['parent_pick'])}


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--inventory-only',action='store_true')
    args=cli.parse_args()
    hist,extra,cover=inventory()
    write(HERE/'native_coverage.json',cover)
    if args.inventory_only:
        return
    model=load(HERE/'model.json')
    model_sha=sha(HERE/'model.json')
    examiner_sha=sha(EXAMINER) if EXAMINER.exists() else hashlib.sha256(subprocess.check_output(
        ['git','show',f'HEAD:{REL(EXAMINER)}'], cwd=ROOT)).hexdigest()
    check(examiner_sha=='65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5','examiner changed')
    cache={}
    rows=[]
    for row in sorted(hist+extra,key=lambda r:(r['seed'],r['seat'],r['t'])):
        rows.append(evaluate_case(row,model,examiner_sha,cache))
        if len(rows)%8==0:
            print('replayed paired roots',len(rows),'of',len(hist)+len(extra),flush=True)
    check(model_sha==sha(HERE/'model.json'),'model changed during replay')
    write(HERE/'native_eval_rows.json.gz',rows)
    groups={g:[r for r in rows if r['historical_use']==g] for g in ('train','hold','outside')}
    bymap=defaultdict(list)
    for r in rows:
        bymap[(r['historical_use'],r['seed'],r['map'])].append(r)
    report={'model_sha256':model_sha,'native_rows_sha256':sha(HERE/'native_eval_rows.json.gz'),
            'code_sha256':sha(Path(__file__)),'economic_sha256':sha(HERE/'economic.py'),
            'original_examiner_sha256':examiner_sha,'parent_sha256':PARENT,'prior_native_sha256':sha(NATIVE),
            'new_responding_games':0,'fresh_native_holdout':False,
            'scope':'Each landscape weights its four preselected historical opportunities equally; repeated source roots across arms are paired. Outside rows form a separate sensitivity check, with one opportunity per map. C is a single root action followed by the archived continuation. A is always stay at the same focal root only. These are already published, outcome-seen native response opportunities.',
            'money':'Stay buys only when this worker pays at its root landing on transition t. Move buys only when this worker pays at its new landing on transition t+1. Income is first tenure until owner changes, tree expires or game ends, minus the actual price. The optional broader two-transition field does not enter aggregates. Each fork uses its separately recorded subsequent native responses.',
            'by_historical_use':{g:summarize(rs) for g,rs in groups.items()},
            'original_64_only':summarize(groups['train']+groups['hold']),
            'per_map':[{'historical_use':g,'seed':seed,'map':land,**summarize(rs)}
                       for (g,seed,land),rs in sorted(bymap.items())]}
    write(HERE/'native_eval.json',report)
    print('original 64',report['original_64_only'],flush=True)
    print('outside',report['by_historical_use']['outside'],flush=True)


if __name__=='__main__':
    main()

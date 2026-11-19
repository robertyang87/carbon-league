"""Evaluate single historically supported order changes against both archived responding continuations."""
from collections import Counter, defaultdict
from hashlib import sha256
import gzip
import json
import math
from pathlib import Path
from statistics import mean
import sys
import subprocess

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parent
ROOT = ACTOR.parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from fastsim import FastCarbon

ARM_ROWS = ACTOR / 'two_arm_plant_economy_dev/native_eval_rows.json.gz'
ARM_RECEIPT = ACTOR / 'two_arm_plant_economy_dev/native_eval.json'
EXAM = ROOT / 'sources/master3s2/master3s2.py'
PUBLIC = ('step', 'player', 'players', 'trees', 'carbon')


def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f:
        return json.load(f)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def object_sha(o):
    return sha256(json.dumps(o, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_panel():
    report = read(ARM_RECEIPT)
    assert report['native_rows_sha256'] == digest(ARM_ROWS)
    exam_sha = digest(EXAM) if EXAM.is_file() else sha256(subprocess.check_output(
        ['git','show','HEAD:sources/master3s2/master3s2.py'],cwd=ROOT)).hexdigest()
    assert report['original_examiner_sha256'] == exam_sha
    rows = read(ARM_ROWS)
    main = [r for r in rows if r['historical_use'] in ('train', 'hold')]
    ext = [r for r in rows if r['historical_use'] == 'outside']
    assert len(main) == 64 and len({r['map'] for r in main}) == 16
    assert len(ext) == 2 and len({r['map'] for r in ext}) == 2
    assert not {r['map'] for r in main}.intersection(r['map'] for r in ext)
    return [r for r in main+ext if 100 <= r['t'] <= 199], report


def root_input(raw, row, sim):
    t, s, uid = row['t'], row['seat'], row['uid']
    previous_cell = None
    while sim.step_count < t:
        if sim.step_count == t-1:
            prior = sim.obs(s)['players'][s][2].get(uid)
            previous_cell = int(prior[0]) if prior else None
        sim.step(*raw['actions'][sim.step_count])
    assert sim.step_count == t
    obs = {k:v for k,v in sim.obs(s).items() if k in PUBLIC}
    saved = ROOT / row['case'] / 'root.json.gz'
    if saved.exists():
        archived = read(saved)['public_at_root']
        assert all(archived[k] == obs[k] for k in PUBLIC)
        obs = {k:archived[k] for k in PUBLIC}
    parent_joint = row['own_root_joint'][row['parent_pick']]
    assert parent_joint == {k:v for k,v in raw['actions'][t][s].items() if v and v != 'NONE'}
    return obs, previous_cell


def arm_proofs(row, report, src_sha):
    case = ROOT / row['case']
    sides = row['own_root_joint']
    uid = row['uid']
    assert set(sides) == {'stay','move'}
    assert {u for u in (sides['stay'].keys() | sides['move'].keys()) if sides['stay'].get(u) != sides['move'].get(u)} == {uid}
    assert uid not in sides['stay'] and sides['move'][uid] in ('UP', 'RIGHT', 'DOWN', 'LEFT')
    arms = {}
    for side, old in row['arms'].items():
        receipt = case / ('live_' + old['native_arm'] + '.json')
        rec = read(receipt)
        tape = case / rec['action_tape']
        assert digest(tape) == old['archived_tape_sha256'] == rec['action_sha256']
        assert digest(case/'live_protocol.json') == rec['protocol_sha256']
        assert rec['pack_sha256'] == report['parent_sha256'] and rec['source_sha256'] == report['original_examiner_sha256']
        assert rec['original_trajectory_sha256'] == src_sha
        assert rec['score'] == old['official_physical_scores'] and rec['margin'] == old['official_margin']
        assert rec['root_actions'][row['seat']] == sides[side]
        opp = {u:a for u,a in rec['root_actions'][1-row['seat']].items() if a and a != 'NONE'}
        assert object_sha(opp) == row['common_opponent_root_sha256']
        assert (old['native_arm']=='parent') == (side==row['parent_pick'])
        arms[side] = {k:old[k] for k in ('native_arm','purchase_action_t','purchase_kind','debit','first_tenure_net','survived_root','official_margin','official_physical_scores','archived_tape_sha256')}
        arms[side]['receipt_sha256'] = digest(receipt)
    return arms


def resolve(action, row, eligible):
    parent = row['parent_pick']
    if not eligible:
        return parent, 'outside_new_plant_location'
    matching = [arm for arm, joint in row['own_root_joint'].items() if joint.get(row['uid'],'NONE') == action]
    if not matching:
        return parent, 'unsupported_third_action'
    if matching == [parent]:
        return parent, 'same_as_parent'
    assert len(matching) == 1
    return matching[0], 'signed_alternative'


def evaluate(rows, report):
    import model
    cache = {}
    out = []
    for r in sorted(rows,key=lambda x:(x['seed'],x['seat'],x['t'])):
        case = ROOT / r['case']
        proto = read(case/'live_protocol.json')
        p = ROOT / proto['record']['source']
        if p not in cache:
            raw = read(p)
            assert r['source_sha256'] == proto['record']['source_sha256'] == digest(p)
            assert raw['seed']==r['seed'] and raw['pack_sha256']==report['parent_sha256']
            assert raw['initial_terrain_u8_sha256']==r['map']
            sim = FastCarbon(raw['game_configuration']).reset(seed=r['seed'])
            assert sha256(bytes(map(int,sim.carbon))).hexdigest() == r['map']
            cache[p] = raw,sim
        raw,sim = cache[p]
        assert raw['candidate_seat']==r['seat'] and proto['root_t']==r['t']
        obs, prev = root_input(raw,r,sim)
        cfg = {k:v for k,v in raw['game_configuration'].items() if k!='randomSeed'}
        uid = r['uid']
        cell,_,role = obs['players'][r['seat']][2][uid]
        assert role == 'PLANTER'
        base = any(cell in pl[1].values() for pl in obs['players'])
        live_tree = next((seat for seat,pl in enumerate(obs['players']) for loc,age in pl[3].values() if loc==cell and age<cfg['treeLifespan']),None)
        alive = sum(age < cfg['treeLifespan'] for pl in obs['players'] for loc,age in pl[3].values())
        me = obs['players'][r['seat']]
        quote = cfg['plantCost']
        if len(me[3])>cfg['playerTreeProtectiveNumber']:
            quote += cfg['plantCostInflationRatio'] * cfg['plantCostInflationBase']**alive
        eligible = not base and live_tree is None and me[0]>=quote
        choice = model.decide(obs,uid,cfg,previous_cell=prev)
        actions = extract_decisions(model,choice)
        policies = {key: dict(zip(('pick','reason'),resolve(a,r,eligible))) for key,a in actions.items()}
        policies['parent']={'pick':r['parent_pick'],'reason':'original'}
        arms=arm_proofs(r,report,r['source_sha256'])
        out.append({
            'case':r['case'],'seed':r['seed'],'terrain':r['map'],'seat':r['seat'],'t':r['t'],'uid':uid,
            'historical_use':r['historical_use'],'parent_pick':r['parent_pick'],'previous_cell':prev,'cell':cell,
            'root_kind':'base' if base else 'vacant' if live_tree is None else 'own_tree' if live_tree==r['seat'] else 'enemy_tree',
            'eligible':eligible,'own_public_cash':me[0],'current_quote':quote,'public_sha256':object_sha(obs),'source_sha256':r['source_sha256'],
            'actions':{arm:js.get(uid,'NONE') for arm,js in r['own_root_joint'].items()},
            'own_joint_sha256':{arm:object_sha(js) for arm,js in r['own_root_joint'].items()},
            'prediction':choice,'policies':policies,'arms':arms})
    return out


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[r['terrain']].append(r)
    results={}
    for name in ('weighted','bc','parent'):
        maps=[]
        for land, group in sorted(groups.items()):
            choices=[r['arms'][r['policies'][name]['pick']] for r in group]
            parent=[r['arms'][r['parent_pick']] for r in group]
            delta=mean(c['official_margin']-p['official_margin'] for c,p in zip(choices,parent))
            maps.append({'terrain':land,'roots':len(group),'eligible':sum(r['eligible'] for r in group),
                         'changed':sum(r['policies'][name]['pick']!=r['parent_pick'] for r in group),
                         'mean_official_margin':mean(c['official_margin'] for c in choices),'vs_parent':delta,
                         'reasons':dict(Counter(r['policies'][name]['reason'] for r in group))})
        choices=[r['arms'][r['policies'][name]['pick']] for r in rows]
        results[name]={'maps':maps,'changed':sum(x['changed'] for x in maps),'n':len(rows),
                       'eligible':sum(r['eligible'] for r in rows),'reasons':dict(Counter(r['policies'][name]['reason'] for r in rows)),
                       'map_equal_official_delta':mean(m['vs_parent'] for m in maps),
                       'positive_maps':sum(m['vs_parent']>1e-8 for m in maps),
                       'chosen_branch_wdl':dict(Counter('W' if c['official_margin']>0 else 'L' if c['official_margin']<0 else 'D' for c in choices)),
                       'real_purchases':dict(Counter(c['purchase_kind'] or 'none' for c in choices)),
                       'positive_first_tenure_new_plants':sum(c['purchase_kind']=='plant' and c['first_tenure_net']>0 for c in choices),
                       'first_tenure_net':sum(c['first_tenure_net'] for c in choices)}
    return results


def extract_decisions(model, decision):
    assert tuple(model.MOVES) == ('NONE','UP','RIGHT','DOWN','LEFT')
    p = decision['probabilities']
    q = decision['productive_probability']
    assert len(p) == len(q) == len(model.MOVES)
    assert abs(sum(p)-1)<1e-5 and all(math.isfinite(v) and 0<=v<=1+1e-6 for v in [*p,*q])
    k = max(range(5), key=lambda i:p[i])
    j = max(range(5), key=lambda i:p[i]*(1+q[i]))
    assert model.MOVES[k] == decision['bc_action'] and model.MOVES[j] == decision['action']
    return {'weighted':decision['action'],'bc':decision['bc_action']}


def panel_identity(rows):
    meta=read(HERE/'sources.json')
    records=[r for r in meta['records'] if r['split'] in ('fit','held')]
    assert {r['split'] for r in records}=={'fit','held'} and {r['domain'] for r in records}=={'native','elite'}
    ranges = {key: {r['terrain'] for r in records if r['split']==key} for key in ('fit','held')}
    assert not ranges['fit'].intersection(ranges['held'])
    # Include the out-of-time root and all 64 roots when checking actual board isolation.
    known = read(ARM_ROWS)
    native = {r['map'] for r in known}
    seeds = {r['seed'] for r in known}
    assert len(native)==18
    main={r['map'] for r in known if r['historical_use']!='outside'}
    external=native-main
    assert len(main)==meta['main_branch_terrains'] and len(external)==meta['external_branch_terrains']
    exposed={r['terrain'] for r in meta['records'] if r['split']=='exposed_loss_panel'}
    assert exposed==external and not exposed.intersection(main)
    assert digest(HERE/'protocol.json')==meta['protocol_sha256']
    overlap = {r['terrain'] for r in records} & native
    assert not overlap, overlap
    assert not seeds.intersection(int(r['seed']) for r in records if r.get('seed') is not None)
    return {'exposed_loss_additional_only_overlaps':len(exposed),'as_run_sources_sha256':digest(HERE/'sources.json'),'main_and_external_pre_window_boards':len(native),
            'in_window_main':sum(r['historical_use']!='outside' for r in rows),
            'in_window_external':sum(r['historical_use']=='outside' for r in rows),
            'fit_boards':len(ranges['fit']),'held_boards':len(ranges['held']),'seed_intersection':0,'terrain_intersection':0,
            'external_note':'The two historical external maps are also used for ancillary observed teacher behavior in the declared known-loss panel. They were excluded from fitting and primary behavior holdout. Only one external root lies in the requested time interval.'}


def run():
    rows,report=source_panel()
    groups=Counter(r['historical_use'] for r in rows)
    assert groups=={'train':25,'hold':9,'outside':1}
    sep=panel_identity(rows)
    all_rows=evaluate(rows,report)
    sections={k:summarize([r for r in all_rows if r['historical_use'] in use]) for k,use in {
        'historical64_in_window':('train','hold'),
        'historical_training_subset':('train',),'historical_held_subset':('hold',),'external2_in_window':('outside',)}.items()}
    primary=sections['historical64_in_window']
    output={'enumeration':sep,'sources':{'audit_sha256':digest(ARM_RECEIPT),'source_two_arm_labels_sha256':digest(ARM_ROWS),
             'protocol_sha256':digest(HERE/'protocol.json'),'scorer_sha256':digest(Path(__file__)),
             'model_source_sha256':digest(HERE/'model.py'),'original_examiner_sha256':report['original_examiner_sha256'],
             'original_parent_sha256':report['parent_sha256']},
            'previously_studied_native_roots':True,'native_games_started':0,
            'definition':'Only an empty currently plantable focal tile and an exact change to the signed alternative count. Off-panel action/same order or current enemy-owned tree retains original. Both complete joints differ solely at this UID. Stay roots can purchase at t; moving roots can purchase the very same destination at t+1. First-tenure net includes observed tree absorption and incoming collision money minus the actual price. Official margin includes the full later team response to each joint order.',
            'summary':sections,'roots':all_rows}
    output['sources']['local_weights_sha256']={'primary.npz':digest(HERE/'primary.npz')}
    assert output['sources']['local_weights_sha256']
    output['main_map_equal_weighted_minus_bc']=primary['weighted']['map_equal_official_delta']-primary['bc']['map_equal_official_delta']
    data=gzip.compress((json.dumps(output,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode(),mtime=0)
    target=HERE/'native.json.gz'
    if target.exists():
        assert target.read_bytes()==data
    else:
        tmp=target.with_suffix('.tmp')
        tmp.write_bytes(data)
        tmp.replace(target)
    return output


if __name__=='__main__':
    doc=run()
    for k,p in doc['summary'].items():
        print(k, {n: (v['n'],v['eligible'],v['changed'],v['map_equal_official_delta']) for n,v in p.items()})

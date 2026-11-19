"""Select training roots from pre-action public economics and reproduce them from signed action prefixes."""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import zipfile

HERE = Path(__file__).resolve().parent
INTAKE = HERE.parent
ROOT = next(p for p in HERE.parents if (p / '.git').exists())
sys.path.insert(0, str(INTAKE.parent / 'natural_original'))
import audit_menu

PROTOCOL_SHA = '7fe96868b3e8cf646e912b6949c27042bdfb4909c06a9978e701e20b0cc11f89'
PROTOCOL_COMMIT = 'ef28aba16dedb3bf6fee65efbd5426bd30be11da'


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(',', ':')).encode()


def save(path, doc):
    data = canonical(doc) + b'\n'
    if path.exists():
        need(path.read_bytes() == data, f'Conflict with existing immutable record: {path}')
    else:
        path.write_bytes(data)
    return digest(data)


def signed():
    raw = (HERE / 'protocol.json').read_bytes()
    need(digest(raw) == PROTOCOL_SHA, 'Selection rules changed')
    return json.loads(raw)['source']


def inventory():
    source = signed()
    raw = (INTAKE / 'hosted/settlement.json').read_bytes()
    need(digest(raw) == source['settlement_sha256'], 'Source certificates changed')
    # Only the archive registry is inspected by the selection stage.
    archives = json.loads(raw)['archives']
    expected = {(s, seat) for s in source['seeds'] for seat in (0, 1)}
    need(len(archives) == len(expected) and {(r['seed'], r['seat']) for r in archives} == expected, 'Bad matrix')
    return sorted(archives, key=lambda r: (r['seed'], r['seat']))


@contextmanager
def open_episode(ref, arm):
    path = ROOT / ref['path']
    need(digest(path.read_bytes()) == ref['sha256'], 'Wrong signed archive')
    with zipfile.ZipFile(path) as z, TemporaryDirectory() as tmp:
        reg = z.read(arm + '/registration.json')
        reg_path = Path(tmp) / 'registration.json'
        reg_path.write_bytes(reg)
        registration = json.loads(reg)
        need((registration['seed'], registration['candidate_seat']) == (ref['seed'], ref['seat']) and
             registration['files']['candidate']['sha256'] == signed()['policy_sha256'], 'Wrong draw')
        live = audit_menu.open_live(reg_path)
        env, _ = live.official()
        with z.open(arm + '/turns.jsonl') as tape:
            yield env, live, tape, registration, reg


def economics(obs, seat):
    rows = obs['players']
    cash = [p[0] for p in rows]
    cargo = [sum(w[1] for w in p[2].values() if w[2] == 'COLLECTOR') for p in rows]
    workers, trees = [len(p[2]) for p in rows], [len(p[3]) for p in rows]
    gap = cash[seat] - cash[1-seat]
    adjusted = gap + .5 * (cargo[seat]-cargo[1-seat])
    return dict(cash=cash,collector_cargo=cargo,workers=workers,trees=trees,cash_gap=round(gap,6),
                adjusted_gap=round(adjusted,6),stratum='behind' if adjusted < -150 else 'ahead' if adjusted > 150 else 'level')


def fingerprint(env, terrain):
    from zerosum_env.envs.carbon import idgen
    obs = env.states[0].observation
    counters = dict(idgen._id_dict)
    state = dict(terrain=terrain,step=obs.step,players=obs.players,trees=obs.trees,
                 full_carbon=obs.full_carbon,counters=counters,states=[[p.status,p.reward] for p in env.states])
    return digest(canonical(state)), counters


def advance(env, row):
    orders = []
    for call in row['calls']:
        need(call['status'] == 'ok', 'Missing public order')
        orders.append(call['response']['action'])
    env.step(orders)


def enumerate_roots(ref, arm):
    seed, seat = ref['seed'], ref['seat']
    cases, counts = [], Counter()
    order = 'draw_a' if (seed + seat) % 2 == 0 else 'draw_b'
    band0 = (seed-621758460+seat) % 3
    with open_episode(ref, arm) as (env, live, tape, reg, reg_bytes):
        bank, prefix = [5.,5.], sha256()
        for t in range(276):
            if env.done:
                break
            need(env.states[0].observation.step == t, 'Invalid official step')
            observations = [live.observe(env,s,bank[s]) for s in (0,1)]
            public_hashes = [digest(live.encoded(v)) for v in observations]
            if 200 <= t:
                band = int(t>=225) + int(t>=250)
                econ = economics(observations[seat], seat)
                key = econ['stratum']
                counts.update(['examined',f'band{band}_examined',f'{key}_examined'])
                if min(econ['workers']) >= 1 and abs(econ['cash_gap']) <= 600 and abs(econ['adjusted_gap']) <= 600:
                    counts.update(['eligible',f'band{band}_eligible',f'{key}_eligible'])
                    state_hash, counters = fingerprint(env, reg['physical_map_sha256'])
                    root = dict(seed=seed,seat=seat,arm=arm,step=t,band=band,**econ,terrain=reg['physical_map_sha256'],
                                state_sha256=state_hash,public_sha256_by_seat=public_hashes,rule_id_counters=counters,
                                bank_before_by_seat=bank,raw_prefix_sha256=prefix.hexdigest(),prefix_actions=t,
                                archive_path=ref['path'],archive_sha256=ref['sha256'],registration_member=arm+'/registration.json',
                                registration_sha256=digest(reg_bytes),tape_member=arm+'/turns.jsonl',history_range=[0,t])
                    cases.append((((band-band0) % 3,t,arm!=order),root))
            if t==275:
                break
            raw = tape.readline()
            row = json.loads(raw)
            need(row['step']==t and row['public_observation_sha256']==public_hashes, 'Signed historical observation mismatch')
            advance(env,row)
            prefix.update(raw)
            bank = row['overage_remaining_seconds']
    return counts, cases


def select():
    refs = inventory()
    all_counts, seen, tickets = Counter(), set(), []
    for ref in refs:
        n, combined = Counter(), []
        for arm in signed()['arms']:
            arm_counts, cases = enumerate_roots(ref,arm)
            n.update(arm_counts)
            combined.extend(cases)
        all_counts.update(n)
        choice = next((root for _,root in sorted(combined,key=lambda x:x[0]) if root['state_sha256'] not in seen),None)
        if choice is not None:
            seen.add(choice['state_sha256'])
        tickets.append(dict(seed=ref['seed'],seat=ref['seat'],counts=dict(n),eligible_distinct_states=len({r['state_sha256'] for _,r in combined}),root=choice))
    doc = dict(schema='sapling-endgame-training-roots/1',protocol_commit=PROTOCOL_COMMIT,protocol_sha256=PROTOCOL_SHA,
               selection_script_sha256=digest(Path(__file__).read_bytes()),counts=dict(all_counts),total_map_seats=len(refs),
               selected=sum(t['root'] is not None for t in tickets),tickets=tickets)
    print('roots_sha256',save(HERE/'roots.json',doc),doc['selected'],dict(all_counts),flush=True)


def restore(root, ref, live, env, tape):
    prefix,bank = sha256(),[5.,5.]
    for step in range(root['step']):
        raw = tape.readline()
        row = json.loads(raw)
        public = [live.observe(env,s,bank[s]) for s in (0,1)]
        need(row['step']==step and row['public_observation_sha256']==[digest(live.encoded(v)) for v in public], 'Prefix observation changed')
        advance(env,row)
        bank = row['overage_remaining_seconds']
        prefix.update(raw)
    state_hash,counters = fingerprint(env,root['terrain'])
    hashes = [digest(live.encoded(live.observe(env,s,bank[s]))) for s in (0,1)]
    need(root['raw_prefix_sha256']==prefix.hexdigest() and root['public_sha256_by_seat']==hashes and
         root['state_sha256']==state_hash and root['rule_id_counters']==counters and root['bank_before_by_seat']==bank and
         root['archive_sha256']==ref['sha256'], 'Immutable training root could not be restored')
    return bank


if __name__ == '__main__':
    select()

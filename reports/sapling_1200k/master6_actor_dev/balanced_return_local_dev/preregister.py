"""Freeze a diagnostic for the last released head. Its two reused maps cannot measure independent strength."""
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREV = HERE.parent / 'balanced_return_dev'
PARENT = HERE.parent / 'actor_partner_dev/candidate.py'
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes())


def encode(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode()


def save(path, value):
    body = value if isinstance(value, bytes) else encode(value)
    if path.exists():
        if path.read_bytes() != body:
            raise ValueError(f'prior receipt differs: {path}')
        return
    temp = path.with_name(path.name + '.tmp')
    temp.write_bytes(body)
    temp.replace(path)


def make():
    reviewed = read(PREV / 'package.json')
    assert sha(PARENT) == reviewed['parent_sha256'] and sha(PREV / 'candidate.py') == reviewed['candidate_sha256']
    plan = read(PREV/'maps.json')
    inputs = {}
    groups = {}
    for seed in plan['training']:
        for physical_source in (0, 1):
            for t in (205, 271):
                p = PREV / 'train' / f'{seed}_{physical_source}_{t}.json.gz'
                ticket = read(p)['state']['header']
                assert (ticket['seed'], ticket['original_seat'], ticket['t']) == (seed, physical_source, t)
                assert ticket['binding'] == read(PREV/'binding.json')
                choice = ticket['choice']
                load, distance, threat = choice['context']
                key = f'{t}/{int(load >= 40)}/{int(threat)}'
                groups.setdefault(key, []).append((load, distance))
                inputs[p.name] = sha(p)
    ranges = {key: dict(count=len(rows), cargo=[min(a for a,b in rows),max(a for a,b in rows)],
                        distance=[min(b for a,b in rows),max(b for a,b in rows)]) for key, rows in groups.items()}
    save(HERE/'protocol.json', dict(schema='sapling-local-worker-head-mechanism-1', diagnosis_only=True,
         maps=plan['development'], physical_seats=[0,1], map_use='Both boards have previously reported parent and global-adapter results. Reuse all four paired seat cases; no independent holdout or native exam.',
         windows=[205,271], roles=['collector'], endpoint_only=True,
         state_gate='Current cargo and toroidal distance to own base must lie in the observed closed interval for that precise step, cargo >=40 band and strict lighter enemy can enter current cell bit. No imputation across absent strata.',
         choice='At most one real collector per registered step. Rank eligible slots with >=2 legal actions by descending parent guard-candidate decode T1 entropy, lower slot tie-break. Do not condition on map identifier, opponent identity, labels or future.',
         implementation='Use the actual released float32 trained head only at that one Sapling slot, in BOTH basic and return-adjusted decodes, with exact public prefix updates. All other roles and slots use original Sapling head and partner. Re-evaluate original whole-command return guard. Direct change to another slot is forbidden; changed prefixes may change subsequent physical commands.',
         checks='On training roots compare independently computed eligible slots, selected focus and resulting full joint order; verify the execution at each branch/slot uses old/new exact FP32 tensors and guarded full re-decode. Adjacent inactive observations must exactly match baseline. On each candidate game compare whole physical actions to baseline at every root; within same observation any outside-window divergence invalidates execution.',
         outcomes='Frozen earlier own-it50 comparison: all four new-from-start candidate games, all four old-parent and four ungated controls signed in previous experiment. Opposing policy, scoring, seats and maps stay identical. Require both map mean U positive, both mean margin nonnegative, at least one actual change and real Sapling ablation, no fewer wins, full eight parent/child receipts. Even if these pass, reuse of exposed maps makes this diagnostic only.',
         neighbor_audit='Count as-if eligible and potential command effects on +/-5 observed positions to characterize risk, do not deploy adjacent eligibility. Selected physical head support is only two exact decisions.',
         references=dict(parent=reviewed['parent_sha256'],ungated=reviewed['candidate_sha256'],
                         weights=reviewed['weights_sha256'],screen=sha(PREV/'screen.json'),
                         historical_comparison=sha(PREV/'development/comparison.json'),maps=sha(PREV/'maps.json'),
                         registered_windows_source=sha(PREV/'protocol.json'),inputs=inputs),
         ranges=ranges, budget='Local CPU2/3 low priority single-thread per process. No original Master3s2 processes, no new formal or native maps.'))


if __name__ == '__main__':
    make()

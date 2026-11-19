"""Independently replay every signed *new candidate* trajectory through three public agents."""
import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parent
ROOT = ACTOR.parents[2]
GAMES = HERE / 'games'
BASELINE = ACTOR / 'blended_actor_dev/serving/candidate.py'
RELEASE = ACTOR / 'blended_actor_dev/opening_training/release.json'
sys.path.insert(0, str(ROOT / 'code'))
from fastsim import FastCarbon


def packed(value):
    return json.dumps(value, separators=(',', ':'), sort_keys=True, allow_nan=False).encode()


def sha(data):
    if isinstance(data, Path):
        data = data.read_bytes()
    return hashlib.sha256(data).hexdigest()


def read(path):
    b = path.read_bytes()
    return json.loads(gzip.decompress(b) if path.suffix == '.gz' else b)


def load_agent(path, name):
    spec = importlib.util.spec_from_file_location('opening_trajectory_' + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def packages(plan):
    release = read(RELEASE)
    assert sha(BASELINE) == plan['baseline_sha256'] == release['base_sha256']
    key = release['materialized_sha256']
    choices = (Path(tempfile.gettempdir()) / ('sapling-only-opening-' + key + '.py'),
               Path('/workspace/sapling-opening-trained-agent.py'))
    path = next((f for f in choices if f.is_file() and sha(f) == key), None)
    if path is None:
        raise FileNotFoundError('Materialize the signed candidate with opening_training/build.py.')
    assert key == plan['candidate_pack_sha256']
    agents = {arm: load_agent(BASELINE if arm=='parent' else path, arm)
              for arm in ('parent', 'candidate', 'disabled')}
    agents['disabled']._OPENING_ENABLED = False
    assert agents['candidate']._OPENING_ENABLED
    assert len({a._PACK_SRC_SHA256 for a in agents.values()}) == 1
    return agents


def compare(seed, seat, plan, binding, agents):
    folder = GAMES / 'candidate'
    receipt = folder / f'receipt_{seed}_{seat}.json'
    signed = read(receipt)
    assert signed['binding'] == binding
    row, official = signed['game'], signed['official']
    path = folder / row['raw_file']
    assert path.name == f'raw_{seed}_{seat}.json.gz' and sha(path) == row['raw_sha256']
    tape = read(path)
    assert (tape['seed'], tape['candidate_seat']) == (seed, seat)
    assert tape['pack_sha256'] == binding['candidate'] == plan['candidate_pack_sha256']
    assert tape['errors'] == row['errors'] == [] and row['valid'] and len(tape['actions']) == row['actual_transitions'] == official['steps']
    assert tape['official_score_by_seat'] == official['official_reward_by_seat']
    assert sha(packed(tape['actions'])) == row['actions_sha256']
    sim = FastCarbon(tape['game_configuration']).reset(seed=seed)
    assert sim.carbon == tape['initial_full_carbon']
    map_hash = sha(bytes(int(x) for x in sim.carbon))
    assert map_hash == official['map_sha256'] == binding['precise_terrain'][str(seed)]
    full_rows = []
    for t, joint in enumerate(tape['actions']):
        assert sim.step_count == t
        # Each adapter has its own Feat/policy and sees fresh equal objects. No raw joint
        # command or opponent-private observation is passed to any public agent.
        obs = sim.obs(seat)
        assert obs['step'] == t and obs['player'] == seat and 'full_carbon' not in obs
        actions = {}
        for arm, module in agents.items():
            actual = module.agent(copy.deepcopy(obs), copy.deepcopy(sim.config))
            assert isinstance(actual, dict) and module._FEAT.step == t
            actions[arm] = actual
        assert len({id(module._FEAT) for module in agents.values()}) == len(agents)
        assert len({id(module._NETS) for module in agents.values()}) == len(agents)
        assert not agents['candidate']._OPENING_FIRST_CALL and not agents['disabled']._OPENING_FIRST_CALL
        full_rows.append({'t': t, 'public_sha256': sha(packed(obs)), 'tape': joint[seat], **actions})
        # Both real actions, including the unseeded native opponent, are replayed verbatim.
        sim.step(*joint)
    assert sha(packed(sim.get_state())) == row['final_state_sha256']
    after = full_rows[1:]
    zero = full_rows[0]
    flags = {
        't0_candidate_changed_from_parent': zero['candidate'] != zero['parent'],
        't0_candidate_matches_natural_game': zero['candidate'] == zero['tape'],
        'candidate_vs_tape_difference_steps': [r['t'] for r in full_rows if r['candidate'] != r['tape']],
        'parent_vs_candidate_after_opening_difference_steps': [r['t'] for r in after if r['parent'] != r['candidate']],
        'disabled_vs_parent_difference_steps': [r['t'] for r in full_rows if r['disabled'] != r['parent']],
        'parent_vs_tape_after_opening_difference_steps': [r['t'] for r in after if r['parent'] != r['tape']],
    }
    passed = all(v if isinstance(v, bool) else not v for v in flags.values())
    return {'seed': seed, 'seat': seat, 'terrain': map_hash, 'receipt_sha256': sha(receipt),
            'raw_sha256': row['raw_sha256'], 'actions_sha256': row['actions_sha256'],
            'restored_final_state_sha256': row['final_state_sha256'], 'steps': len(full_rows),
            't0_commands': zero, 'checks': flags, 'pass': passed, 'rows': full_rows}


def run(follow):
    plan = read(HERE / 'protocol.json')
    binding = read(GAMES / 'binding.json')
    assert sha(HERE / 'protocol.json') == binding['registered_sha256']
    assert sha(BASELINE) == binding['baseline']
    targets = [tuple(pair) for pair in plan['new_order']]
    assert len(targets) == len(set(targets)) == plan['new_games'] == 4
    finished = {}
    agents = packages(plan)
    while len(finished) != len(targets):
        for seed, seat in targets:
            if (seed, seat) in finished:
                continue
            if not (GAMES / 'candidate' / f'receipt_{seed}_{seat}.json').is_file():
                continue
            g = compare(seed, seat, plan, binding, agents)
            finished[(seed, seat)] = g
            print(seed, seat, g['steps'], g['checks'], flush=True)
        if len(finished) == len(targets):
            break
        if not follow:
            raise FileNotFoundError('Signed candidate receipts available: ' + str(sorted(finished)))
        time.sleep(8)
    games = [finished[pair] for pair in targets]
    out = {
        'source': {str(p.relative_to(ROOT)): sha(p) for p in
                   (Path(__file__).resolve(), ROOT/'code/fastsim.py', HERE/'protocol.json', GAMES/'binding.json', RELEASE)},
        'parent_sha256': binding['baseline'], 'candidate_sha256': binding['candidate'],
        'exact_receipts': len(games), 'complete_natural_candidate_turns': sum(g['steps'] for g in games),
        'observed_t0_changes': sum(g['checks']['t0_candidate_changed_from_parent'] for g in games),
        'post_opening_turns': sum(g['steps']-1 for g in games),
        'post_opening_parent_candidate_differences': sum(len(g['checks']['parent_vs_candidate_after_opening_difference_steps']) for g in games),
        'disabled_control_differences': sum(len(g['checks']['disabled_vs_parent_difference_steps']) for g in games),
        'candidate_vs_actual_differences': sum(len(g['checks']['candidate_vs_tape_difference_steps']) for g in games),
        'live_evidence': 'Candidate full tournament action tape against the registered native opponent on these two development maps. Both submitted actions advance the state. Every public input and every output command is preserved below.',
        'identity_control': 'Separately, the actual candidate implementation with its opening head disabled is run at every turn against the parent. Observations and configurations match by value, policy and feature state is separately instantiated.',
        'scope': 'This tests direct changes to commands on the same observed candidate trajectory. The new opening can change subsequent observed states, both players’ actions and the result. The test does not isolate the causal value of the new recruit against the naturally sampling opponent.',
        'games': games, 'pass': all(g['pass'] for g in games),
    }
    path = HERE / 'parity.json.gz'
    tmp = path.with_name('.parity.' + str(os.getpid()) + '.tmp')
    tmp.write_bytes(gzip.compress(packed(out), mtime=0))
    assert read_output(tmp)['pass'] == out['pass']
    os.replace(tmp, path)
    print('pass', out['pass'], 'sha256', sha(path), flush=True)


def read_output(path):
    return json.loads(gzip.decompress(path.read_bytes()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--follow', action='store_true', help='Wait for 4 signed receipts, never opens a new game.')
    run(p.parse_args().follow)

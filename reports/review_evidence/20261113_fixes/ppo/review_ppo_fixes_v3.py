"""Read-only F01/F06/S5a follow-up checks on real resume/rollout/decoder functions.
Synthetic tiny policies and constructed elimination-boundary states replace
expensive training. Reachability by a complete legal game sequence is not tested.
Set CARBON_REPO to the repository root, or place this script inside that repository.
Run with Python plus NumPy, Torch and the original environment's dependencies.
Results go to stdout; temporary checkpoint fixtures are automatically removed.
"""
import ast
import copy
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import types
import warnings

os.environ['CNET2_TEMP_KIND'] = '0.15,0.3,0.3'
os.environ['RL2_SNAP_TEMP'] = '0.5'
os.environ['RL2_PACK'] = '1'


def find_repo():
    configured = os.environ.get('CARBON_REPO')
    if configured:
        root = pathlib.Path(configured).expanduser().resolve()
        if not (root / 'code/fastsim.py').is_file():
            raise RuntimeError(f'CARBON_REPO does not contain code/fastsim.py: {root}')
        return root
    for root in pathlib.Path(__file__).resolve().parents:
        if (root / 'code/fastsim.py').is_file():
            return root
    raise RuntimeError('Cannot find the repository; set CARBON_REPO to its root.')


REPO = find_repo()
sys.path.insert(0, str(REPO / 'code'))
sys.path.insert(0, str(REPO / 'code/env_src/carbon_challenge-master'))
import numpy as np
import torch
torch.set_num_threads(1)
import rl2_ppo_train as TR
import rl2_ppo_rollout as RO
import rl2_ppo_cnet2 as CN
from fastsim import FastCarbon, DEFAULT_CONFIG
from zerosum_env import utils
from zerosum_env.envs.carbon import carbon as rules, idgen

result = {}

# Execute the actual resume AST, including CLI-vs-checkpoint selection.
path = REPO / 'code/rl2_ppo_train.py'
main = next(n for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == 'main')
block = next(n for n in main.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'args.resume')
compiled = compile(ast.Module(body=[block], type_ignores=[]), str(path), 'exec')
resume_cases = []
with tempfile.TemporaryDirectory(prefix='carbon-fix-resume-') as tmp:
    net, vnet = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    oldopt = torch.optim.Adam(net.parameters(), lr=1e-5)
    oldvopt = torch.optim.Adam(vnet.parameters(), lr=3e-4)
    ckpt = str(pathlib.Path(tmp) / 'state.pt')
    torch.save({'net': net.state_dict(), 'vnet': vnet.state_dict(),
                'opt': oldopt.state_dict(), 'vopt': oldvopt.state_dict(),
                'it': 24, 'kl_coef': .1}, ckpt)
    for case, req, expected in [
        ('explicit_all', (3e-5, 8e-4, .025), (3e-5, 8e-4, .025)),
        ('inherit_all', (None, None, None), (1e-5, 3e-4, .1)),
        ('explicit_zero_kl', (None, None, 0.), (1e-5, 3e-4, 0.)),
        ('override_lr_only', (2e-5, None, None), (2e-5, 3e-4, .1))]:
        args = types.SimpleNamespace(resume=ckpt, lr=req[0], vlr=req[1], kl=req[2])
        opt = torch.optim.Adam(net.parameters(), lr=req[0] if req[0] is not None else 1e-5)
        vopt = torch.optim.Adam(vnet.parameters(), lr=req[1] if req[1] is not None else 3e-4)
        ns = dict(torch=torch, args=args, net=net, vnet=vnet, opt=opt, vopt=vopt,
                  it0=0, kl_coef=req[2] if req[2] is not None else .1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            exec(compiled, ns)
        got = (opt.param_groups[0]['lr'], vopt.param_groups[0]['lr'], ns['kl_coef'])
        assert got == expected, (case, got)
        resume_cases.append(dict(case=case, requested=req, effective=got, passed=True))
result['resume'] = resume_cases


class FixedPolicy:
    w = {'value.weightT': np.zeros((1, 1)), 'value.bias': np.zeros(1),
         'aemb.weight': np.zeros((9, 1))}

    def encode(self, x, teacher):
        return np.zeros((CN.F.T, 1)), np.zeros(1)

    def lin(self, x, name):
        if name == 'whead':
            return np.array([10., 0., 0., 0., 0.])
        if name == 'bhead':
            return np.array([10., 0., 0.])
        if name == 'V':
            return np.zeros((5, 1))
        return np.zeros(1)

    def gru(self, x, h):
        return np.zeros(1)


class TinyActor(CN.Cnet2Actor):
    def __init__(self, weights):
        self.pol = FixedPolicy()
        self.remote = None
        self.key = None
        self.teacher = CN.TEACHER
        self.sched = True
        self.temp_kind = False
        self.obs_calls = []

    def prepare(self, states, obss):
        self.obs_calls.extend((id(s), int(o['step'])) for s, o in zip(states, obss))
        return super().prepare(states, obss)


class FixtureSim(FastCarbon):
    def reset(self, seed=None, **kw):
        if seed == 17:
            # Actual generated initial map for the legal-action sequence test.
            return super().reset(seed=42)
        super().reset(full_carbon=[0.] * 225)
        if seed in (11, 13, 14):
            # A constructed elimination-boundary position: side 0 has only 20
            # cash, side 1 has 10 cash plus a live collector. No actions change
            # it. We do not prove reachability through a complete game sequence.
            obs = self.full_observation()
            if seed == 13:
                obs['players'][0][0] = 20.
                obs['players'][1][0] = 20.
            else:
                survivor = 0 if seed == 14 else 1
                obs['players'][1 - survivor][0] = 20.
                obs['players'][survivor][0] = 10.
                pos = next(iter(obs['players'][survivor][1].values()))
                cargo = 100. if seed == 14 else 0.
                obs['players'][survivor][2] = {f'player-{survivor}-worker-0': [pos, cargo, 'COLLECTOR']}
            self.load_observation(obs)
        return self


def original_finish(seed, turns=1, recruit=False):
    """The real original interpreter establishes the authoritative terminal score."""
    origin = FixtureSim().reset(seed=seed).full_observation()
    states = []
    for p in range(2):
        obs = copy.deepcopy(origin)
        obs['full_carbon'] = obs['carbon'][:]
        obs['player'] = p
        states.append(utils.structify(dict(observation=obs, action={}, status='ACTIVE',
                                          reward=origin['players'][p][0])))
    idgen.reset()
    for _ in range(turns):
        if recruit:
            bid = next(iter(states[0].observation.players[0][1]))
            states[0].action = {bid: 'RECCOLLECTOR'}
        states = rules.compute_next_board(states, types.SimpleNamespace(configuration=utils.structify(DEFAULT_CONFIG)))
    return states


original = original_finish(11)
original_margin = float(original[0].reward - original[1].reward)

# Keep actual actor cache, play_batch, Cnet2Actor decoding, features, trajectory
# packing, buffer batching and GAE; replace only expensive model weights and seeds.
RO._W_CACHE.clear()
RO.AD.load_npz = lambda path: {}
RO.AD.get_adapter = lambda name: types.SimpleNamespace(make_actor=TinyActor)
RO.FastCarbon = FixtureSim
out = RO.play_batch({'adapter': 'cnet2', 'wkey': 'tiny-A',
                     'games': [('SELF', 11, 0), ('SELF', 12, 0)],
                     'temp': .7, 'greedy': True, 'record': True, 'rng_seed': 19})
games = out['games']
assert [g['steps'] for g in games] == [1, 1, 299, 299]
assert [len(g['recs']) for g in games] == [1, 1, 299, 299]
assert all(len(g['recs']) == len(g['dmargin']) == len(g['dworth']) == len(g['dep']) == len(g['lost'])
           for g in games)
assert all(np.isclose(g['dmargin'].sum(), g['margin']) and np.isclose(g['dworth'].sum(), g['margin'])
           for g in games)
assert np.array_equal(games[0]['dmargin'], -games[1]['dmargin'])
buf = CN.Cnet2Buffer([g['recs'] for g in games])
batch = buf.batch([0, 1, 2, buf.n - 1], torch.device('cpu'))
assert buf.n == 600 and batch['B'] == 4
assert out['ndec'] == buf.n
for g in games:
    adv, ret = TR.gae(g['dmargin'], np.zeros(len(g['recs']), np.float32), .995, .95)
    assert len(adv) == len(ret) == len(g['recs'])
result['rollout_lengths_and_pack'] = dict(
    lengths=[len(g['recs']) for g in games], terminal=[g['term'] for g in games],
    buffer_decisions=buf.n, batch_size=batch['B'], last_normal_obs_step=int(round(games[-1]['recs'].d['scal'][-1, 1] * 300)),
    actual_decodes=out['ndec'], recorded_decisions=buf.n,
    unnecessary_terminal_decodes=out['ndec'] - buf.n,
    reward_arrays_and_GAE_lengths_pass=True)
reward_cases = []
for kind in ('dense', 'terminal', 'worth'):
    args = types.SimpleNamespace(reward=kind, c_dep=0., c_lost=0., shape_iters=200,
                                 rscale=1., win_bonus=.1)
    reward_cases.append(dict(kind=kind, sum_reward=float(TR.game_rewards(games[0], args, 1).sum())))
result['elimination_reward_parity'] = dict(
    original_rewards=[float(s.reward) for s in original], original_margin=original_margin,
    rollout_margin=games[0]['margin'], rollout_win=games[0]['margin'] > 0,
    reward_modes=reward_cases)
assert original_margin == games[0]['margin'] == -310.
assert all(np.isclose(r['sum_reward'], original_margin - .1) for r in reward_cases)

# F06: same-path primary and SNAP must be distinct cached instances.
primary = RO._actor('cnet2', 'tiny-A', role='learner')
fresh = RO._actor('cnet2', 'tiny-B', role='snap')
cached_snapshot = RO.make_opponent('SNAP:cnet2:tiny-A', 1, 9)
result['actor_roles'] = dict(primary_temp_kind=primary.temp_kind,
                             fresh_snapshot_temp_kind=fresh.temp_kind,
                             cached_snapshot_is_primary=cached_snapshot.actor is primary,
                             cached_snapshot_temp_kind=cached_snapshot.actor.temp_kind)
assert primary.temp_kind and not fresh.temp_kind and not cached_snapshot.actor.temp_kind
assert cached_snapshot.actor is not primary

# Use the real decoder to prove fresh snapshot probabilities depend on the
# caller temperature, learner probabilities don't, and cached snapshots are isolated.
obs = FixtureSim().reset(seed=12).obs(0)
probs = {}
for name, actor in [('learner', primary), ('fresh_snapshot', fresh), ('cached_snapshot', cached_snapshot.actor)]:
    lp = []
    for temp in (.5, 1.):
        f = CN.F.Feat(0)
        x = f.observe(obs)
        decoded = actor._decode(x, f.aux, np.random.default_rng(3), temp, True)
        lp.append(float(decoded[4]))
    probs[name] = lp
result['role_temperature_logprobs'] = probs
assert probs['learner'][0] == probs['learner'][1]
assert probs['fresh_snapshot'][0] != probs['fresh_snapshot'][1]
assert probs['cached_snapshot'][0] != probs['cached_snapshot'][1]

# Additional terminal boundary cases, both seats, both record modes, all reward
# modes and SELF mirrors. Seed 14 changes the survivor's cash on the final step;
# original scoring must retain the PRE-step reward (10, not post-step 110).
cases = []
for seed in (11, 13, 14):
    original = original_finish(seed)
    truth = [float(s.reward) for s in original]
    for side in (0, 1):
        for record in (True, False):
            r = RO.play_batch({'adapter': 'cnet2', 'wkey': 'terminal-check',
                               'games': [('SELF', seed, side)], 'temp': .7,
                               'greedy': True, 'record': record, 'rng_seed': 23})
            assert r['ndec'] == 2, r['ndec']
            for j, g in enumerate(r['games']):
                p = side if j == 0 else 1 - side
                expected = truth[p] - truth[1 - p]
                assert g['steps'] == 1 and g['term'] and g['margin'] == expected
                if record:
                    assert np.isclose(g['dmargin'].sum(), expected)
                    assert np.isclose(g['dworth'].sum(), expected)
                    for kind in ('dense', 'terminal', 'worth'):
                        args = types.SimpleNamespace(reward=kind, c_dep=0., c_lost=0., shape_iters=200,
                                                     rscale=1., win_bonus=.1)
                        reward = TR.game_rewards(g, args, 1)
                        assert np.isclose(reward.sum(), expected + .1 * np.sign(expected))
            cases.append(dict(seed=seed, side=side, record=record, original_rewards=truth,
                              rollout_margin=r['games'][0]['margin'], ndec=r['ndec'],
                              post_cash=[r['games'][0]['my'], r['games'][0]['op']]))
result['terminal_boundary_cases'] = cases

# Mixed early/full SNAP games, both same-path and separate-path opponents. Only
# live states should reach either actor; the returned ndec counts learner states.
mixed = []
for same_path in (False, True):
    RO._W_CACHE.clear()
    wkey = 'mixed-primary'
    oppkey = wkey if same_path else 'mixed-snapshot'
    r = RO.play_batch({'adapter': 'cnet2', 'wkey': wkey,
                       'games': [(f'SNAP:cnet2:{oppkey}', 11, 0), (f'SNAP:cnet2:{oppkey}', 12, 0)],
                       'temp': .7, 'greedy': True, 'record': True, 'rng_seed': 29})
    pa = RO._actor('cnet2', wkey, role='learner')
    sa = RO._actor('cnet2', oppkey, role='snap')
    assert pa is not sa and pa.temp_kind and not sa.temp_kind
    assert r['ndec'] == sum(len(g['recs']) for g in r['games']) == 300
    assert len(pa.obs_calls) == len(sa.obs_calls) == 300
    mixed.append(dict(same_path=same_path, lengths=[len(g['recs']) for g in r['games']],
                      primary_inferences=len(pa.obs_calls), snapshot_inferences=len(sa.obs_calls),
                      separate_actor_objects=pa is not sa))
result['mixed_snapshot_batches'] = mixed


class OneBaseRecruitPolicy(FixedPolicy):
    def __init__(self):
        self.bases_seen = 0

    def lin(self, x, name):
        if name == 'bhead':
            self.bases_seen += 1
            if self.bases_seen % 2:
                return np.array([0., 10., 0.])
        return super().lin(x, name)


# A two-transition extinction produced by legal actions from the normal start:
# recruit twice on the same base; equal-cargo workers collide on step 2.
RO._W_CACHE.clear()
pa = RO._actor('cnet2', 'recruiting-primary', role='learner')
pa.pol = OneBaseRecruitPolicy()
RO.load_bot = lambda path: (lambda obs, cfg: {})
r = RO.play_batch({'adapter': 'cnet2', 'wkey': 'recruiting-primary',
                   'games': [('noop.py', 17, 0)], 'temp': .7,
                   'greedy': True, 'record': True, 'rng_seed': 31})
g = r['games'][0]
original = original_finish(17, turns=2, recruit=True)
truth = float(original[0].reward - original[1].reward)
assert g['term'] and g['steps'] == 2 and r['ndec'] == 2
assert g['margin'] == truth and np.isclose(g['dmargin'].sum(), truth)
assert np.isclose(g['dworth'].sum(), truth)
result['legal_action_two_step_extinction'] = dict(
    generated_map_seed=42,
    original_rewards=[float(s.reward) for s in original], original_margin=truth,
    rollout_margin=g['margin'], steps=g['steps'], ndec=r['ndec'],
    dense_increments=g['dmargin'].tolist(), worth_increments=g['dworth'].tolist())
result['reviewed_commit'] = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
result['verdict'] = {'F01': 'PASS', 'F06': 'PASS', 'S5a': 'PASS', 'remaining_confirmed_bugs': [],
                     'scope': 'Isolated CPU probes of original environment, actual rollout, actor cache/decoder, resume block, packing and GAE; no training.'}
print(json.dumps(result, indent=2))

"""Read-only R01-R04 regression checks on real resume/rollout/decoder functions.
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
        super().reset(full_carbon=[0.] * 225)
        if seed == 11:
            # A constructed elimination-boundary position: side 0 has only 20
            # cash, side 1 has 10 cash plus a live collector. No actions change
            # it. We do not prove reachability through a complete game sequence.
            obs = self.full_observation()
            obs['players'][0][0] = 20.
            obs['players'][1][0] = 10.
            pos = next(iter(obs['players'][1][1].values()))
            obs['players'][1][2] = {'player-1-worker-0': [pos, 0., 'COLLECTOR']}
            self.load_observation(obs)
        return self


# The actual original interpreter establishes the authoritative score.
origin = FixtureSim().reset(seed=11).full_observation()
states = []
for p in range(2):
    obs = copy.deepcopy(origin)
    obs['full_carbon'] = obs['carbon'][:]
    obs['player'] = p
    states.append(utils.structify(dict(observation=obs, action={}, status='ACTIVE',
                                      reward=origin['players'][p][0])))
idgen.reset()
original = rules.compute_next_board(states, types.SimpleNamespace(configuration=utils.structify(DEFAULT_CONFIG)))
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
result['elimination_reward_mismatch'] = dict(
    original_rewards=[float(s.reward) for s in original], original_margin=original_margin,
    rollout_margin=games[0]['margin'], rollout_win=games[0]['margin'] > 0,
    reward_modes=reward_cases)
assert original_margin < 0 < games[0]['margin']

# R04: fresh snapshot gets its own temperature, but a previously-primary
# cached actor retains True when requested later as a snapshot (same file).
primary = RO._actor('cnet2', 'tiny-A')
fresh = RO._actor('cnet2', 'tiny-B')
cached_snapshot = RO.make_opponent('SNAP:cnet2:tiny-A', 1, 9)
result['actor_roles'] = dict(primary_temp_kind=primary.temp_kind,
                             fresh_snapshot_temp_kind=fresh.temp_kind,
                             cached_snapshot_is_primary=cached_snapshot.actor is primary,
                             cached_snapshot_temp_kind=cached_snapshot.actor.temp_kind)
assert primary.temp_kind and not fresh.temp_kind and cached_snapshot.actor.temp_kind

# Use the real decoder to prove fresh snapshot probabilities depend on the
# caller temperature, learner probabilities don't, and the cached snapshot leaks.
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
assert probs['cached_snapshot'][0] == probs['cached_snapshot'][1]
print(json.dumps(result, indent=2))

"""Public-observation proxy worlds for the six declared maps. No search opponent or trainer launches here."""
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT / "code"))

from eval_v53 import CFG, EPISODE_STEPS, _extinct
from fastsim import FastCarbon
from map_identity import terrain_sha256

OPPONENTS = {
    "economy": ("policy_v0_5", "4d435035e8e274c5c4d49dfcc71ca293c6b4989ebadc974796a90837f9f569c1"),
    "raid": ("policy_raid_v0_1", "a569594a78e0ed5632aa7921399a9e906f2ddbe5b237529d9c0ffb6b1ec5bdc5"),
}


def check_seed(seed):
    declaration = json.loads((HERE / "map_declaration.json").read_bytes())
    if type(seed) is not int or seed not in declaration["training"] + declaration["new_proxy_development"]:
        raise ValueError("world is outside the six declared nonformal maps")
    if seed in declaration["new_proxy_development"]:
        clearance = json.loads((HERE / "map_clearance.json").read_bytes())
        if not clearance["eligible"] or str(seed) not in clearance["development"]:
            raise ValueError("proxy development terrain has no positive preflight")


def outcomes(scores, transitions, terrain, extinct):
    if len(scores) != 2 or not all(math.isfinite(n) for n in scores):
        raise ValueError("invalid official physical-seat scores")
    margins = [float(scores[0] - scores[1]), float(scores[1] - scores[0])]
    wdl = [1. if m > 0 else .5 if m == 0 else 0. for m in margins]
    # Same sigmoid as rl2_ppo_train.terminal_u, stable for very large absolute margins.
    def sigmoid(z):
        e = math.exp(-abs(z))
        return 1. / (1. + e) if z >= 0 else e / (1. + e)
    return dict(scores=list(scores), margin_by_seat=margins, wdl_by_seat=wdl,
                u_by_seat=[.5 * w + .5 * sigmoid(m / 1000.) for w, m in zip(wdl, margins)],
                result_by_seat=["win" if m > 0 else "loss" if m < 0 else "draw" for m in margins],
                steps=transitions, terrain_sha256=terrain, extinct_by_seat=list(extinct),
                termination="extinction" if any(extinct) else "official_episode_limit")


class World:
    """Advance simultaneous commands. deepcopy() preserves board, hidden environment state and ID counters."""
    def __init__(self, seed):
        check_seed(seed)
        self._sim = FastCarbon(CFG).reset(seed=seed)
        self.terrain_sha256 = terrain_sha256(self._sim.carbon)
        registry = json.loads((HERE / "map_declaration.json").read_bytes())
        if seed in registry["new_proxy_development"]:
            expected = json.loads((HERE / "map_clearance.json").read_bytes())["development"][str(seed)]["terrain"]
            if self.terrain_sha256 != expected:
                raise ValueError("locally generated proxy development board differs from preregistered terrain")
        self.terminal = None

    @property
    def step(self):
        return self._sim.step_count

    @property
    def config(self):
        return dict(CFG)

    def obs(self, seat):
        if self.terminal is not None or type(seat) is not int or seat not in (0, 1):
            raise ValueError("a current physical-seat decision is required")
        return self._sim.obs(seat)

    def advance(self, actions_by_seat):
        if self.terminal is not None:
            raise ValueError("cannot advance beyond the first official terminal")
        if not isinstance(actions_by_seat, (list, tuple)) or len(actions_by_seat) != 2:
            raise ValueError("exactly two physical-seat command dictionaries are required")
        sim = self._sim
        for seat, commands in enumerate(actions_by_seat):
            workers = sim.player_workers[seat]
            bases = {uid for uid, _ in sim.bases[seat]}
            if not isinstance(commands, dict) or any(
                    not ((uid in workers and move in (None, "UP", "RIGHT", "DOWN", "LEFT")) or
                         (uid in bases and move in (None, "RECCOLLECTOR", "RECPLANTER")))
                    for uid, move in commands.items()):
                raise ValueError("invalid physical-seat unit or command")
        pre_cash = list(sim.cash)
        sim.step(*actions_by_seat)
        extinct = [False, False]
        if any(not sim.player_workers[p] and not sim.player_trees[p] for p in (0, 1)):
            post = sim.get_state()
            extinct = [_extinct(post, p) for p in (0, 1)]
        if any(extinct) or self.step == EPISODE_STEPS - 1:
            cash = pre_cash if any(extinct) else sim.cash
            scores = [self.step - EPISODE_STEPS - 1 if extinct[p] else cash[p] for p in (0, 1)]
            self.terminal = outcomes(scores, self.step, self.terrain_sha256, extinct)
        return self.terminal


class CheapOpponent:
    """Every instance owns the rule strategy's full belief, opponent model and route caches."""
    def __init__(self, name):
        module_name, expected_sha = OPPONENTS[name]
        module = importlib.import_module(module_name)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected_sha:
            raise ValueError("research opponent changed")
        self.policy = module.Policy()

    def __call__(self, obs, config):
        return self.policy.act(obs, config)


def make_opponent(name):
    return CheapOpponent(name)


def replay_official(seed, actions_by_step):
    """Score an already selected joint-action tape with the original interpreter, without calling a competitor."""
    check_seed(seed)
    from official_game_replay import make
    # The original map generator uses global NumPy state. Preserve callers' sampling streams.
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        np.random.seed(seed)
        env = make("carbon", configuration={**CFG, "randomSeed": seed})
        terrain = terrain_sha256(env.states[0].observation.full_carbon)
        n = 0
        for actions in actions_by_step:
            if env.done:
                raise ValueError("trailing joint action after official terminal")
            env.step(actions)
            n += 1
        if not env.done or n == 0 or env.states[0].observation.step != n or [s.status for s in env.states] != ["DONE", "DONE"]:
            raise ValueError("missing complete ordinary two-seat terminal")
        info = env.states[0].observation.players
        extinct = [not workers and not trees and (not bases or cash < min(CFG["recCollectorCost"], CFG["recPlanterCost"]))
                   for cash, bases, workers, trees in info]
        return outcomes([s.reward for s in env.states], n, terrain, extinct)
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)

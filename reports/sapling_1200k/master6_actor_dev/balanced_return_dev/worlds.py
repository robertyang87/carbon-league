"""Nonformal worlds with separate, terrain-bound screen, train, and development pools."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECL = HERE / "maps.json"
PRIOR = HERE.parent / "blended_policy_return_dev/world.py"
PRIOR_SHA = "f914f793688b264196e3120d544d2466b94ba2234d20357bf45b15b160e851b3"
SPLITS = ("screen", "training", "development")
if hashlib.sha256(PRIOR.read_bytes()).hexdigest() != PRIOR_SHA:
    raise ValueError("the verified two-engine world implementation changed")
# Load a private module so its old registration hook can be rebound without changing any other experiment.
_spec = importlib.util.spec_from_file_location("balanced_return_private_world_primitives", PRIOR)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
CFG = _base.CFG
make_opponent = _base.make_opponent


def registry():
    plan = json.loads(DECL.read_bytes())
    seeds = [s for key in SPLITS for s in plan[key]]
    historic = plan["historical_maps"]
    if (plan["formal"] is not False or plan["physical_seats"] != [0, 1]
            or [len(plan[key]) for key in SPLITS] != [2, 8, 2]
            or len(seeds) != len(set(seeds)) or any(type(s) is not int for s in seeds)
            or set(map(int, historic)) != set(plan["screen"] + plan["training"])
            or len({r["terrain_sha256"] for r in historic.values()}) != len(historic)):
        raise ValueError("malformed or overlapping nonformal map registry")
    return plan


def split_for(seed):
    if type(seed) is not int:
        raise ValueError("map seed must be an integer")
    plan = registry()
    for key in SPLITS:
        if seed in plan[key]:
            return key
    raise ValueError("map outside this experiment")


def locked_terrain(seed, purpose=None):
    role = split_for(seed)
    if purpose is not None and purpose != role:
        raise ValueError(f"map belongs to {role}, requested {purpose}")
    if role != "development":
        return registry()["historical_maps"][str(seed)]["terrain_sha256"]
    clearance = json.loads((HERE / "map_clearance.json").read_bytes())
    if (clearance["declaration_sha256"] != hashlib.sha256(DECL.read_bytes()).hexdigest()
            or clearance["eligible"] is not True):
        raise ValueError("unverified development declaration")
    return clearance["development"][str(seed)]["terrain_sha256"]


_base.check_seed = locked_terrain


class World(_base.World):
    """step, obs(seat), advance([seat0,seat1]), terminal, config and deepcopy share the proven semantics."""
    def __init__(self, seed, purpose=None):
        expected = locked_terrain(seed, purpose)
        self._sim = _base.FastCarbon(CFG).reset(seed=seed)
        self.terrain_sha256 = _base.terrain_sha256(self._sim.carbon)
        if self.terrain_sha256 != expected:
            raise ValueError("initial board differs from the registered physical terrain")
        self.terminal = None


def replay_official(seed, actions_by_step, purpose=None):
    expected = locked_terrain(seed, purpose)
    result = _base.replay_official(seed, actions_by_step)
    if result["terrain_sha256"] != expected:
        raise ValueError("original interpreter produced a different physical board")
    return result

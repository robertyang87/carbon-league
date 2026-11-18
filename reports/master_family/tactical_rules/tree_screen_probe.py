"""Compare immediate tree defense and immediate banking from identical public positions.

Run from the repository root:
  python reports/master_family/tactical_rules/tree_screen_probe.py --out /tmp/tree_screen.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from defense import ACTIONS, DefenseBoard

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "code"))
from fastsim import FastCarbon

CASES = (
    ("empty_defender", 0, 9, 80, False),
    ("loaded_defender", 50, 9, 80, False),
    ("one_income_step_left", 50, 49, 80, False),
    ("original_tree_expires_now", 50, 50, 80, False),
    ("intruder_cannot_pay", 50, 9, 10, False),
    ("intruder_escorted_by_empty_collector", 50, 9, 80, True),
)


def root_position(side, cargo, age, enemy_cash, escort):
    # A reachable three-cell fork: home -- our collector -- original own tree/hostile planter.
    sim = FastCarbon().reset(full_carbon=[0.0] * 225, cash=0)
    screen, bank = ("RIGHT", "LEFT") if side == 0 else ("LEFT", "RIGHT")
    home = sim.bases[side][0][1]
    between = sim._move_nbr[screen][home]
    target = sim._move_nbr[screen][between]
    behind = sim._move_nbr[screen][target]
    assert not {between, target, behind}.intersection(sim.base_at)
    obs = sim.full_observation()
    obs["step"] = 80
    me, enemy = obs["players"][side], obs["players"][1-side]
    enemy[0] = enemy_cash
    defender_id = f"player-{side}-worker-0"
    intruder_id = f"player-{1-side}-worker-0"
    escort_id = f"player-{1-side}-worker-1"
    tree_id = f"player-{side}-tree-0"
    me[2][defender_id] = [between, cargo, "COLLECTOR"]
    enemy[2][intruder_id] = [target, 0, "PLANTER"]
    me[3][tree_id] = [target, age]
    obs["trees"][tree_id] = [None, 0.]
    for neighbor in sim._nbr4[target]:
        obs["carbon"][neighbor] = 25.
    if escort:
        enemy[2][escort_id] = [behind, 0, "COLLECTOR"]
    sim.load_observation(obs)
    return sim, tree_id, target, defender_id, intruder_id, escort_id, screen, bank


def run_case(case, cargo, age, enemy_cash, escort, side):
    sim, tree_id, target, defender_id, intruder_id, escort_id, screen, bank = root_position(
        side, cargo, age, enemy_cash, escort)
    observation = sim.obs(side)
    board = DefenseBoard(observation, sim.config)
    risk = board.assess(defender_id, screen)
    branches = []
    for reply in ACTIONS:
        # Pick enemy actions before the own alternatives. Five planter actions are exhausted.
        # In the escorted case the enemy collector always advances to the initial tree cell.
        enemy_actions = {} if reply == "NONE" else {intruder_id: reply}
        if escort:
            enemy_actions[escort_id] = bank
        outcomes = {}
        for name, action in (("screen", screen), ("bank", bank)):
            trial = FastCarbon().set_state(sim.get_state())
            actions = [{}, {}]
            actions[side] = {defender_id: action}
            actions[1-side] = dict(enemy_actions)
            trial.step(*actions)
            survivor = trial.workers.get(defender_id)
            original = trial.trees.get(tree_id)
            on_cell = [tree for tree in trial.trees.values() if tree.pos == target]
            # The only own cash channels in these roots are normal deposit and tree absorption.
            # No own recruits, planters, spending, enemy cargo donations, or mixed-collision tie.
            tree_income = [sum(trial.trees_dict[tid][1] for tid in trial.player_trees[p]) for p in (side, 1-side)]
            banked = round(trial.cash[side] - sim.cash[side] - tree_income[0], 2)
            assert banked == (cargo if action == bank else 0)
            assert not any(entry[0] == defender_id for entry in trial.last_collisions) or not banked
            if banked:
                assert survivor and survivor.pos == sim.bases[side][0][1] and survivor.carbon == 0
            outcomes[name] = {
                "cash_own_enemy": [trial.cash[side], trial.cash[1-side]],
                "cash_delta_own_enemy": [trial.cash[side] - sim.cash[side], trial.cash[1-side] - enemy_cash],
                "original_tree_owner": None if not original else ("own" if original.player == side else "enemy"),
                "original_tree_age_after": None if not original else original.age,
                "next_cleanup_removes_original": bool(original and original.age >= trial.config["treeLifespan"]),
                "tree_cell_owner": None if not on_cell else ("own" if on_cell[0].player == side else "enemy"),
                "banked_cargo": banked,
                "tree_absorption_own_enemy": tree_income,
                "defender_alive": survivor is not None,
                "defender_post_cargo": None if not survivor else survivor.carbon,
                "intruder_alive": intruder_id in trial.workers,
                "collisions": trial.last_collisions,
            }
        branches.append({"enemy_planter_action": reply, **outcomes})
    return {
        "case": case, "seat": side, "cargo": cargo, "initial_tree_age": age,
        "initial_enemy_cash": enemy_cash, "enemy_escort": escort, "tree_cell": target,
        "screen_flag": risk.can_screen_tree, "possible_loss_on_screen": risk.collision_loss_possible,
        "branches": branches,
    }


def check(rows):
    for row in rows:
        branches = row["branches"]
        stay = branches[0]
        assert stay["enemy_planter_action"] == "NONE"
        assert len(branches) == 5
        for branch in branches:
            assert branch["bank"]["banked_cargo"] - branch["screen"]["banked_cargo"] == row["cargo"]
        if row["case"] == "original_tree_expires_now":
            assert not row["screen_flag"]
            assert all(b[arm]["original_tree_owner"] is None for b in branches for arm in ("screen", "bank"))
        elif row["case"] == "intruder_escorted_by_empty_collector":
            assert not row["screen_flag"] and row["possible_loss_on_screen"]
            assert stay["screen"]["defender_alive"] is False
        elif row["case"] == "intruder_cannot_pay":
            assert stay["bank"]["original_tree_owner"] == "own"
        else:
            assert row["screen_flag"]
            assert stay["bank"]["original_tree_owner"] == "enemy"
            assert stay["screen"]["original_tree_owner"] == "own" and not stay["screen"]["intruder_alive"]
            assert all(b[arm]["original_tree_owner"] == "own" for b in branches[1:] for arm in ("screen", "bank"))
        if row["case"] == "one_income_step_left":
            assert stay["screen"]["next_cleanup_removes_original"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    rows = [run_case(*case, side) for case in CASES for side in (0, 1)]
    check(rows)
    result = {
        "scope": "Mechanism and one-turn opportunity cost; six hand-constructed roots mirrored by seat, "
                 "five simultaneous enemy planter responses crossed with two own choices. Branches have no "
                 "probability weights. No next-turn game value, search labels or policy-strength claim.",
        "conditions": "Standard rules, all four tree-adjacent cardinal cells start at 25 carbon; all other cells "
                      "empty. The tree can earn in the same step as seizure. Only the defended side decides "
                      "between screen and bank. Banked cargo is liquid cash this turn; cargo may survive a screen.",
        "sources_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (Path(__file__).resolve(), HERE / "defense.py", ROOT / "code/fastsim.py")},
        "roots": len(rows), "transitions": len(rows) * len(ACTIONS) * 2, "rows": rows,
    }
    with args.out.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    main()

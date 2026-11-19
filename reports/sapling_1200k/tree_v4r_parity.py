"""Check V4r's common decoder prefix and settle cash fixtures with real FastCarbon."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "data/weights/sapling_1200k/tree_v4r_agent.py"
sys.path.insert(0, str(ROOT / "code"))
from fastsim import FastCarbon, DEFAULT_CONFIG


def load():
    spec = importlib.util.spec_from_file_location("sapling_v4r", PACK)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def actors(p, cash, trees=()):
    sim = FastCarbon(DEFAULT_CONFIG).reset(seed=620147302)
    st = sim.get_state()
    st["workers"] = {}
    for i, (pos, role) in enumerate(((1, "COLLECTOR"), (133, "COLLECTOR"),
                                      (39, "PLANTER"), (126, "PLANTER"))):
        st["workers"][f"player-{p}-worker-{i}"] = dict(pos=pos, type=role, player=p, carbon=0)
    st["wid_counter"] = [4, 4]
    st["trees"] = {f"player-{owner}-tree-{i}": dict(pos=pos, player=owner, age=age)
                   for i, (owner, pos, age) in enumerate(trees)}
    st["tid_counter"] = [len(trees), len(trees)]
    st["trees_dict"] = {tid: [None, 0] for tid in st["trees"]}
    st["cash"][p] = cash
    sim.set_state(st)
    return sim


MOVES = np.ones(5, bool)
BASE = np.array([1, 1, 1, 0, 0], bool)
KEEP = np.array([9., 5., 0., 0., 0.])
COLLECT = np.array([0., 9., 1., 0., 0.])
PLANT = np.array([0., 1., 9., 0., 0.])


def fixtures(mod, seat):
    checked = {}
    canonical = mod._F.CANON if seat else mod._F.GRID
    for name, cash, new_cells, captures, tree in (
        ("two_stays_49", 49, 0, 0, ()),
        ("two_stays_50", 50, 1, 0, ()),
        ("two_stays_70", 70, 2, 0, ()),
        ("seize_boundary", 50, 0, 1, ((1-seat, 39, 8),)),
        ("expired_enemy_is_fresh_plant", 50, 1, 0, ((1-seat, 39, 50),)),
    ):
        sim = actors(seat, cash, tree)
        obs = sim.obs(seat)
        budget = mod._budget(obs, sim.config)
        commands = {}
        for i, rawpos in enumerate((39, 126), 2):
            a = mod._planter_action(budget, int(canonical[rawpos]), KEEP, KEEP, MOVES)
            if a:
                commands[f"player-{seat}-worker-{i}"] = mod._F.MOVES[mod._F.CANON_MOVE[a] if seat else a]
        old = {k: t.player for k,t in sim.trees.items()}
        sim.step(*((commands, {}) if seat == 0 else ({}, commands)))
        actual_new = sum(t.player == seat and k not in old for k,t in sim.trees.items())
        flips = sum(t.player == seat and k in old and old[k] != seat for k,t in sim.trees.items())
        assert (actual_new, flips) == (new_cells, captures), (name, actual_new, flips)
        assert sim.cash[seat] >= 30 and budget["unavoidable"] == 0
        checked[name] = dict(new_trees=actual_new, captures=flips, explicit_moves=len(commands))
    for cash, expected in ((79, (0,1)), (80, (2,0)), (110, (2,1))):
        sim = actors(seat, cash)
        obs = sim.obs(seat)
        obs["players"][seat][2] = {uid: w for uid,w in obs["players"][seat][2].items() if w[2]=="COLLECTOR"}
        budget = mod._budget(obs, sim.config)
        decisions = (mod._choose_base(budget, KEEP, PLANT, BASE), mod._choose_base(budget, COLLECT, PLANT, BASE))
        assert decisions == expected, (cash, decisions)
        checked["two_bases_cash_"+str(cash)] = decisions
    # Two claims on the same unmoved planter cell carry only one possible charge.
    b = mod._budget(actors(seat, 50).obs(seat), DEFAULT_CONFIG)
    assert [mod._planter_action(b, int(canonical[39]), KEEP, KEEP, MOVES) for _ in range(2)] == [0,0]
    assert b["held_tree_cost"] == 20
    # A restricted caller offering only STAY cannot pretend it prevented a charge.
    b = mod._budget(actors(seat, 20).obs(seat), DEFAULT_CONFIG)
    assert mod._planter_action(b, int(canonical[39]), KEEP, KEEP, np.array([1,0,0,0,0],bool)) == 0
    assert b["unavoidable"] == 1 and b["redirected"] == 0 and b["held_tree_cost"] == 20
    checked["unavoidable_forced_stay"] = dict(charged=b["held_tree_cost"], unavoidable=1)
    # With our own trees above the protected count, exact market formula applies.
    b = mod._budget(actors(seat, 100, ((seat,33,15), (1-seat,55,49), (seat,56,50))).obs(seat), DEFAULT_CONFIG)
    assert abs(b["plant"] - (20 + 5 * 1.235**2)) < 1.e-10
    return checked


def joint_prefix(mod, seat):
    sim = actors(seat, 120)
    public = mod._F.Feat(seat)
    slots = dict(collector=0, planter=0, base=0)
    for _ in range(24):
        obs = sim.obs(seat)
        actions = mod.agent(obs, sim.config)
        x = public.observe(obs)
        labels = public.labels(actions)
        v3, v4 = [net.act(x, public.aux, teacher=2, forced=labels)[1] for net in mod._NETS]
        b = mod._budget(obs, sim.config)
        prefix = mod._F.Prefix(x, public.aux)
        for s in np.flatnonzero(x["tok_mask"][:mod._F.OWN]):
            _, _, legal = prefix.feats(s)
            role = int(x["tok_kind"][s])
            if role == 0:
                expected = int(v3[s].argmax())
            elif role == 1:
                expected = mod._planter_action(b, int(x["tok_pos"][s]), v3[s], v4[s], legal)
            else:
                expected = mod._choose_base(b, v3[s], v4[s], legal)
            assert expected == labels[s] and legal[expected], (seat, s, expected, labels[s])
            prefix.advance(s, expected)
            slots[("collector","planter","base")[role]] += 1
        assert mod._LAST_BUDGET[seat]["unavoidable"] == 0
        sim.step(*((actions,{}) if seat == 0 else ({},actions)))
    return slots


if __name__ == "__main__":
    pack = load()
    outputs = {seat: dict(actual_rule_fixtures=fixtures(pack, seat), common_prefix_slots=joint_prefix(pack, seat))
               for seat in (0,1)}
    print(json.dumps(dict(pack_sha256=hashlib.sha256(PACK.read_bytes()).hexdigest(), checks=outputs), indent=2))

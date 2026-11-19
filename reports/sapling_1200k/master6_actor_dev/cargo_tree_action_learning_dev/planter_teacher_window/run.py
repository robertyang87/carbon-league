"""Original teacher selects planter slots on its own public seat for 16 midgame turns."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import fcntl
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"
os.environ["CARBON_DEBUG"] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "candidate.py"
ACTOR = HERE.parent.parent
sys.path.insert(0, str(ACTOR / "elite_role_mix_dev/conditional"))
import responses as R

PLAN = R.load(HERE / "protocol.json")
DIGEST = "9a8f1a551d14059ad6d0e39097a5aadc4815ec006dd1ebf3a192dd73d468afbe"
START, END = PLAN["first"], PLAN["last_inclusive"] + 1


def archive(case):
    raw_path = R.ROOT / case["path"]
    receipt = R.ROOT / case["receipt_path"]
    assert R.sha(raw_path) == case["sha256"] and R.sha(receipt) == case["receipt_sha256"]
    matched = [g for run in R.load(receipt)["runs"].values() for g in run["games"]
               if (g["seed"], g["seat"], g["transcript_sha256"]) == (case["seed"], case["seat"], case["sha256"])]
    assert len(matched) == 1 and matched[0]["statuses"] == ["DONE", "DONE"] and matched[0]["map_sha256"] == case["terrain"]
    raw = R.load(raw_path)
    assert not raw["errors"] and raw["initial_terrain_u8_sha256"] == case["terrain"] and raw["candidate_seat"] == case["seat"]
    assert raw["official_score_by_seat"] == matched[0]["official_reward_by_seat"]
    return raw


def identity():
    assert __debug__ and R.sha(HERE / "protocol.json") == DIGEST
    assert R.sha(PARENT) == PLAN["incumbent"] and R.sha(R.live.SOURCE) == PLAN["original"]
    assert hashlib.sha256(subprocess.check_output(["git","-C",str(R.ROOT),"show","HEAD:sources/master3s2/master3s2.py"])).hexdigest() == PLAN["original"]
    return dict(protocol=DIGEST, program=R.sha(Path(__file__)), parent=R.sha(PARENT), original=R.sha(R.live.SOURCE))


def required_spend(view, cfg, action):
    """Conservative public quote, before collisions/deposits, for this entire own order."""
    side = view["player"]
    players = view["players"]
    cash, bases, workers, own_trees = players[side]
    base_cells = {cell for player in players for cell in player[1].values()}
    alive = {cell: seat for seat, player in enumerate(players) for cell, age in player[3].values()
             if age < cfg["treeLifespan"]}
    price = cfg["plantCost"]
    if len(own_trees) > cfg["playerTreeProtectiveNumber"]:
        price += cfg["plantCostInflationRatio"] * cfg["plantCostInflationBase"] ** len(alive)
    kinds = Counter(w[2] for player in players for w in player[2].values())
    total = 0.
    for uid in bases:
        cmd = action.get(uid)
        if cmd in ("RECCOLLECTOR", "RECPLANTER"):
            k = "Collector" if cmd == "RECCOLLECTOR" else "Planter"
            total += cfg["rec"+k+"Cost"] + kinds[k.upper()]*cfg["rec"+k+"IncreaseCost"]
    for uid, (pos, _, role) in workers.items():
        if role == "PLANTER" and uid not in action and pos not in base_cells:
            if pos not in alive:
                total += price
            elif alive[pos] != side:
                total += cfg["seizeCost"]
    return total


class Intervention:
    """Reuses the incumbent's production GRU and guard. Only a legal planter choice is forced."""
    def __init__(self, model):
        self.model = model
        self.production = model._return_decode
        marker = "action = int(np.argmax(np.where(legal, combined, -np.inf)))"
        source = inspect.getsource(self.production)
        assert source.count(marker) == 1
        scope = dict(vars(model), override=self.override)
        exec(source.replace(marker, marker+"\n        action = override(x, slot, legal, action)"), scope)
        self.decode = scope["_return_decode"]
        self.wanted = {}

    def override(self, x, slot, legal, chosen):
        if slot in self.wanted:
            assert slot < 10 and x["tok_kind"][slot] == 1 and legal[self.wanted[slot]]
            return self.wanted[slot]
        return chosen

    def select(self, view, cfg, teacher):
        m = self.model
        feat = m._FEAT
        x = feat.observe(view)
        normal = m._mixed_actions(x, feat.aux)
        physical_parent = feat.to_actions(normal)
        teacher_labels = feat.labels(teacher)
        planters = {u for u, w in view["players"][view["player"]][2].items() if w[2] == "PLANTER"}
        self.wanted = {i: int(teacher_labels[i]) for i, u in enumerate(feat.own_uid) if u in planters}
        assert all(teacher.get(u, "NONE") in ("UP","RIGHT","DOWN","LEFT","NONE") for u in planters)
        self.decode.__globals__["_COUNTS"] = m._COUNTS
        m._return_decode = self.decode
        try:
            alternative = m._mixed_actions(x, feat.aux)
        finally:
            m._return_decode = self.production
        assert all(alternative[i] == a for i, a in self.wanted.items())
        proposed = feat.to_actions(alternative)
        quoted = required_spend(view, cfg, proposed)
        rejects = []
        if m._return_conflicts(x, alternative) - m._return_conflicts(x, normal):
            rejects.append("new_friendly_overlap")
        if quoted > max(view["players"][view["player"]][0], required_spend(view, cfg, physical_parent)) + 1e-7:
            rejects.append("new_unfunded_orders")
        result = physical_parent if rejects else proposed
        return physical_parent, result, dict(parent=physical_parent, teacher_planter={u:teacher.get(u,"NONE") for u in sorted(planters)},
            proposed=proposed, eligible=result, rejected=rejects, changed_planters=[u for u in sorted(planters)
            if physical_parent.get(u) != result.get(u)], quote=quoted)


def history(module, view, actual):
    module._ST["tr"].step(view["players"], view["step"], view["carbon"])
    module._ST["tr"].set_actions(view["players"], actual)
    module._ST["prev"] = view["players"]


def original(case, role, seat, warm):
    m = R.live.module(R.live.SOURCE, "studied_original_"+role)
    assert m._SEARCH_OK and m._RX and m._TEMP == 1 and m._S_ABORT == .58 and m._SEED is None
    seed = int(hashlib.sha256(f"planter-window-v1|{role}|{case['terrain']}|{case['seat']}".encode()).hexdigest()[:8],16)
    m._S_COST[:] = warm
    m._ST.clear()
    m._ST.update(tr=m.BeliefTracker(m._Bel,seat),prev=None,rng=np.random.RandomState(seed))
    return m, seed


def audit(raw, tape, end_cash):
    sim = R.FastCarbon(raw["game_configuration"]).reset(seed=raw["seed"])
    assert list(sim.carbon) == raw["initial_full_carbon"]
    for joint in raw["actions"][:START]:
        sim.step(*joint)
    sim.cash = R.CashEvents(sim.cash)
    initial = list(sim.cash)
    ledger = {period:[Counter(),Counter()] for period in ("window","to199","full")}
    steps = []
    for joint in tape:
        t = sim.step_count
        sim.cash.events.clear()
        sim.step(*joint)
        for kind, seat, amount, *meta in sim.cash.events:
            ev = {"ground_rent":amount-meta[2], "cargo_tax":meta[2]} if kind == "tree_absorption" else {kind:amount}
            for phase in ["full"] + (["to199"] if t < 200 else []) + (["window"] if t < END else []):
                ledger[phase][seat].update(ev)
        steps.append(dict(t=t, cash=list(sim.cash), events=list(sim.cash.events)))
    assert np.allclose(end_cash, sim.cash, atol=1e-6, rtol=0)
    assert np.allclose([sum(d.values()) for d in ledger["full"]], np.asarray(end_cash)-initial, atol=1e-6, rtol=0)
    return dict(start=initial, end=end_cash, ledger=ledger, steps=steps)


def arm(job):
    case, label, cpu, warm = job
    os.sched_setaffinity(0,{cpu})
    dest = HERE/f"{case['seed']}_{case['seat']}_{label}.json.gz"
    binding = dict(**identity(), archive=case["sha256"], label=label, cpu=cpu, warm=warm)
    if dest.exists():
        old = R.load(dest)
        assert old["binding"] == binding and old["statuses"] == ["DONE","DONE"]
        return old
    raw = archive(case)
    seat = case["seat"]
    opp = 1-seat
    cfg = raw["game_configuration"]
    env = R.live.begin(raw)
    own = R.live.module(PARENT, "tested_incumbent")
    editor = Intervention(own)
    enemy, rng = original(case,"enemy",opp,warm)
    teacher, rng_teacher = original(case,"teacher",seat,warm)
    enemy_cl = R.native_clock.Recorder(enemy)
    teacher_cl = R.native_clock.Recorder(teacher)
    do_enemy = enemy_cl.wrap(enemy.agent)
    do_teacher = teacher_cl.wrap(teacher.agent)
    for t, joint in enumerate(raw["actions"][:START]):
        so, eo = R.live.public(env,seat), R.live.public(env,opp)
        history(teacher,so,joint[seat])
        history(enemy,eo,joint[opp])
        if t == 0: own.agent(so,cfg)
        else: own._FEAT.observe(so)
        env.step(joint)
    before = R.native_clock.host_snapshot(enemy)
    R.native_clock.validate_host(before,[cpu])
    started = time.monotonic()
    tape, decisions = [], []
    while not env.done:
        t = env.states[0].observation.step
        so = R.live.public(env,seat)
        eo = R.live.public(env,opp)
        if t < END:
            order = R.live.clean(do_teacher(so,cfg))
            base, modified, row = editor.select(so,cfg,order)
            actual = modified if label == "teacher" else base
            # Original updates its tracker for the full teacher team by default.
            # Only the actually submitted partial-team destinations should remain.
            teacher._ST["tr"].set_actions(so["players"],actual)
            decisions.append(dict(t=t, actual=actual, **row))
        else:
            actual = R.live.clean(own.agent(so,cfg))
        joint = [None,None]
        joint[seat] = actual
        joint[opp] = R.live.clean(do_enemy(eo,cfg))
        tape.append(joint)
        env.step(joint)
        if time.monotonic()-started > 270:
            R.write(HERE/f"{dest.stem}_unfinished.json.gz",dict(binding=binding,tape=tape,decisions=decisions))
            raise TimeoutError("270-second branch budget")
    elapsed = time.monotonic()-started
    statuses = [x.status for x in env.states]
    assert statuses == ["DONE","DONE"] and not enemy_cl.errors and not teacher_cl.errors
    assert len(tape) == len(enemy_cl.calls) and len(decisions) == len(teacher_cl.calls) == END-START
    after = R.native_clock.host_snapshot(enemy)
    R.native_clock.validate_host(after,[cpu])
    scores = [x.reward for x in env.states]
    cash = [p[0] for p in env.states[0].observation.players]
    result = dict(binding=binding,seed=case["seed"],seat=seat,statuses=statuses,scores=scores,margin=scores[seat]-scores[opp],
        tape=tape, decisions=decisions, enemy_calls=enemy_cl.calls,teacher_calls=teacher_cl.calls, enemy_seed=rng,teacher_seed=rng_teacher,
        clock=R.native_clock.summarize(enemy_cl.calls,enemy._S_ABORT,enemy._S_D2),teacher_clock=R.native_clock.summarize(teacher_cl.calls,teacher._S_ABORT,teacher._S_D2),
        wall=elapsed, before=before,after=after, cash=audit(raw,tape,cash))
    R.write(dest,result)
    return result


def preflight():
    bind = identity()
    rows = []
    for case in PLAN["cases"]:
        raw = archive(case)
        cfg = raw["game_configuration"]
        env = R.live.begin(raw)
        models = [R.live.module(PARENT,"check_"+label) for label in ("normal","edit")]
        editor = Intervention(models[1])
        s=case["seat"]
        for t, joint in enumerate(raw["actions"][:END]):
            view = R.live.public(env,s)
            if t == 0:
                for m in models: m.agent(view,cfg)
            elif t < START:
                for m in models: m._FEAT.observe(view)
            else:
                normal = R.live.clean(models[0].agent(view,cfg))
                base, redecoded, _ = editor.select(view,cfg,normal)
                assert normal == base == redecoded
            env.step(joint)
        rows.append(dict(seed=case["seed"],seat=s,teacher_equal_parent_identical_steps=END-START))
    R.write(HERE/"preflight.json",dict(binding=bind,checks=rows))
    print(rows,flush=True)


def play(i):
    assert R.load(HERE/"preflight.json")["binding"] == identity()
    case = PLAN["cases"][i]
    with R.LOCK.open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
        warmfile=HERE/f"warm_{case['seed']}.json"
        if not warmfile.exists():
            with ProcessPoolExecutor(1) as pool:
                R.write(warmfile,dict(binding=identity(),warm=pool.submit(R.native_warm).result()))
        warm=R.load(warmfile)
        assert warm["binding"]==identity()
        with ProcessPoolExecutor(2) as pool:
            a,b=pool.map(arm,[(case,"parent",i,warm["warm"]),(case,"teacher",1-i,warm["warm"])])
        s=case["seat"]
        assert a["enemy_seed"]==b["enemy_seed"] and a["teacher_seed"]==b["teacher_seed"]
        def diff(k,phase):
            def signed(z):
                d=z["cash"]["ledger"][phase]
                return d[s].get(k,0.)-d[1-s].get(k,0.)
            return signed(b)-signed(a)
        changes=[r for r in b["decisions"] if r["changed_planters"]]
        out=dict(seed=case["seed"],seat=s,parent_scores=a["scores"],teacher_scores=b["scores"],parent_margin=a["margin"],teacher_margin=b["margin"],
            delta_margin=b["margin"]-a["margin"],changed_planter_steps=len(changes),changed_planter_slots=sum(len(row["changed_planters"]) for row in changes),
            opponent_steps_different=sum(x[1-s]!=y[1-s] for x,y in zip(a["tape"],b["tape"])),
            opponent_first_order_same=a["tape"][0][1-s]==b["tape"][0][1-s],cash_delta_by_horizon={p:{k:diff(k,p)
            for d in (a,b) for side in d["cash"]["ledger"][p] for k in side} for p in ("window","to199","full")})
        R.write(HERE/f"pair_{case['seed']}.json",out)
        print({k:v for k,v in out.items() if k!="cash_delta_by_horizon"},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["preflight","pair0","pair1"])
    args=parser.parse_args()
    if args.stage=="preflight": preflight()
    else: play(int(args.stage[-1]))

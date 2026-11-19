"""Choose a registered economy contrast on the acting seat's physical team orders.

The normal 851c policy acts once. The alternative is built only from the public
state and the action-time features of that same call. No enemy order is accepted.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "live_curriculum/shared_enemy_root"))
from shared import play, runner, student_state
import proposals


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def focal_effect(obs, cfg, category, order_a, order_b, witness):
    """Return the proposed paid event and its arm from current public rules."""
    seat = obs["player"]
    own, foe = obs["players"][seat], obs["players"][1-seat]
    uid = witness["physical_context"]["uid"]
    if uid not in own[2]:
        return None
    pos, cargo, role = own[2][uid]
    commands = [cmd.get(uid, "NONE") for cmd in (order_a, order_b)]
    cash = float(own[0])
    if category == "plant_or_capture":
        if role != "PLANTER" or commands.count("NONE") != 1:
            return None
        if pos in (*own[1].values(), *foe[1].values()):
            return None
        trees = [{v[0]:v[1] for v in side[3].values()} for side in (own, foe)]
        if pos in trees[0]:
            return None
        if pos in trees[1]:
            kind, cost = "capture", float(cfg["seizeCost"])
        else:
            kind, cost = "plant", float(cfg["plantCost"])
            if len(trees[0]) > cfg["playerTreeProtectiveNumber"]:
                live_count = sum(age < cfg["treeLifespan"] for side in trees for age in side.values())
                cost += cfg["plantCostInflationRatio"] * cfg["plantCostInflationBase"] ** live_count
        if cash < cost:
            return None
        arm = commands.index("NONE")
        detail = {"cash":cash, "public_rule_price":cost, "enemy_tree_age":trees[1].get(pos)}
    elif category == "freight":
        if role != "COLLECTOR" or cargo <= cfg["smeltCost"]:
            return None
        width = int(cfg["size"])
        row,col = divmod(pos,width)
        delta = {"NONE":(0,0), "UP":(-1,0), "DOWN":(1,0), "LEFT":(0,-1), "RIGHT":(0,1)}
        dest = [((row+delta[a][0]) % width)*width+(col+delta[a][1]) % width for a in commands]
        on_home = [p in own[1].values() for p in dest]
        if sum(on_home) != 1:
            return None
        arm,kind = on_home.index(True),"deposit"
        detail = {"destinations":dest,"cargo":float(cargo),"smelt_cost":float(cfg["smeltCost"])}
    else:
        raise ValueError("unknown experimental category")
    return {"focal_uid":uid, "effect_arm":arm, "kind":kind, "position":pos,
            "physical_actions":commands, "public_conditions":detail}


def pick(catalog, own_public, cfg, real_a, category, native=None):
    """Select from the registered, already ranked witnesses versus genuine A."""
    menu, obs, order = catalog, own_public, real_a
    key = proposals.command_key
    weights = {key(item["command"]):item["probability"] for item in menu["orders"]}
    seen = {key(order)}
    skips = []
    for w in menu["classes"][category]["witnesses"]:
        alternative = w["command"]
        this_key = key(alternative)
        if this_key in seen:
            skips.append({"slot":w["slot"], "reason":"same_order"})
            continue
        seen.add(this_key)
        effect = focal_effect(obs,cfg,category,order,alternative,w)
        if effect is None:
            skips.append({"slot":w["slot"], "reason":"no_registered_physical_contrast"})
            continue
        native_p = native["p"][native["pick"]] if native is not None else None
        if (native_p is not None and not 0 < native_p <= 1) or not 0 < weights[this_key] <= 1:
            raise ValueError("missing native A or catalog B support")
        return {**effect, "options":[dict(order),dict(alternative)], "native_A":dict(order),
                "native_choice":native, "native_A_probability":native_p,
                "catalog_probabilities":[weights.get(key(order),0.), weights[this_key]],
                "catalog_baseline":menu["baseline"], "catalog_sha256":digest(menu),
                "category":category, "witness":copy.deepcopy(w), "skipped":skips,
                "conditional_selection_probability_given_A_and_menu":1.0}
    return None


def probe(student, obs, cfg, category):
    """Advance a warmed real 851c once; return its measured A and a supported contrast."""
    if obs["step"] not in proposals.STEPS or category not in ("plant_or_capture","freight"):
        raise ValueError("unsupported diagnostic turn or category")
    if "full_carbon" in obs:
        raise ValueError("a single public observation is required")
    mod = student.mod
    runtime = mod._mixed_actions
    captured = []

    def capture(x, aux):
        captured.append((x,aux))
        return runtime(x,aux)

    t,c = time.perf_counter(),time.process_time()
    mod._mixed_actions = capture
    try:
        own,native = runner.action(student,obs,cfg)
    finally:
        mod._mixed_actions = runtime
    if len(captured) != 1:
        raise ValueError("missing action-time network features")
    original_state = student_state.fingerprint(student)
    original_rng = play.student_history.behavior_rng_snapshot(student)
    # The encoder assigns a temporary trunk for each evaluated policy. Restore
    # the actual public-call trunks even on a failed proposal reconstruction.
    trunks = [net.trunk for net in mod._NETS]
    try:
        with play.audit_menu.collect.attached(student.modules):
            menu = proposals.build(mod,*captured[0],obs)
    finally:
        for net,trunk in zip(mod._NETS,trunks):
            net.trunk=trunk
    if original_state!=student_state.fingerprint(student) or original_rng!=play.student_history.behavior_rng_snapshot(student):
        raise ValueError("catalog construction altered natural 851c poststate")
    default = native["options"][0] if native is not None else own["order"]
    if menu["baseline"]!=default:
        raise ValueError("catalog is not attached to actual 851c decoder")
    selected=pick(menu,obs,cfg,own["order"],category,native)
    if selected is not None and native is None:
        # Unlike an offline call to pick(), the real RNG tap confirms a singleton.
        selected["native_A_probability"]=1.0
    if selected is not None:
        selected.update(parent_after_sha256=original_state, parent_rng_after_sha256=original_rng,
                        public_observation_sha256=hashlib.sha256(json.dumps(
                            obs,separators=(",",":"),allow_nan=False).encode()+b"\n").hexdigest())
    return {**own, "native_wall":own["wall"], "native_cpu":own["cpu"],
            "wall":time.perf_counter()-t,"cpu":time.process_time()-c}, selected


def issue_student(student, obs, cfg, index, category, expected):
    """Callback for shared_enemy_root.external_menu.issue_external after restoring a fork."""
    if index not in (0,1):
        raise ValueError("unknown arm")
    own,selected=probe(student,obs,cfg,category)
    if selected is None or selected!=expected or own["order"]!=expected["native_A"]:
        raise ValueError("restored 851c call or entire registered selection changed")
    return {**own,"order":dict(selected["options"][index])},{**selected,"pick":index}

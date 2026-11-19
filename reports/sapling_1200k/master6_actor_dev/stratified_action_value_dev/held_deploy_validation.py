"""After the training gate, compare frozen selections with deployed first edits on the exact held games."""
import argparse
import hashlib
import json
from pathlib import Path

import build
import cards
from core import HOLD_SEEDS, HERE, ROOT, read, sha
from package_probe import create, fixture


def require(test, details):
    if not test:
        raise ValueError(details)


def evaluate(game, entry, expected, actor):
    key = (game["seed"], game["seat"])
    raw, certificate = ROOT/game["raw"],ROOT/game["official"]
    require(sha(raw)==game["raw_sha256"] and sha(certificate)==game["official_sha256"],("source fingerprint",key))
    tape = read(raw)
    require(tape["initial_terrain_u8_sha256"]==game["map"] and tape["pack_sha256"]==build.PARENT_SHA
            and tape["seed"]==key[0] and tape["candidate_seat"]==key[1]
            and len(tape["actions"])==tape["steps"] and tape["steps"]>=296,("source tape",key))
    env = cards.frozen_v1.live.begin(tape)
    digest = hashlib.sha256()
    seat = game["seat"]
    result={"seed":key[0],"seat":key[1],"map":game["map"],"source_sha256":game["raw_sha256"],
            "expected":entry["first_changed_card"],"observed":None,"same_before":0}
    for t,joint in enumerate(tape["actions"][:296]):
        view = cards.frozen_v1.live.public(env,seat)
        require(view["step"]==t,("unexpected step",key,t))
        actual = actor.agent(view,tape["game_configuration"])
        trace = actor._VALUE_TRACE
        prior = joint[seat]
        digest.update(json.dumps([t,prior,actual,trace["reason"]],sort_keys=True,separators=(",",":")).encode())
        if trace["parent"]!=prior:
            result.update(error="unguarded source prefix differs from the frozen parent's published order",
                          t=t,driver_parent=trace["parent"],source=prior)
            break
        if actual!=prior:
            different = {u for u in actual.keys()|prior.keys() if actual.get(u)!=prior.get(u)}
            chosen = trace["chosen"]
            if len(different)!=1 or chosen is None:
                result.update(error="candidate mutated multiple orders or lacks selected identity",t=t)
                break
            u,action,kind = chosen
            identity = f"{key[0]}-{seat}-{t:03d}-{kind}"
            result["observed"] = {"id":identity,"t":t,"uid":u,"action":action,"original":prior.get(u,"NONE")}
            aligned = (expected is not None and entry["first_changed_card"]==identity==expected["id"]
                       and different=={u} and expected["t"]==t and expected["base"]==prior and expected["alt"]==actual
                       and dict(uid=u,original=prior.get(u,"NONE"),action=action)==expected["edits"][0]
                       and action==actual.get(u,"NONE") and trace["reason"]=="accepted")
            if not aligned:
                result["error"]="first production intervention differs from frozen selection"
            break
        result["same_before"]+=1
        env.step(joint)
    else:
        if expected is not None:
            result["error"]="candidate preserved parent through step 295 despite a registered intervention"
    result.update(public_commands_sha256=digest.hexdigest(),passed=("error" not in result))
    return result


def main():
    fit_path=HERE/"fit_verdict.json"
    selection_path=HERE/"held_selection.json"
    fit=read(fit_path)
    require(fit["decision"]=="PASS","training gate must pass before inspecting held deployment")
    selection=read(selection_path)
    weights_sha=sha(HERE/"weights.json")
    require(fit["weights_sha256"]==selection["weights_sha256"]==weights_sha and
            selection["cards_sha256"]==sha(HERE/"cards.json"),"frozen selection identity")
    protocol=read(HERE/"protocol.json")
    inventory_path=HERE/"source/inventory.json"
    require(sha(inventory_path)==protocol["inventory_sha256"],"inventory changed")
    if not (HERE/"package.json").exists() or not (HERE/"candidate.py").exists():
        build.main()
    receipt=read(HERE/"package.json")
    candidate=(HERE/"candidate.py").read_bytes()
    require(receipt["candidate_sha256"]==build.sha(candidate) and weights_sha==receipt["weights_sha256"] and
            receipt["protocol_sha256"]==sha(HERE/"protocol.json") and receipt["parent_sha256"]==build.PARENT_SHA,
            "candidate/evaluation package identity")
    held=[g for g in read(inventory_path)["games"] if g["split"]=="held"]
    registered={(s,seat) for s in HOLD_SEEDS for seat in (0,1)}
    rows={(v["seed"],v["seat"]):v for v in selection["all_registered"]}
    require(len(held)==len(rows)==len(registered)==len(selection["all_registered"])==4
            and {(g["seed"],g["seat"]) for g in held}==set(rows)==registered,"expected four complete held map-seats")
    lookup={c["id"]:c for c in read(HERE/"cards.json")["cases"] if c["split"]=="held"}
    ids=[v["first_changed_card"] for v in rows.values() if v["first_changed_card"] is not None]
    require(len(ids)==len(set(ids)) and set(ids)==set(selection["active_ids"]) and set(ids)<=lookup.keys(),"active card set")
    actor=create(candidate,"frozen_stratified_held_candidate")
    tests=[]
    for game in sorted(held,key=lambda g:(g["seed"],g["seat"])):
        entry=rows[game["seed"],game["seat"]]
        selected=entry["first_changed_card"]
        card=lookup[selected] if selected else None
        if card is not None:
            require((card["map"],card["seed"],card["seat"])==(game["map"],game["seed"],game["seat"]),"wrong held card")
        tests.append(evaluate(game,entry,card,actor))
    result={"pass":all(r["passed"] for r in tests),"map_seats":len(tests),"distinct_maps":len({r["map"] for r in tests}),
            "before_commands_checked":sum(r["same_before"] for r in tests),
            "candidate_sha256":receipt["candidate_sha256"],"weights_sha256":weights_sha,"fit_sha256":sha(fit_path),
            "selection_sha256":sha(selection_path),"protocol_sha256":sha(HERE/"protocol.json"),
            "inventory_sha256":sha(inventory_path),"script_sha256":sha(Path(__file__)),"rows":tests}
    build.atomic(HERE/"held_deploy_validation.json",(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode())
    print(json.dumps({k:result[k] for k in ("pass","map_seats","distinct_maps","before_commands_checked")}))
    return result["pass"]


def check_training_fixture():
    inventory = read(HERE/"source/inventory.json")
    game = min((g for g in inventory["games"] if g["split"]=="train"),key=lambda r:(r["seed"],r["seat"]))
    roots = cards.public_roots(game)
    t, choices = next((t, rows) for t, rows in roots if rows)
    roots.close()
    source = next(iter(choices.values()))
    results = {}
    for name, margin, expected in (("positive",16.,source),("negative",0.,None)):
        raw, package_receipt = build.compose(fixture(margin))
        actor = create(raw, "fixture_"+name)
        result = evaluate(game,{"first_changed_card":expected and expected["id"]},expected,actor)
        require(result["passed"],(name,result))
        result["fixture_pack_sha256"] = package_receipt["candidate_sha256"]
        results[name]=result
    require(results["positive"]["same_before"]==t and results["negative"]["same_before"]==296,"wrong cursor")
    receipt={"purpose":"training map with independent constant diagnostic weights checks the held-driver code paths without reading held responses",
             "script_sha256":sha(Path(__file__)),**results}
    build.atomic(HERE/"held_deploy_fixture.json",(json.dumps(receipt,indent=2)+"\n").encode())
    print(t,source["proposal"]["kind"],results["negative"]["same_before"])
    return True


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-fixture",action="store_true")
    args = parser.parse_args()
    raise SystemExit(0 if (check_training_fixture() if args.train_fixture else main()) else 1)

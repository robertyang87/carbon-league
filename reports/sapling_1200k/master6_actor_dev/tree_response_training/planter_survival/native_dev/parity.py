"""Independent portable-veto parity and immediate same-root outcomes on intake 03/04."""
import argparse
from collections import Counter, defaultdict, deque
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SURVIVAL = HERE.parent
sys.path.insert(0, str(SURVIVAL))
import extract
from model import MOVES, death_risk

WEIGHTS = "1efe7d4fbcbda256362e6204c5e42fe68216c166c9e5fd4560f5a96050323e3f"
SOURCE_PARENT = "9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0"
MAPS = {"03":[621473120,621473121], "04":[621474120,621474121]}


def check(value, reason):
    if not value:
        raise ValueError(reason)


def clean(commands):
    return {k:v for k,v in commands.items() if v and v != "NONE"}


def document(path, digest):
    check(extract.sha(path)==digest,f"content changed: {path}")
    return extract.load(path)


def read_index(source, maps):
    registered = defaultdict(list)
    source_roots = Counter()
    with gzip.open(source/"opportunities.jsonl.gz", "rt") as stream:
        for line in stream:
            r = json.loads(line)
            seed,seat,t,uid = r["seed"],r["seat"],r["t"],r["uid"]
            source_roots[seed,seat] += 1
            hold = r["actions"]["NONE"]
            if r["parent_command"] != "NONE" or not hold["entry_legal"] or hold["friendly_conflict"] or not hold["can_pay_now"] or not hold["price"] or hold["price"] <= 0:
                continue
            moves = [cmd for cmd in MOVES if cmd!="NONE" and r["actions"][cmd]["entry_legal"] and not r["actions"][cmd]["friendly_conflict"]]
            if not moves:
                continue
            def draw(cmd):
                return hashlib.sha256(f"{seat}|{t}|{uid}|{cmd}".encode()).digest()
            move = min(moves,key=draw)
            old_move = extract.pair(r)
            registered[seed,seat,t].append((uid,move,old_move,hold["price"]))
    check(sorted(source_roots)==[(seed,seat) for seed in maps for seat in (0,1)],"incomplete opportunity census")
    return registered, source_roots


def package(path):
    spec = importlib.util.spec_from_file_location("frozen_planter_survival_portable",path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def counters():
    return Counter({k:0 for k in
        "opportunities leaf_agreements action_turns unchanged_v1_turns comparison_moves_same comparisons_moves_different "
        "portable_selected old_selected both_selected_same_order both_selected_different_order portable_only old_only "
        "saved harmed unchanged_alive_label foregone_payment foregone_debit savable missed_rescues both_die factual_deaths".split()})


def score_decision(module, obs, history, own, cfg, model, fixed):
    selected, old_selected = None, None
    candidates = []
    for uid,move,legacy,price in sorted(fixed):
        a = death_risk(model,obs,history,own,uid,"NONE")
        b = death_risk(model,obs,history,own,uid,move)
        c = death_risk(model,obs,history,own,uid,legacy)
        fire = a >= .20 and a >= b+.05
        if fire and selected is None:
            selected = (uid,move)
        if a>=.20 and a>=c+.05 and old_selected is None:
            old_selected = (uid,legacy)
        candidates.append(dict(uid=uid,move=move,old_move=legacy,price=price,p_stay=a,p_move=b,p_legacy=c,trigger=fire))
    expected = clean(own)
    if selected:
        expected[selected[0]] = selected[1]
    actual = clean(module.agent(obs,cfg))
    if actual != expected:
        raise ValueError(f"pack action mismatch: t{obs['step']}/{obs['player']} {expected} vs {actual}")
    trace = module._SURV_TRACE
    # Interface is emitted by the frozen candidate. An absent empty decision or an absent zero
    # score cannot pass just because the resulting outer command happens to be identical.
    check(clean(trace["v1_commands"])==clean(own),"candidate parent path differs from recorded v1")
    check(trace["step"]==obs["step"] and trace["seat"]==obs["player"], "stale pack trace")
    check(all(a["old"]=="NONE" for a in trace["opportunities"]), "pack did not restrict to actual holds")
    seen = {a["uid"]:a for a in trace["opportunities"]}
    check(len(seen)==len(trace["opportunities"])==len(candidates),"candidate opportunity population differs")
    for a in candidates:
        b = seen[a["uid"]]
        check(b["move"]==a["move"] and bool(b["trigger"])==a["trigger"],"pack root move/trigger mismatch")
        check(abs(b["p_stay"]-a["p_stay"])<1e-10 and abs(b["p_move"]-a["p_move"])<1e-10,"pack probability mismatch")
    check((trace["selected"] or None)==(selected[0] if selected else None),"candidate tie-break mismatch")
    return candidates, selected, old_selected


def run_game(target, game, roots, weights, destination, signed):
    raw = document(extract.ROOT/game["raw_path"],game["raw_sha256"])
    cfg = raw["game_configuration"]
    seed,seat = game["seed"],game["seat"]
    sim = extract.FastCarbon(cfg).reset(seed=seed)
    check(hashlib.sha256(bytes(int(v) for v in sim.carbon)).hexdigest()==game["map_sha256"],"source map identity")
    branch = extract.FastCarbon(cfg)
    history = deque(maxlen=3)
    agent = package(target)
    rows = []
    modcfg = dict(cfg,randomSeed=None)
    stats = counters()
    for t,joint in enumerate(raw["actions"]):
        obs = sim.obs(seat)
        own = joint[seat]
        state = sim.full_observation()
        registered = roots.get((seed,seat,t),[])
        cases,chosen,legacy = score_decision(agent,obs,list(history),own,modcfg,weights,registered)
        stats["action_turns"] += 1
        stats["opportunities"] += len(cases)
        stats["leaf_agreements"] += len(cases)*2
        stats["comparison_moves_same"] += sum(r["move"]==r["old_move"] for r in cases)
        stats["comparisons_moves_different"] += sum(r["move"]!=r["old_move"] for r in cases)
        stats["portable_selected"] += chosen is not None
        stats["old_selected"] += legacy is not None
        stats["unchanged_v1_turns"] += chosen is None
        if chosen and legacy:
            stats["both_selected_same_order" if chosen==legacy else "both_selected_different_order"] += 1
        elif chosen:
            stats["portable_only"] += 1
        elif legacy:
            stats["old_only"] += 1
        saved = sim.get_state() if cases else None
        actual = extract.step(sim,state,cfg,joint)
        witness = None
        for candidate in cases:
            uid,cmd = candidate["uid"],candidate["move"]
            q = [side.copy() for side in joint]
            q[seat][uid] = cmd
            branch.set_state(saved)
            alternative = extract.step(branch,state,cfg,q)
            pos = obs["players"][seat][2][uid][0]
            da,kind,cost = extract.label(state,actual,seat,uid,pos,"NONE",cfg["treeLifespan"])
            db,other_kind,pay = extract.label(state,alternative,seat,uid,extract.destination(pos,cmd),cmd,cfg["treeLifespan"])
            check(not other_kind and pay==0,"moving planter paid")
            selected = chosen == (uid,cmd)
            outcome = dict(focal=uid,move=cmd,died_stay=da,died_move=db,payment_type=kind,debit_stay=cost,debit_move=pay)
            candidate["one_step"] = outcome
            savable = da and not db
            stats["savable"] += savable
            stats["factual_deaths"] += da
            stats["both_die"] += da and db
            stats["missed_rescues"] += savable and not selected
            if selected:
                stats["saved"] += savable
                stats["harmed"] += not da and db
                stats["unchanged_alive_label"] += da==db
                stats["foregone_payment"] += cost>0
                stats["foregone_debit"] += cost
                witness = outcome
        if cases:
            rows.append(dict(t=t,seat=seat,seed=seed,roots=cases,chosen=chosen,legacy=legacy,outcome=witness))
        history.append(obs)
    check([round(x,2) for x in sim.cash]==game["scores"],"source original score mismatch")
    stats["net_saved"] = stats["saved"]-stats["harmed"]
    signed = {**signed,"tape_sha256":game["raw_sha256"],"map_sha256":game["map_sha256"]}
    extract.write(destination,dict(signed=signed,seed=seed,seat=seat,stats=dict(stats),rows=rows))
    print(destination.name,dict(stats),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True,choices=tuple(MAPS))
    parser.add_argument("--candidate-sha",required=True)
    parser.add_argument("--protocol-sha",required=True)
    args = parser.parse_args()
    target = HERE/"candidate.py"
    check(extract.sha(target)==args.candidate_sha,"wrong frozen pack")
    plan = document(HERE/"protocol.json",args.protocol_sha)
    check(plan["risk_weights_sha256"] == WEIGHTS and plan["parent_pack_sha256"] == SOURCE_PARENT,"mismatched plan")
    check(extract.sha(SURVIVAL/"model.py") == plan["model_code_sha256"], "risk implementation mismatch")
    model = document(SURVIVAL/"weights.json",WEIGHTS)
    source = SURVIVAL.parent/"batches"/("intake_"+args.batch)
    maps = MAPS[args.batch]
    receipts = extract.load(source/"source_receipt.json")
    official = document(source/"source/official_replay.json",receipts["official_sha256"])["runs"]["source"]
    check(official["pack_sha256"]==SOURCE_PARENT,"source actor mismatch")
    check(sorted((g["seed"],g["seat"]) for g in receipts["games"])==[(s,p) for s in maps for p in (0,1)],"different source games")
    roots, census = read_index(source,maps)
    signature = dict(pack_sha256=args.candidate_sha,protocol_sha256=args.protocol_sha,
                     weights_sha256=WEIGHTS,reference_chooser_sha256=extract.sha(Path(__file__)),
                     opportunity_sha256=extract.sha(source/"opportunities.jsonl.gz"),source_receipt_sha256=extract.sha(source/"source_receipt.json"))
    output = HERE/("transfer"+args.batch)
    output.mkdir(exist_ok=True)
    paths = []
    for g in receipts["games"]:
        check(any(x["seed"]==g["seed"] and x["seat"]==g["seat"] and x["transcript_sha256"]==g["raw_sha256"] and x["official_reward_by_seat"]==g["scores"] and x["statuses"]==["DONE","DONE"] for x in official["games"]),"unsigned factual score")
        file = output/f"{g['seed']}_{g['seat']}.json.gz"
        if file.exists():
            prior = extract.load(file)["signed"]
            check(all(prior[k]==v for k,v in signature.items()) and prior["tape_sha256"]==g["raw_sha256"],"stale shard; use an empty output path only for a separately registered version")
        else:
            run_game(target,g,roots,model,file,signature)
        paths.append(file)
    by_map = {seed:counters() for seed in maps}
    for path in paths:
        g = extract.load(path)
        by_map[g["seed"]].update(g["stats"])
    ok = all(c["net_saved"]>0 and c["harmed"]==0 for c in by_map.values())
    total = counters()
    for c in by_map.values():
        total.update(c)
    for c in by_map.values():
        c["edits"]=c["portable_selected"]
    result = dict(identity=signature,split=args.batch,batch=args.batch,passed=ok,parity_passed=True,
                  candidate_sha256=args.candidate_sha, protocol_sha256=args.protocol_sha,
                  maps=by_map,decision="pass" if ok else "fail",all_actions_matched=True,all_opportunity_predictions_matched=True,
                  all_original_v1_scores_matched=True,source_planter_roots=sum(census.values()),per_map=by_map,total=total,
                  per_seat_files={p.relative_to(HERE).as_posix():extract.sha(p) for p in paths},
                  tested_scope="One-step fixed simultaneous opponent commands on saved factual v1 trajectories. No new episodes or terminal-return test.")
    extract.write(HERE/f"transfer{args.batch}.json",result)
    print("passed",ok,"totals",dict(total),flush=True)
    return int(not ok)


if __name__ == "__main__":
    raise SystemExit(main())

"""Reproduce the two learner decisions and settle recorded original responses offline."""
import argparse
import base64
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PREV = HERE.parent / "original_response/settle.py"
spec = importlib.util.spec_from_file_location("recorded_window_settlement", PREV)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
K, Q, R, REPO = S.K, S.Q, S.R, S.REPO
PARENT_SHA = "8da4b465f7fa6e601e51bd7c2e81f25825cb59b9b5ab285f7f56d0f6bbfe1cac"
STUDENT_SHA = "6c54899562382552b95bc1299d83119088ef1d1206b93135e33769815ed69c9f"
MOVES = {"UP": (-1, 0), "RIGHT": (0, 1), "DOWN": (1, 0), "LEFT": (0, -1), "NONE": (0, 0)}


def verify_feature(case, own, commands):
    seat = own["player"]
    focal = case["focal"]
    uid = focal["uid"]
    _, bases, workers, _ = own["players"][seat]
    pos, cargo, role = workers[uid]
    assert role == "COLLECTOR" and cargo >= 20
    assert pos == focal["position"] and cargo == focal["cargo"]
    assert set(focal["own_bases"]) == set(bases.values())
    distance = {}
    for arm in ("parent", "student"):
        move = commands[arm].get(uid, "NONE")
        dr, dc = MOVES[move]
        row, col = ((pos//15+dr)%15, (pos%15+dc)%15)
        dest = row*15 + col
        def board_distance(base):
            vr, vc = abs(row-base//15), abs(col-base%15)
            return min(vr, 15-vr) + min(vc, 15-vc)
        distance[arm] = min(map(board_distance, bases.values()))
        assert (move, dest, distance[arm]) == (focal[arm+"_command"], focal[arm+"_destination"], focal[arm+"_distance"])
    weight = min(float(cargo)/100., 1.)
    contrast = [weight*(distance["parent"]-distance["student"])/2,
                weight*(int(distance["student"] == 0)-int(distance["parent"] == 0))]
    assert contrast == case["features"] and distance["parent"] != distance["student"]
    immediate = (distance["parent"] == 0) != (distance["student"] == 0)
    assert case["immediate_base"] is immediate
    if not immediate:
        def from_start(base):
            dr,dc = abs(pos//15-base//15),abs(pos%15-base%15)
            return min(dr,15-dr)+min(dc,15-dc)
        assert (case["t"]-210)%28 >= 14 and 1 <= min(map(from_start,bases.values())) <= 5


def inventory(env):
    result = []
    for cash, _, workers, trees in env.states[0].observation.players:
        roles = Counter(worker[2] for worker in workers.values())
        assert set(roles) <= {"COLLECTOR", "PLANTER"}
        result.append(dict(cash=float(cash), carried=float(sum(w[1] for w in workers.values())),
                           collectors=roles["COLLECTOR"], planters=roles["PLANTER"], trees=len(trees)))
    return result


def horizons(ledger, states, seat, focal_uid):
    length = len(ledger["steps"])
    out = {}
    for label, n in (("after_0", 0), ("after_8", min(8,length)), ("after_24", min(24,length)), ("terminal", length)):
        categories, focus = [Counter(), Counter()], [0., 0.]
        for row in ledger["steps"][:n]:
            for kind, side, amount, *info in row["events"]:
                if kind == "tree_absorption":
                    categories[side]["ground_rent"] += amount-info[2]
                    categories[side]["cargo_tax"] += info[2]
                else:
                    categories[side][kind] += amount
                if kind == "own_delivery" and info[0] == focal_uid:
                    focus[side] += amount
        totals = [sum(c.values()) for c in categories]
        position = states[n]
        cash = [p["cash"] for p in position]
        assert all(abs(cash[i]-states[0][i]["cash"]-totals[i]) <= .011 for i in (0,1))
        if n == length:
            for who in (0,1):
                for key in set(categories[who]) | set(ledger["whole"][who]):
                    assert abs(categories[who][key]-ledger["whole"][who].get(key, 0.)) < 1e-6
        own_deposits = [c.get("own_delivery", 0.) for c in categories]
        out[label] = dict(actual_transitions=n, by_physical_seat=[dict(c) for c in categories],
                          inventory_by_physical_seat=position, normal_deposits=own_deposits,
                          deposit_margin=own_deposits[seat]-own_deposits[1-seat], focal_deposits=focus[seat],
                          cash_margin=cash[seat]-cash[1-seat])
    return out


def same_host(host, baseline):
    R.native_clock.validate_host(host, baseline["affinity"])
    for key in ("hostname", "cpu_model", "kernel", "affinity", "cpu_max"):
        assert host[key] == baseline[key]
    assert all(p["num_threads"] == 1 for p in host["pools"] if p["user_api"] == "blas")
    assert host["torch_threads"] == 2 and all(x["cpus"] == baseline["affinity"] for x in host["thread_affinity"])


def replay_arm(case, arm, folder, raw, bank, shared_sha, root, root_sha, parent, root_observations):
    seat, t0 = case["seat"], case["t"]
    path = folder / (arm+".json.gz")
    receipt = R.load(path)
    assert receipt["identity"] == bank["identity"] and receipt["shared_sha"] == shared_sha
    for key in ("state_sha", "snapshot_sha"):
        assert root[key] == receipt[key]
    assert receipt["root_sha"] == root_sha
    assert (receipt["seed"],receipt["seat"],receipt["arm"],receipt["root_t"],receipt["case_id"]) == (case["seed"],seat,arm,t0,case["id"])
    tape = receipt["tape"]
    assert tape and receipt["tape_sha"] == S.digest(tape) and tape[0] == receipt["root_joint"]
    assert receipt["original_orders"] == [joint[1-seat] for joint in tape]
    assert tape[0][seat] == case[arm] and tape[0][1-seat] == root["enemy"]
    assert len(receipt["suffix_observations"]) == len(tape)
    assert len(receipt["own_public_seconds"]) == len(tape) and all(math.isfinite(w) and w >= 0 for w in receipt["own_public_seconds"])
    clock = S.clock_proof(receipt["clock"], t0+1, t0+len(tape))
    for key in ("host_at_start","host_at_end"): same_host(receipt[key], bank["host"])
    env = K.checked_env(raw)
    for joint in raw["actions"][:t0]: env.step(joint)
    states = [inventory(env)]
    for i,joint in enumerate(tape):
        own, enemy = K.public(env, seat), K.public(env, 1-seat)
        obs_hashes = [S.digest(own),S.digest(enemy)]
        assert not env.done and own["step"] == t0+i and obs_hashes == receipt["suffix_observations"][i]
        if i == 0:
            assert obs_hashes == root_observations
        else:
            assert R.live.clean(parent.agent(own,raw["game_configuration"])) == joint[seat]
        env.step(joint)
        states.append(inventory(env))
    scores, cash, status = K.score(env)
    margin = scores[seat]-scores[1-seat]
    assert (scores, cash, status, margin) == (receipt["official_scores"],receipt["physical_cash"],receipt["status"],receipt["official_margin"])
    changed = S.changed_units(case["parent"],case["student"])
    old = Q.BEGIN, Q.END
    try:
        Q.BEGIN, Q.END = t0, t0+24
        ledger = Q.accounting(raw,tape,cash,[dict(t=t0,changed=changed)])
    finally:
        Q.BEGIN, Q.END = old
    for inv,row in zip(states[1:],ledger["steps"],strict=True):
        assert [p["trees"] for p in inv] == row["live_trees"]
        assert all(abs(v["cash"]-q) <= .011 for v,q in zip(inv,row["cash"],strict=True))
    stats = horizons(ledger,states,seat,case["focal"]["uid"])
    output = dict(margin=margin,wdl=int(margin>0)-int(margin<0),
                  deposit8=stats["after_8"]["normal_deposits"][seat],deposit24=stats["after_24"]["normal_deposits"][seat],
                  own_cash=cash[seat],enemy_cash=cash[1-seat],physical_cash=cash,official_scores=scores,status=status,
                  receipt_sha256=R.sha(path),tape_sha256=S.digest(tape),root_to_terminal_steps=len(tape),
                  subsequent_parent_deployments_verified=len(tape)-1,clock=clock,horizons=stats)
    return output, tape, ledger


def verify_root(case, folder, bank, shared_sha, raw, own, enemy, commands, parent):
    t, seat = case["t"], case["seat"]
    path = folder / "root.json.gz"
    root = R.load(path)
    root_hashes = [S.digest(own),S.digest(enemy)]
    assert root["t"] == t and root["case"] == case and root["identity"] == bank["identity"]
    assert [root["own_sha"], root["enemy_sha"]] == root_hashes
    assert root_hashes[0] == case["public_sha256"] and case["prefix_sha256"] == root["prefix_sha"] == S.digest(raw["actions"][:t])
    assert bank["trace"][t]["original_proposed"] == root["enemy"]
    snapshot = base64.b64decode(root["snapshot_b64"],validate=True)
    assert hashlib.sha256(snapshot).hexdigest() == root["snapshot_sha"] and len(root["state_sha"]) == 64
    assert all(commands[a] == case[a] for a in commands) and commands["parent"] != commands["student"]
    assert S.digest(commands) == case["joint_pair_sha256"] and S.changed_units(commands["parent"],commands["student"]) == case["all_changed"]
    for slot_trace in case["legal_prefix"].values():
        assert all(record["legal"][record["label"]] for record in slot_trace)
    verify_feature(case,own,commands)
    # Both deployed modules keep per-episode state in _FEAT and diagnostics in _COUNTS.
    # Each tested branch starts just after the common root's own public inference.
    common_state = parent._FEAT, parent._COUNTS
    arm_data, tapes, ledgers = {}, {}, {}
    try:
        for arm in ("parent", "student"):
            parent._FEAT, parent._COUNTS = deepcopy(common_state)
            arm_data[arm], tapes[arm], ledgers[arm] = replay_arm(case,arm,folder,raw,bank,shared_sha,root,R.sha(path),parent,root_hashes)
    finally:
        parent._FEAT, parent._COUNTS = common_state
    assert tapes["parent"][0][1-seat] == tapes["student"][0][1-seat] and tapes["parent"][0][seat] != tapes["student"][0][seat]
    for arm,ledger in ledgers.items():
        ledger_path = folder / (arm+"_verified_ledger.json.gz")
        R.write(ledger_path,ledger)
        arm_data[arm]["ledger_sha256"] = R.sha(ledger_path)
    delta = {k:arm_data["student"][k]-arm_data["parent"][k] for k in ("margin","wdl","deposit8","deposit24","own_cash","enemy_cash")}
    first_enemy = S.first_order_difference(tapes["parent"],tapes["student"],1-seat,t)
    return dict(id=case["id"],seed=case["seed"],terrain=case["terrain"],seat=seat,t=t,features=case["features"],
                source_sha256=case["source_sha256"],public_sha256=case["public_sha256"],prefix_sha256=case["prefix_sha256"],
                root_file_sha256=R.sha(path),shared_file_sha256=shared_sha,root_clock=bank["clock"]["calls"][t],
                root_state_sha256=root["state_sha"],snapshot_sha256=root["snapshot_sha"],economic_target="official_margin",
                focal=case["focal"],arms=arm_data,delta=delta,first_subsequent_enemy_order_difference=first_enemy)


def verify_map(cases, data_dir, packages, source_cards, expected_cpu):
    first = cases[0]
    seed,seat=first["seed"],first["seat"]
    folder=data_dir/"native"/f"{seed}_{seat}"
    path=folder/"shared.json.gz"
    bank=R.load(path)
    shared_sha=R.sha(path)
    assert len({(c["source_sha256"],c["terrain"],c["source"]) for c in cases}) == 1
    assert bank["cpu"] == expected_cpu and bank["host"]["affinity"] == [expected_cpu]
    assert (bank["seed"],bank["seat"],bank["terrain"]) == (seed,seat,first["terrain"])
    same_host(bank["host"],bank["host"])
    same_host(bank["host_at_end"],bank["host"])
    card=source_cards[first["source_sha256"]]
    assert (card["split"],card["seed"],card["seat"],card["terrain"],card["path"]) == ("train",seed,seat,first["terrain"],first["source"])
    source=REPO/first["source"]
    assert R.sha(source)==first["source_sha256"]
    assert R.sha(REPO/card["receipt_path"])==card["receipt_sha256"]
    assert bank["identity"] == dict(protocol=R.sha(data_dir/"protocol.json"),roots=R.sha(data_dir/"roots.json"),
                                  runner=R.sha(HERE/"collect.py"),kernel=R.sha(Path(K.__file__)),
                                  window_kernel=R.sha(HERE.parent/"original_response/run.py"),
                                  original=R.sha(K.ORIGINAL),packages={k:R.sha(p) for k,p in packages.items()},
                                  source=first["source_sha256"],source_receipt=card["receipt_sha256"],
                                  seed=seed,seat=seat,terrain=first["terrain"],cpu=expected_cpu)
    raw=R.load(source)
    assert raw["seed"]==seed and raw["candidate_seat"]==seat and not raw["errors"]
    assert S.sha_bytes(bytes(map(int,raw["initial_full_carbon"]))) == first["terrain"] == raw["initial_terrain_u8_sha256"]
    by_t = {c["t"]:c for c in cases}
    last = max(by_t)
    assert len(by_t) == len(cases) and bank["prefix"] == raw["actions"][:last]
    assert bank["trace"][-1]["archived_enemy"] is None
    assert len(bank["trace"]) == last+1
    clock=S.clock_proof(bank["clock"],0,last+1)
    models={k:R.live.module(p,"settle_return_"+k) for k,p in packages.items()}
    env=K.checked_env(raw)
    verified, different = [], []
    for t,record in enumerate(bank["trace"]):
        own,enemy=K.public(env,seat),K.public(env,1-seat)
        assert not env.done and record["t"]==own["step"]==enemy["step"]==t
        assert record["own_sha"]==S.digest(own) and record["enemy_sha"]==S.digest(enemy)
        orders={arm:R.live.clean(m.agent(own,raw["game_configuration"])) for arm,m in models.items()}
        assert all(record[arm]==cmd for arm,cmd in orders.items())
        if t in by_t:
            case=by_t[t]
            verified.append(verify_root(case,folder/case["id"],bank,shared_sha,raw,own,enemy,orders,models["parent"]))
        if t < last:
            joint=raw["actions"][t]
            assert record["archived_enemy"]==R.live.clean(joint[1-seat])
            if record["original_proposed"]!=record["archived_enemy"]:
                different.append(t)
            env.step(joint)
    return verified,dict(seed=seed,seat=seat,terrain=first["terrain"],root_ids=[r["id"] for r in verified],
                         shared_sha256=shared_sha,source_sha256=first["source_sha256"],native_prefix_clock=clock,
                         original_proposal_differs_from_executed_opening_steps=different)


def settle(directory):
    assert __debug__
    plan=R.load(directory/"protocol.json")
    signed=R.load(directory/"roots.json")
    assert signed["protocol_sha256"] == R.sha(directory/"protocol.json")
    packs={a:REPO/plan["packages"][a]["path"] for a in ("parent","student")}
    assert all(R.sha(p)==plan["packages"][a]["sha256"] for a,p in packs.items())
    assert all(R.sha(REPO/v["path"])==v["sha256"] for v in plan["inputs"].values())
    assert (R.sha(packs["parent"]),R.sha(packs["student"]))==(PARENT_SHA,STUDENT_SHA)
    assert R.sha(K.ORIGINAL)==R.live.MASTER_SHA==plan["original_sha256"]
    source_path=HERE.parent/"sources.json"
    cards={c["sha256"]:c for c in R.load(source_path)["cards"] if c["split"]=="train"}
    assert len({c["terrain"] for c in cards.values()})==13
    groups=defaultdict(list)
    for case in signed["roots"]:
        assert 210 <= case["t"] <= 265 and case["source_sha256"] in cards
        groups[case["seed"],case["seat"]].append(case)
    assert len(groups)<=4 and len({c["id"] for c in signed["roots"]})==len(signed["roots"])
    rows, maps, clocks=[], [], []
    registered={(m["seed"],m["seat"]):(i,m) for i,m in enumerate(plan["maps"])}
    assert len(registered)==4 and set(groups)<=set(registered)
    for key,cases in sorted(groups.items()):
        idx,m=registered[key]
        assert all(all(c[f]==m[f] for f in ("source","source_sha256","seed","seat","terrain")) for c in cases)
        assert len(cases)<=2 and len({(c["t"]-210)//28 for c in cases})==len(cases)
        added, summary=verify_map(cases,directory,packs,cards,idx%2)
        rows.extend(added);maps.append(summary)
        clocks.append(summary["native_prefix_clock"])
        clocks.extend(v["clock"] for row in added for v in row["arms"].values())
    result=dict(roots_sha256=R.sha(directory/"roots.json"),protocol_sha256=R.sha(directory/"protocol.json"),
                sources_sha256=R.sha(source_path),original_sha256=plan["original_sha256"],packages=plan["packages"],
                scope="Two selected full commands at each historical training-map root; one natural original draw and shared 8da4 continuation.",
                maps=maps,rows=rows,registered_map_count=len(registered),measured_map_count=len(groups),
                unfilled_windows=2*len(registered)-len(rows),unique_native_calls=sum(c["calls"] for c in clocks),
                unique_native_multicandidate_zero_depth=sum(len(c["zero_depth_steps"]) for c in clocks))
    R.write(directory/"labels.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,default=HERE)
    args=parser.parse_args()
    os.sched_setaffinity(0,{0})
    settle(args.directory)

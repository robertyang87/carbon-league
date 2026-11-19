"""Verify the shared-root examiner census and independently replay both official short branches."""
from collections import Counter
from hashlib import sha256
import copy
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import play

PROTOCOL = "7b917c1a53ace904fab729a1c83186dac53bebf4bea990bfda6697ed27351631"

def require(test, message):
    if not test:
        raise ValueError(message)

def filehash(path):
    return sha256(path.read_bytes()).hexdigest()

def check_stats(call):
    for key in ("wall", "cpu"):
        require(math.isfinite(call[key]) and call[key] > 0, "Invalid physical timing")
    s = call["search"]
    counts = s["counts"]
    require(counts["single"] + counts["searched"] + counts["aborted"] == 1, "Missing real native work unit")
    if counts["single"]:
        require((s["candidates"],s["depth"]) == (1,0), "Single case misclassified")
    elif counts["searched"]:
        require(s["candidates"] >= 2 and s["depth"] > 0, "Missing completed native search")
    else:
        require(s["candidates"] >= 2 and s["depth"] == 0, "Missing native deadline fallback")

def legal(observation, order):
    own = observation["players"][observation["player"]]
    for uid, cmd in order.items():
        if uid in own[1]:
            require(cmd in ("NONE","RECCOLLECTOR","RECPLANTER"), "Unknown base order")
        else:
            require(uid in own[2] and cmd in ("NONE","UP","RIGHT","DOWN","LEFT"), "Unknown worker order")

def public(live,env,bank):
    obs = [live.observe(env,s,bank[s]) for s in (0,1)]
    return obs, [sha256(live.encoded(o)).hexdigest() for o in obs]

def replay(seed,seat,terrain,prefix,arm):
    env,_,live = play.official(seed,terrain,seat)
    bank = [5.,5.]
    game_inputs, inputs, boards = [], [], []
    for t,row in enumerate([*prefix,*arm]):
        require(not env.done and env.states[0].observation.step == row["step"] == t, "Broken clock or suffix sequence")
        obs, hashes = public(live,env,bank)
        require(hashes == row["public_before_by_seat"], "Wrong physical/public state before move")
        require(row["orders"][seat] == row["student"]["order"] and
                row["orders"][1-seat] == row["original"]["order"], "Physical-seat request/command mismatch")
        if t >= len(prefix):
            inputs.append(hashes[1-seat])
            game_only = dict(obs[1-seat])
            del game_only["remainingOverageTime"]
            game_inputs.append(sha256(live.encoded(game_only)).hexdigest())
            boards.append(play.board_hash(env,terrain))
        for side,call in ((seat,row["student"]),(1-seat,row["original"])):
            legal(obs[side],call["order"])
            require(math.isfinite(call["wall"]) and call["wall"] > 0, "Missing timed physical action")
            bank[side] -= max(call["wall"]-1.,0.)
        check_stats(row["original"])
        require(len(row["bank_after"]) == 2 and all(abs(c-d)<1e-10 for c,d in zip(bank,row["bank_after"])),
                "Incorrect single debit of the native call in a counterfactual ledger")
        env.step(row["orders"])
        require(public(live,env,bank)[1] == row["public_after_by_seat"] and
                play.board_hash(env,terrain) == row["board_after_sha256"], "Official successor observation/IDs differ")
    require(not env.done, "Short fixture unexpectedly reached terminal")
    return dict(enemy_public=inputs,enemy_game_only=game_inputs,boards=boards,official_transitions=len(prefix)+len(arm))

def audit(data):
    require(filehash(HERE/"protocol.json") == data["protocol_sha256"] == PROTOCOL, "Different registered test")
    require(all(filehash(play.ROOT/path) == sha for path,sha in data["sources"].items()), "Executed source digest changed")
    spec = play.read(HERE/"protocol.json")["fixture"]
    require(data["seed"] == spec["training_seed"] and data["seat"] == spec["seat"] and
            data["new_full_games"] == data["terminal_labels"] == 0, "Wrong subjects or eligibility")
    if data["status"] == "no_registered_supported_root":
        return dict(status="no_comparison",fixture_status=data["status"])
    require(data["status"] == "verified", "Incomplete physical fixture")
    seat,seed,terrain = data["seat"],data["seed"],data["terrain"]
    prior = play.read(play.INTAKE/"protocol.json")["physical_boards"]
    require(prior[str(seed)] == terrain, "Training terrain identity mismatch")
    t = data["root_step"]
    prefix,arms,s = data["prefix"],data["branches"],data["shared"]
    require(t in spec["registered_steps"] and len(prefix) == t and set(arms) == {"A","B"}, "Unregistered root or missing branch")
    horizon = spec["joint_steps_per_arm"]
    require(horizon == 3, "Unexpected mechanism horizon")
    reqs = data["native_requests"]
    ids = [r["id"] for r in reqs]
    steps = [r["step"] for r in reqs]
    require(len(ids) == len(set(ids)) and steps == [*range(t),t,*range(t+1,t+horizon),*range(t+1,t+horizon)],
            "Native physical call census differs from single root followed by separate branches")
    handles = {r["id"]:r for r in reqs}
    shared_id = s["request_id"]
    require(ids[t] == shared_id and s["original_observation_sha256"] == s["pre_public_by_seat"][1-seat],
            "Root examiner request did not use the common pre-student observation")
    require(s["post_original_fingerprint"] != s["pre_fingerprints"]["original"], "Examiner state not advanced by true root call")
    require(all(len(s["pre_fingerprints"][k])==64 for k in ("original","official","student")), "Missing registered prior states")
    rootcall = s["root_native_call"]
    check_stats(rootcall)
    uses = Counter()
    require(data["restored_final"], "Unconfirmed final root restoration")
    def verify_request(row,ident):
        r = handles[ident]; uses[ident] += 1
        for unit in ("wall","cpu"):
            require(0 < r[unit] <= row["original"][unit] + 2e-6, "Invalid nested independent request timing")
        require(r["order"] == row["orders"][1-seat] and r["observation_sha256"] == row["public_before_by_seat"][1-seat]
                and r["step"] == row["step"], "Standalone native receipt differs from physical input, order, or step")
    for row,ident in zip(prefix,ids):
        require(row["native_request_id"] == ident and not row["shared_root"], "Duplicated pre-root request")
        verify_request(row,ident)
    replays = {}
    logical_wall, logical_cpu = 0., 0.
    for idx,name in enumerate(("A","B")):
        arm = arms[name]
        require(len(arm["steps"]) == arm["replayed_transitions"] == horizon, "Incomplete branch")
        require(arm["pre_original_fingerprint"] == s["pre_fingerprints"]["original"] and
                arm["restored_post_original_fingerprint"] == s["post_original_fingerprint"], "Incorrect parent or restored examiner state")
        replays[name] = replay(seed,seat,terrain,prefix,arm["steps"])
        require(replays[name]["boards"][0] == s["pre_board_sha256"], "Different physical/ID root")
        for offset,row in enumerate(arm["steps"]):
            ident = row["native_request_id"]
            require((ident == shared_id) == (offset == 0) == bool(row["shared_root"]), "Improper reuse of a native response")
            verify_request(row,ident)
            require(handles[ident]["game_observation_sha256"] == replays[name]["enemy_game_only"][offset],
                    "Native request does not equal the independently replayed game-only enemy observation")
            if offset == 0:
                require(row["public_before_by_seat"] == s["pre_public_by_seat"] and row["original"] == rootcall,
                        "Shared real root clock/workload/order differs across branches")
                event = arm["root_event"]
                require(event["options"][idx] == row["orders"][seat] and event["pick"] == idx,
                        "Requested physical student option was not actually issued")
            logical_wall += row["original"]["wall"]
            logical_cpu += row["original"]["cpu"]
    require(uses == Counter({k:2 if k == shared_id else 1 for k in ids}), "Some branch successor shared a call or a logged request went unused")
    a,b = arms["A"]["root_event"],arms["B"]["root_event"]
    require(a["options"] == b["options"] and a["options"][0] != a["options"][1] and
            a["p"] == b["p"] and a["unforced_choice"] == b["unforced_choice"] and
            all(0 < x < 1 for x in a["p"]) and abs(sum(a["p"])-1) < 1e-12,
            "Changed support or natural selection between branches")
    counts = [{"offset":offset,"separate_genuine_calls":offset>0,
               "different_native_observation":replays["A"]["enemy_public"][offset]!=replays["B"]["enemy_public"][offset],
               "different_game_observation_excluding_clock":replays["A"]["enemy_game_only"][offset]!=replays["B"]["enemy_game_only"][offset],
               "different_original_order":arms["A"]["steps"][offset]["original"]["order"]!=arms["B"]["steps"][offset]["original"]["order"]}
              for offset in range(horizon)]
    return dict(status="mechanism_verified",seed=seed,seat=seat,root_step=t,terrain=terrain,
                original_physical_calls=len(reqs),physical_prefix_calls=t,actual_shared_root_calls=1,
                physical_separate_successor_calls=2*(horizon-1),logical_root_debits=2,
                independent_outer_original_root_call={k:handles[shared_id][k] for k in ("id","wall","cpu","observation_sha256","game_observation_sha256")},
                root_native_wall_seconds=rootcall["wall"],root_native_cpu_seconds=rootcall["cpu"],root_search=rootcall["search"],
                branch_logical_wall_seconds=logical_wall,branch_logical_cpu_seconds=logical_cpu,
                branch_physical_wall_seconds=logical_wall-rootcall["wall"],branch_physical_cpu_seconds=logical_cpu-rootcall["cpu"],
                replayed=replays,successor_response=counts,new_games=0,terminal_labels=0,
                scope="Native receipt census and independent official action replay; conditioning on one real opponent root response. Absolute physical time and CPU load come from the run receipts. Excludes cold start and snapshot/audit overhead from summed native calls. A three-step fixture does not estimate gain, certify ladder resources, or certify any other root.")

if __name__ == "__main__":
    path = HERE/"receipt.json"
    data = play.read(path)
    report = audit(data)
    if report["status"] == "mechanism_verified":
        probes = {}
        for key in ("reused_successor", "changed_debit", "different_private_state", "wrong_public_observation"):
            corrupted = copy.deepcopy(data)
            b = corrupted["branches"]["B"]
            if key == "reused_successor":
                b["steps"][1]["native_request_id"] = corrupted["branches"]["A"]["steps"][1]["native_request_id"]
            elif key == "changed_debit":
                b["steps"][0]["bank_after"][1-data["seat"]] -= .125
            elif key == "different_private_state":
                b["restored_post_original_fingerprint"] = b["pre_original_fingerprint"]
            else:
                b["steps"][1]["public_before_by_seat"][1-data["seat"]] = "0"*64
            try:
                audit(corrupted)
            except ValueError as exc:
                probes[key] = str(exc)
            else:
                raise ValueError("Tampered receipt accepted: "+key)
        report["rejected_negative_controls"] = probes
    report.update(receipt_sha256=filehash(path),audit_sha256=filehash(Path(__file__)),protocol_sha256=PROTOCOL)
    dest = HERE/"audit.json"
    if dest.exists():
        require(play.read(dest)==report,"Inconsistent existing audit")
    else:
        play.save(dest,report)
    print(report["status"],filehash(dest))

"""Identify a one-step defense offer using only the current learner observation and its own issued order."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ARENA = HERE.parent
STRAT = HERE.parents[2]
SOURCE = STRAT / "replication/closing_response/run.py"
spec = importlib.util.spec_from_file_location("registered_replay_support", SOURCE)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
MOVES = {"NONE": (0, 0), "UP": (-1, 0), "RIGHT": (0, 1), "DOWN": (1, 0), "LEFT": (0, -1)}
ARM = "elite"
TARGET = "player-1-worker-22"

def dest(pos, move):
    row, col = divmod(pos, 15)
    dy, dx = MOVES[move]
    return ((row + dy) % 15) * 15 + (col + dx) % 15

def distance(a, b):
    y, x = divmod(a, 15)
    v, u = divmod(b, 15)
    return min(abs(y-v), 15-abs(y-v)) + min(abs(x-u), 15-abs(x-u))

def cards(obs, issued):
    own, enemy = obs["players"]
    at, cargo, role = enemy[2][TARGET]
    assert role == "COLLECTOR" and cargo > 0
    bases = sorted(enemy[1].values())
    home_dist = lambda p: min(distance(p, base) for base in bases)
    toward = sorted({dest(at, move) for move in MOVES if home_dist(dest(at, move)) == home_dist(at)-1})
    options = []
    for uid, (p, carried, typ) in own[2].items():
        if typ != "COLLECTOR" or carried >= cargo:
            continue
        occupied = {dest(w[0], issued.get(k, "NONE")) for k,w in own[2].items() if k != uid}
        occupied.update(pos for k,pos in own[1].items() if k in issued)
        for action in MOVES:
            tile = dest(p, action)
            if tile not in toward or tile in occupied or action == issued.get(uid, "NONE"):
                continue
            options.append(dict(uid=uid, pos=p, carried=carried, action=action, tile=tile,
                                rival_lighter_at_one_step=sorted(k for k,w in enemy[2].items()
                                   if w[2]=="COLLECTOR" and w[1]<=carried and distance(w[0],tile)<=1)))
    options.sort(key=lambda x:(x["carried"], x["tile"], x["uid"], list(MOVES).index(x["action"])))
    return dict(target=TARGET,position=at,cargo=cargo,enemy_bases=bases,closer_destinations=toward,candidates=options)

def read_source():
    settle = json.loads((ARENA / "hosted/settlement.json").read_bytes())
    proof, = [a for a in settle["archives"] if a["seed"]==621758521 and a["seat"]==0]
    path = ARENA/"hosted"/(str(proof["id"])+".zip")
    assert native.sha(path.read_bytes()) == proof["sha256"]
    with zipfile.ZipFile(path) as z:
        raw=z.read("elite/turns.jsonl")
        tape=[json.loads(b) for b in raw.splitlines()]
        meta=json.loads(z.read("elite/registration.json"))
        rec=json.loads(z.read("elite/receipt.json"))
    assert native.sha(raw)==rec["turns_sha256"] and rec["official_transitions"]==299 and rec["status"]=="complete"
    return proof,tape,meta,rec

def prepare():
    proof,tape,meta,rec=read_source()
    import gzip
    cfg=json.loads(gzip.decompress((native.CARGO/"direct_original/games/candidate/raw_621758346_1.json.gz").read_bytes()))["game_configuration"]
    cfg={**cfg,"agentTimeout":20,"actTimeout":1,"runTimeout":1500}
    env=native.collect.begin(621758521,cfg);bank=[5.,5.];checks=[]
    for t in range(209):
        seen=native.public(env,0,bank)
        assert native.sha(native.encoded(seen))==tape[t]["public_observation_sha256"][0]
        # No root opponent order enters the selector. The actual joint orders before the observation are history.
        sent=tape[t]["calls"][0]["response"]["action"]
        if t in (206,208):
            case=cards(seen,sent);checks.append(dict(step=t,inputs=case))
            if case["candidates"]:
                choice=case["candidates"][0]
                alt=copy.deepcopy(sent)
                if choice["action"]=="NONE": alt.pop(choice["uid"],None)
                else: alt[choice["uid"]]=choice["action"]
                public_hash=tape[t]["public_observation_sha256"]
                # Bind the archived simultaneous reply only after forming the alternative.
                enemy=tape[t]["calls"][1]["response"]["action"]
                p=dict(schema="sapling/exposed-defense-response/1",source_run=44651865095,seed=621758521,seat=0,
                    terrain=meta["physical_map_sha256"],terminal_step=299,root_step=t,
                    zip_id=proof["id"],zip_sha256=proof["sha256"],turns_sha256=rec["turns_sha256"],
                    original_sha256="65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5",
                    student_sha256="5e4f5da3fd63962ed559ba9efdeb788f454e3f6fb9ce617a3b5c877080795cb2",
                    root_public_sha256=public_hash,selector_sha256=native.sha(Path(__file__).read_bytes()),
                    target_uid=TARGET,defender_uid=choice["uid"],observed_checks=checks,
                    anchor_orders=dict(issued=sent,alternative=alt,enemy=enemy),
                    selection="Retrospective exposed diagnostic: the 521 s0 defeat, two candidate times 206 then 208 and rival 22 were identified from an already observed delivery. At the first registered time a lighter collector can reach a strictly distance-reducing neighbor of 22 toward the closest enemy base and change its actual order without a new allied destination conflict, choose the lightest collector, then lowest tile, uid, move order NONE/UP/RIGHT/DOWN/LEFT. Same-step future response is hidden from this selector. Check the entire changed order under the actual released Sapling prefix at execution. An adjacent rival lighter than the defender may contest; retain that risk.",
                    branches="Only one physical command at the root changes. Both arms apply the same archived simultaneous opponent action. Rebuild all public history, teacher belief and student state from turn zero, then measure original 0.58 second wall-clock search and natural temperature 1 in every subsequent live call. Both arms are controlled by the same frozen elite package. Two pre-fixed original private sampling streams use identical initial state within a pair, alternating order. The archived root game, two map-conditional pairs and four diagnostic suffixes remain outside the intact game and training denominators. Local CPU affinity cannot certify platform timing.",
                    pairs=[dict(pair=0,stream_seed=1206278037,order=["issued","alternative"]),
                           dict(pair=1,stream_seed=2170832603,order=["alternative","issued"])],
                    responses="Count immediate root captures, survival and first actual target deposit within the branch; include all later target deposits, own/other normal deposit, net tree operating cash, costs, final official score and both signed paired margins. Retain both pairs even if opposite in sign or independent streams diverge. No population interval. Report original positive depth, singles, zero depth, CPU/wall time and all exceptions.")
                return p
        orders=[c["response"]["action"] for c in tape[t]["calls"]];env.step(orders);bank=tape[t]["overage_remaining_seconds"]
    raise ValueError("No qualifying corridor intervention")
if __name__=="__main__":
    p=prepare();out=HERE/"protocol.json"; raw=(json.dumps(p,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode()
    if out.exists() and out.read_bytes()!=raw:raise ValueError("Registration already frozen")
    out.write_bytes(raw)
    print(p["root_step"],p["observed_checks"],native.sha(raw))

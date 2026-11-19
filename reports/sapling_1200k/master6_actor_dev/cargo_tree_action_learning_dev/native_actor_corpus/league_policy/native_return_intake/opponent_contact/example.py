"""Predict first, then join the two recorded first transitions of the exposed 462/t84 example."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
INTAKE = HERE.parent
TICKET = INTAKE / "economy_proposals/terminal_intake/outputs/621758462_0"
sys.path.insert(0,str(INTAKE / "live_curriculum"))
import play

SEED, SEAT, STEP = 621758462, 0, 84
PLANTER = "player-0-worker-6"
COLLECTOR = "player-1-worker-7"
ACTIONS = ("NONE","UP","RIGHT","DOWN","LEFT")
SHIFT = ((0,0),(-1,0),(0,1),(1,0),(0,-1))
PLAN = "2c904de2e1c31c49a9d8897123f0569e71728896b46da9f792ccbdb0b9e3c18c"
PUBLIC = {
    "claim.json":"005fb7c5018a497b3303f2b8b8dc6cc9301e816b9b572ce3f43b987f4affac0a",
    "prefix.json":"83cfb0c3c43ff61bbb7572073faeeca66983d6cde05dedb218e6deca9eb69dc4",
    "prefix.jsonl.gz":"b404aeec15c74f49b23d7f313c5f3e2c3163167e9303b9320ac2843e8cc42524",
}
TERRAIN = "50d7da141c5b0809e5255e4da132f44a7031ef92fd26a2231e73805d7a5b05ab"
ROOT_PUBLIC = "50120bd80beddaaf7b4580ac15c23d2f33a4ceea155317bb0fdcffc9e9a80c5a"


def require(test, why):
    if not test:
        raise ValueError(why)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def encoded(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()


def published(path, obj):
    blob = encoded(obj)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as fp:
        fp.write(blob)
        tmp = Path(fp.name)
    try:
        try:
            os.link(tmp,path)
        except FileExistsError:
            require(path.read_bytes()==blob, "Different result already sealed at " + str(path))
    finally:
        tmp.unlink()
    return sha(blob)


def destination(pos, move, size):
    y,x = divmod(int(pos),size)
    dy,dx = SHIFT[move]
    return ((y+dy)%size)*size+(x+dx)%size


def current_public(on_observation=None):
    """Advance the official referee on 84 historical transitions. A tracker receives only the acting seat."""
    require(sha((HERE / "protocol.json").read_bytes())==PLAN, "Changed registration")
    plan = json.loads((HERE / "protocol.json").read_bytes())
    require(SEED not in plan["fit_terrain_seeds"] and SEED in plan["heldout_terrain_seeds"],"Unexpected map membership")
    data = {n:(TICKET/n).read_bytes() for n in PUBLIC}
    require({n:sha(b) for n,b in data.items()}==PUBLIC, "Changed public historical input")
    meta,claim = json.loads(data["prefix.json"]),json.loads(data["claim.json"])
    require(meta["root_step"]==STEP and meta["terrain"]==TERRAIN and
            (claim["seed"],claim["seat"])==(SEED,SEAT) and meta["sources"]==claim["source_sha256"], "Wrong source ticket")
    roster = json.loads((INTAKE/"live_curriculum/protocol.json").read_bytes())["actors"]
    for name,digest in (("student",plan["sources"]["student"]),("original",plan["sources"]["examiner"])):
        src = roster[name]["file"]
        require(meta["sources"][src]==digest==sha((play.ROOT/src).read_bytes()), "Different "+name+" source")
    env,cfg,live = play.official(SEED,TERRAIN,SEAT)
    actor = play.audit_menu.collect.Player(play.student_history.PACK) if on_observation else None
    raw = gzip.decompress(data["prefix.jsonl.gz"])
    require(sha(raw)==meta["raw_turns_sha256"],"Changed uncompressed prefix")
    rows = [json.loads(line) for line in raw.splitlines()]
    require(len(rows)==STEP, "Incomplete prefix")
    bank = [5.,5.]
    for t,line in enumerate(rows):
        obs = live.observe(env,SEAT,bank[SEAT])
        require(t==line["step"]==obs["step"] and "full_carbon" not in obs and
                sha(live.encoded(obs))==line["public_before_by_seat"][SEAT], "Wrong pre-action history")
        if on_observation:
            on_observation(obs)
            play.audit_menu.per_order(actor,"candidate",obs,cfg,line["orders"][SEAT])
        for who,side in (("student",SEAT),("original",1-SEAT)):
            bank[side] -= max(0.,line[who]["wall"]-1.)
            require(line[who]["order"]==line["orders"][side], "Wrong signed physical order")
        require(bank==line["bank_after"],"Invalid bank")
        env.step(line["orders"])
        require(play.board_hash(env,TERRAIN)==line["board_after_sha256"],"Official prefix or entity IDs differ")
    obs = live.observe(env,SEAT,bank[SEAT])
    require(not env.done and obs["step"]==STEP and sha(live.encoded(obs))==ROOT_PUBLIC
            ==meta["root_public_by_seat"][SEAT] and "full_carbon" not in obs,"Wrong public root")
    if on_observation:
        on_observation(obs)
    return env, cfg, live, bank, obs


def predict():
    import features
    import model
    tracks=features.Tracks()
    env,cfg,live,bank,obs=current_public(tracks.observe)
    ids,x=features.board(obs,tracks)
    enemy=obs["players"][1-SEAT][2]
    own=obs["players"][SEAT][2]
    require(len(ids)==len(set(ids)) and set(ids)=={uid for uid,v in enemy.items() if v[2]=="COLLECTOR"}
            and own[PLANTER]==[129,0,"PLANTER"] and enemy[COLLECTOR]==[130,0,"COLLECTOR"],
            "Wrong roles, workers or input population")
    links=list(features.target_links(obs,ids))
    models={}
    for name in ("full","geometry"):
        path=HERE/(name+".npz")
        with np.load(path,allow_pickle=False) as pack:
            w={key:pack[key] for key in pack.files}
        q=model.predict(w,x)
        require(q.shape==(len(ids),5) and np.isfinite(q).all() and np.all(q>=0) and
                np.allclose(q.sum(axis=1),1.,atol=1e-6,rtol=0),"Invalid network predictions")
        targets={}
        for move in ("NONE","UP"):
            intent=ACTIONS.index(move)
            pos=destination(own[PLANTER][0],intent,cfg["size"])
            matching=[(i,j) for i,uid in enumerate(ids) for j in range(5)
                      if destination(enemy[uid][0],j,cfg["size"])==pos]
            declared=[(cell,enemies,moves) for uid,act,cell,enemies,moves in links
                      if uid==PLANTER and act==intent]
            require(len(declared)==1 and declared[0][0]==pos and
                    sorted(zip(declared[0][1],declared[0][2]))==sorted(matching),"Contact geometry differs")
            targets[move]={"cell":pos,"contact_probability":1.-math.prod(1.-float(q[i,j]) for i,j in matching),
                           "contributors":[{"uid":ids[i],"move":ACTIONS[j],"probability":float(q[i,j])} for i,j in matching]}
        models[name]={"weights_sha256":sha(path.read_bytes()),
                      "enemy_action_probabilities":{uid:q[i].tolist() for i,uid in enumerate(ids)},
                      "own_planter_targets":targets}
    return dict(schema="sapling-native-contact-exposed-prediction/1",seed=SEED,seat=SEAT,step=STEP,
                own_planter=PLANTER,highlighted_enemy=COLLECTOR,action_order=ACTIONS,protocol_sha256=PLAN,
                distinct_acting_seat_observations=STEP+1,public_input_sha256=ROOT_PUBLIC,
                public_history_files=PUBLIC,terrain=TERRAIN,training_or_test_metric_rows_from_this_trajectory=0,
                enemy_collectors={uid:{"position":enemy[uid][0],"cargo":enemy[uid][1]} for uid in ids},
                code_sha256={p:sha((HERE/p).read_bytes()) for p in ("example.py","model.py","features.py")},
                feature_sha256=sha(np.ascontiguousarray(x).tobytes()),feature_shape=list(x.shape),models=models,
                contact_definition="At least one opposing collector intends that physical cell; independent-collector product approximation.",
                scope="Historical contact and worker IDs motivated this example. Outputs are sealed before this program joins the simultaneous command. Separate trajectory from fit and held-out metrics.")


def join(prediction_path):
    prior=prediction_path.read_bytes()
    require(encoded(predict())==prior,"Sealed prediction could not be independently regenerated before the label join")
    prediction=json.loads(prior)
    require(prediction["public_input_sha256"]==ROOT_PUBLIC and prediction["protocol_sha256"]==PLAN and
            prediction["public_history_files"]==PUBLIC,"Wrong published prediction")
    for name,model in prediction["models"].items():
        require(model["weights_sha256"]==sha((HERE/(name+".npz")).read_bytes()),"Frozen model changed")
    require(prediction["code_sha256"]=={f:sha((HERE/f).read_bytes()) for f in ("example.py","model.py","features.py")},
            "Prediction code changed")
    common_path=TICKET/"shared.json"
    common=json.loads(common_path.read_bytes())
    original=common["root_native_call"]["order"]
    require(common["pre_public_by_seat"][SEAT]==ROOT_PUBLIC,"Wrong common native request")
    moves={uid:original.get(uid,"NONE") for uid in prediction["enemy_collectors"]}
    require(all(a in ACTIONS for a in moves.values()),"Unknown actual worker move")
    branches={}
    for arm in ("A","B"):
        env,cfg,live,bank,obs=current_public()
        tape=TICKET/(arm+".jsonl.gz")
        with gzip.open(tape,"rb") as f:
            row=json.loads(next(f))
        require(row["step"]==STEP and row["shared_root"] and row["native_request_id"]==common["request_id"]
                and row["orders"][1-SEAT]==original and row["public_before_by_seat"][SEAT]==ROOT_PUBLIC,
                "Different simultaneous enemy command")
        own_move=row["orders"][SEAT].get(PLANTER,"NONE")
        require(own_move==("UP" if arm=="A" else "NONE"),"Different recorded intervention")
        target=prediction["models"]["full"]["own_planter_targets"][own_move]["cell"]
        contacts=[uid for uid,move in moves.items()
                  if destination(prediction["enemy_collectors"][uid]["position"],ACTIONS.index(move),cfg["size"])==target]
        env.step(row["orders"])
        require(play.board_hash(env,TERRAIN)==row["board_after_sha256"],"Official post-intervention board mismatch")
        players=env.states[0].observation.players
        branches[arm]=dict(own_move=own_move,target=target,enemy_collectors_issued_to_target=contacts,
                           focal_survived=PLANTER in players[SEAT][2],focal_next=players[SEAT][2].get(PLANTER),
                           hostile_next=players[1-SEAT][2].get(COLLECTOR),official_board_sha256=row["board_after_sha256"],
                           compressed_tape_sha256=sha(tape.read_bytes()))
    require(moves[COLLECTOR]=="LEFT" and branches["B"]["enemy_collectors_issued_to_target"]==[COLLECTOR]
            and not branches["B"]["focal_survived"] and branches["A"]["focal_survived"], "Exposed collision did not reproduce")
    return {"prediction_sha256":sha(prior),"shared_examiner_receipt_sha256":sha(common_path.read_bytes()),
            "shared_examiner_root_request_id":common["request_id"],"actual_opposing_collector_moves":moves,
            "focal_order":"LEFT","branches":branches,"new_matches":0,"terminal_strength_observations":0}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("predict","join"))
    args=parser.parse_args()
    sealed=HERE/"example_prediction.json"
    if args.stage=="predict":
        print(published(sealed,predict()))
    else:
        print(published(HERE/"example_observed.json",join(sealed)))

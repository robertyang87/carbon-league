"""Fixed-budget, terminal-only trial driver. Each run command owns one pair of original-opponent games."""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from build import BASE, HERE, PARENT, PLAN, digest, emit, prepare, save

DEV = HERE.parent
BOARDS = {621461120:"3ebcf8c720f63bf0f857175c7c52ca8565593f35d6082781a240d25b8ad6c4d8",
          621461220:"d3e7259051367fd0db834ffa53fbf1db0f98e93707ee8d20e15b211219187a59"}
ARMS = ("plus","minus","learned","parent")


def read(p):
    return json.loads(p.read_bytes())


def frozen_file(path, obj):
    if path.exists():
        if read(path)!=obj:
            raise ValueError(f"refusing to replace different frozen evidence at {path}")
    else:
        save(path,obj)


def arm_seed(arm):
    return PLAN["training_seed"] if arm in ("plus","minus") else PLAN["blind_development_seed"]


def perturb(arm):
    if arm not in ("plus","minus"):
        raise ValueError(arm)
    sign=1 if arm=="plus" else -1
    return [sign*PLAN["perturbation"]*d for d in PLAN["direction"]]


def recipe(arm, directory):
    if arm in ("plus","minus"):
        theta = perturb(arm)
    else:
        snapshot = read(HERE/"learned_checkpoint.json")
        if not snapshot["nonzero_update"]:
            raise ValueError("zero update; redundant development match suppressed")
        if snapshot != compute_fit():
            raise ValueError("training checkpoint or source has changed")
        if arm=="parent":
            ident=DEV/"hybrid_manifest.json"
            if digest(PARENT.read_bytes())!=PLAN["parent"] or read(ident)["input_and_output_sha256"]["data/weights/sapling_1200k/master6_actor_hybrid.py"]!=PLAN["parent"]:
                raise ValueError("original parent mismatch")
            return PARENT,ident,PLAN["parent"]
        theta=snapshot["theta"]
    pack,weights,identity=emit(arm,theta,directory)
    if arm=="learned" and identity!=snapshot["learned_pack"]:
        raise ValueError("checkpoint does not describe this package")
    return pack,weights,identity["pack_sha256"]


def run(arm,cpus):
    gate=read(HERE/"mechanism.json")
    if (not gate["passed"] or gate["source_sha256"] !=
        {f:digest((HERE/f).read_bytes()) for f in ("build.py","runtime.py","protocol.json","mechanism.py")}):
        raise ValueError("frozen mechanism gate missing or stale")
    clearance=read(HERE/"map_clearance.json")
    if clearance["verdict"]!="eligible_in_checked_scope" or {int(k):v for k,v in clearance["boards"].items()}!=BOARDS:
        raise ValueError("missing map separation proof")
    seed=arm_seed(arm)
    d=HERE/('train_'+arm if arm in ('plus','minus') else 'dev_'+arm)
    env=dict(os.environ, OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with tempfile.TemporaryDirectory() as tmp:
        pack,weights,fingerprint=recipe(arm,Path(tmp))
        subprocess.run([sys.executable,'-u',str(BASE/'tree_pack_arena.py'),'--pack',str(pack),'--pack-sha256',fingerprint,
                        '--weights',str(weights),'--maps',str(seed),'--cpus',*map(str,cpus),'--run',str(d)],env=env,check=True)
    with tempfile.TemporaryDirectory() as tmp:
        official=Path(tmp)/"verified.json"
        subprocess.run([sys.executable,str(BASE/'official_game_replay.py'),str(d),'--out',str(official)],env=env,check=True)
        frozen_file(d/'official_replay.json',read(official))


def terminal(margin):
    if not math.isfinite(margin):
        raise ValueError("nonfinite official score")
    w=1. if margin>0 else 0. if margin<0 else .5
    u=.5*w+.5/(1.+math.exp(max(-700,min(700,-margin/1000))))
    return u,.65*math.tanh(margin/3000)+.35*(2*u-1)


def read_tape(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def read_arm(arm,pack_sha,weight_sha):
    seed = arm_seed(arm)
    directory = HERE/("train_"+arm if arm in ("plus","minus") else "dev_"+arm)
    score=read(directory/"results.json")
    off_payload=read(directory/"official_replay.json")
    official=off_payload["runs"][directory.name]
    h=score["header"]
    if (score["status"]!="complete" or h["seeds"]!=[seed] or h["pack_sha256"]!=pack_sha or
        official["pack_sha256"]!=pack_sha or h["weight_sha256"]!=weight_sha or h["examiner_md5"]!=PLAN["examiner_md5"] or
        len(official["games"])!=2 or len(score["games"])!=2):
        raise ValueError("invalid run, identity or incomplete seats: "+arm)
    rows={}
    for o in official["games"]:
        seat=o["seat"]
        g=next(r for r in score["games"] if r["candidate_seat"]==seat)
        raw=directory/g["raw_file"]
        tape=read_tape(raw)
        config=tape["configuration"]
        if (tape["pack_sha256"]!=pack_sha or tape["weight_identity"]!=weight_sha or tape["examiner_md5"]!=PLAN["examiner_md5"] or
            tape["seed"]!=seed or tape["candidate_seat"]!=seat or tape["initial_terrain_u8_sha256"]!=BOARDS[seed] or
            tape["errors"] or tape["official_score_by_seat"]!=o["official_reward_by_seat"] or not config["_S_ENGINE"] or
            config["_TEMP"]!=1. or config["_SEED"] is not None):
            raise ValueError("actor, examiner, seat, terrain or terminal provenance disagrees in original action tape")
        if (seat not in (0,1) or seat in rows or o["seed"]!=seed or g["seed"]!=seed or not g["valid"] or g["errors"] or
            any(g["agent_errors_by_seat"]) or o["statuses"]!=["DONE","DONE"] or o["map_sha256"]!=BOARDS[seed] or
            o["official_reward_by_seat"]!=g["official_score_by_seat"] or o["steps"]!=g["actual_transitions"] or
            digest(raw.read_bytes())!=o["transcript_sha256"] or g["raw_sha256"]!=o["transcript_sha256"]):
            raise ValueError("corrupt training/validation record")
        scores=o["official_reward_by_seat"]
        m=scores[seat]-scores[1-seat]
        u,j=terminal(m)
        if abs(g["margin"]-m)>1e-7 or g["result"]!=o["result"] or o["result"]!=("win" if m>0 else "loss" if m<0 else "draw"):
            raise ValueError("inconsistent terminal label")
        rows[seat]={"seat":seat,"seed":seed,"margin":m,"U":u,"J":j,"result":o["result"],
                    "raw_sha256":o["transcript_sha256"],"steps":o["steps"]}
    return [rows[s] for s in (0,1)]


def manifest_sha(obj):
    return digest((json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())


def compute_fit():
    arms={}
    for arm in ("plus","minus"):
        _,ident=prepare(arm,perturb(arm))
        arms[arm]=read_arm(arm,ident["pack_sha256"],manifest_sha(ident))
    change=sum(p["J"]-m["J"] for p,m in zip(arms["plus"],arms["minus"]))/2
    derivative=change/(2*PLAN["perturbation"])
    cap=PLAN["coordinate_clip"]
    theta=[max(-cap,min(cap,PLAN["learning_rate"]*derivative*d)) for d in PLAN["direction"]]
    _,learned=prepare("learned",theta)
    return {"protocol_sha256":digest((HERE/"protocol.json").read_bytes()),"trainer_sha256":digest(Path(__file__).read_bytes()),
            "source":"four official terminal returns from paired physical seats on the training map",
            "training":arms,"mean_J_plus_minus":change,"directional_derivative":derivative,"theta":theta,
            "nonzero_update":any(v!=0 for v in theta),"learned_pack":learned}


def fit():
    learned=compute_fit()
    frozen_file(HERE/"learned_checkpoint.json",learned)
    print("checkpoint",learned["theta"],learned["learned_pack"]["pack_sha256"],flush=True)


def evaluate():
    frozen=read(HERE/"learned_checkpoint.json")
    if frozen!=compute_fit():
        raise ValueError("training checkpoint not frozen")
    if not frozen["nonzero_update"]:
        frozen_file(HERE/"development.json",{"status":"no_update","checkpoint_sha256":digest((HERE/"learned_checkpoint.json").read_bytes())})
        return
    parent=read_arm("parent",PLAN["parent"],digest((DEV/"hybrid_manifest.json").read_bytes()))
    learned=frozen["learned_pack"]
    child=read_arm("learned",learned["pack_sha256"],manifest_sha(learned))
    deltas=[{"seat":a["seat"],"margin":b["margin"]-a["margin"],"U":b["U"]-a["U"]} for a,b in zip(parent,child)]
    avg=sum(d["margin"] for d in deltas)/2
    frozen_file(HERE/"development.json",{"map":PLAN["blind_development_seed"],"checkpoint_sha256":digest((HERE/"learned_checkpoint.json").read_bytes()),
        "parent":parent,"candidate":child,"delta":deltas,"mean_margin_delta":avg,"replication_signal":avg>0 and all(d["U"]>0 for d in deltas),
        "formal":False})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest="cmd",required=True)
    play=subs.add_parser("run");play.add_argument("arm",choices=ARMS);play.add_argument("--cpus",type=int,nargs=2,required=True)
    subs.add_parser("fit");subs.add_parser("evaluate")
    a=parser.parse_args()
    if a.cmd=="run":run(a.arm,a.cpus)
    elif a.cmd=="fit":fit()
    else:evaluate()

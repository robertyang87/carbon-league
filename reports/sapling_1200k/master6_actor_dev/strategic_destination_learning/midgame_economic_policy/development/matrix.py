"""Register two fresh maps and run the original three-arm dual-cgroup referee."""
import argparse
import gzip
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
TRAIN=HERE.parent
STRATEGY=TRAIN.parent
BASE=STRATEGY/"development"
spec=importlib.util.spec_from_file_location("economic_three_arm_runner",BASE/"matrix.py")
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
ROOT,play,timing=runner.ROOT,runner.play,runner.timing
read,sha,record,require=runner.read,runner.sha,runner.record,runner.require
PLAN=HERE/"protocol.json"
SEEDS=(621758550,621758551)
ARMS=("parent","uniform","weighted")
WORKFLOW="sapling-midgame-economic-dev.yml"
DESIGN="952cf49aac8e8a934596ca9bd01cf73ffe1a473caa91225e4b6a6fabf99f4a28"
REFERENCE=STRATEGY/"elite_direct_policy/development/protocol.json"

def serving_hash(path):
    lines=path.read_bytes().splitlines(keepends=True)
    require(sum(x.startswith(b"_W64 = ") for x in lines)==1,"Unknown neural package format")
    return sha256(b"".join(x for x in lines if not x.startswith(b"_W64 = "))).hexdigest()

def contract():
    design=read(HERE/"design.json")
    require(sha(HERE/"design.json")==DESIGN and tuple(design["seeds"])==SEEDS and tuple(design["arms"])==ARMS,
            "Different registered experiment")
    clear=read(HERE/"clearance.json")
    more=read(HERE/"clearance_increment.json")
    require(design["physical_boards"]==clear["by_seed"]==more["physical_boards"] and
            not any(r[k] for r in (clear,more) for k in ("physical_intersections","seed_intersections")) and
            sha(HERE/"clearance.py")==clear["code_sha256"] and more["baseline_sha256"]==sha(HERE/"clearance.json") and more["top5_unique_openings"]>=1650,"Development board not clear")
    paths={"design":HERE/"design.json","census":HERE/"clearance.py","clearance":HERE/"clearance.json",
           "clearance_increment":HERE/"clearance_increment.json","matrix":HERE/"matrix.py","cli":HERE/"workflow_cli.py",
           "settler":HERE/"settle.py","economic_analysis":HERE/"economy.py","registered_runner":BASE/"matrix.py","registered_sparse":BASE/"workflow_cli.py",
           "registered_settler":BASE/"settle.py","prior":REFERENCE,"workflow":ROOT/".github/workflows"/WORKFLOW,
           "parent":STRATEGY/"bc.py","parent_weights":STRATEGY/"bc.npz",
           "panel":STRATEGY.parent/"broader_elite_backbone_dev/panel.json"}
    for k,fn in {"training_protocol":"protocol.json","trainer":"learn.py","fit":"fit.json","assessor":"evaluate.py",
            "true_entry":"entry.json","trace":"held_commands.json.gz","audit":"audit.json","auditor":"audit.py",
            "advantage":"advantage.json.gz","ridge":"ridge.json"}.items():
        paths[k]=TRAIN/fn
    for arm in ARMS[1:]:
        paths[arm]=TRAIN/(arm+".py")
        paths[arm+"_weights"]=TRAIN/(arm+".npz")
    files={k:record(p) for k,p in paths.items()}
    ref=read(REFERENCE)
    for k in ("opponent","rules","environment","official_helpers","ladder_runner","worker","container","base_registration",
              "original_game_configuration","current_guide","match_runner","replayer","native_timing_audit","previous_exposure_guard"):
        v=ref["files"][k]
        require(record(ROOT/v["path"])==v,"Changed original-platform dependency "+k)
        files[k]=v
    hashes={k:v["sha256"] for k,v in files.items()}
    rules,fit,entry,audit=(read(paths[k]) for k in ("training_protocol","fit","true_entry","audit"))
    refs={a:hashes[a] for a in ARMS}
    require(hashes["current_guide"]==design["guide_sha256"] and hashes["opponent"]==design["original_sha256"] and
            refs["parent"]==rules["parent"]==design["parent_sha256"] and hashes["parent_weights"]==rules["weights"]==design["parent_weights_sha256"] and
            hashes["training_protocol"]==fit["registered"]==entry["registered"]==design["training_protocol_sha256"],
            "Wrong training, parent, original or platform identity")
    require(hashes["trainer"]==fit["source"] and hashes["ridge"]==fit["ridge"] and hashes["advantage"]==fit["labels"] and
            entry["export_sha256"]==refs and hashes["fit"]==entry["fit"] and hashes["assessor"]==entry["evaluation_source"] and
            hashes["trace"]==entry["trace"]==audit["held_log_sha256"],"Wrong evidence bytes")
    require(len(set(refs.values()))==3 and len({serving_hash(paths[a]) for a in ARMS})==1 and
            fit["arms"]["uniform"]["updates"]==fit["arms"]["weighted"]["updates"]>0,"Unmatched network control")
    for arm in ARMS[1:]:
        f=fit["arms"][arm]
        require(f["package"]==hashes[arm] and f["weights"]==hashes[arm+"_weights"] and f["size"]==files[arm]["bytes"]<=10_000_000
                and all(f["parameter_changes"][k]>0 for k in ("stem.weight","gru.weight_ih","whead.weight")),"Quantized update not bound")
    rows=json.loads(gzip.decompress(paths["trace"].read_bytes()))
    pairs=(("parent","uniform"),("parent","weighted"),("uniform","weighted"))
    changed={f"{a}_vs_{b}":sum(r[a]["actions"]!=r[b]["actions"] for r in rows) for a,b in pairs}
    require(all(changed.values()) and changed==entry["pooled"]["changed"] and entry["eligible"] and entry["all_commands_valid"] and
            entry["continuous_cold_control_seats"]==[0,1] and len({(r["terrain"],r["t"]) for r in rows})==len(rows),
            "Ordinary public entry comparison failed")
    require(hashes["panel"]==rules["panel"] and not any(g["terrain_sha256"] in design["physical_boards"].values() or
            g["seed"] in SEEDS for g in read(paths["panel"])["games"]),"Training/evaluation overlap")
    return {**design,"schema":"sapling-midgame-economic/frozen/1","platform":ref["platform"],"files":files}

def check(digest):
    require(sha(PLAN)==digest and read(PLAN)==contract(),"Different registered sources")
    return read(PLAN)

runner.HERE,runner.PLAN,runner.SEEDS,runner.ARMS,runner.check=HERE,PLAN,SEEDS,ARMS,check
verify_block=runner.verify_block

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("op",choices=("freeze","list","check","play","verify"))
    p.add_argument("--matrix-sha");p.add_argument("--seed",type=int);p.add_argument("--seat",type=int,choices=(0,1))
    p.add_argument("--out",type=Path);p.add_argument("--image");a=p.parse_args()
    if a.op=="freeze":
        obj=contract()
        if PLAN.exists():require(read(PLAN)==obj,"Different prior freeze")
        else:play.live.save(PLAN,obj)
        print(sha(PLAN))
    elif a.op=="play":sys.exit(runner.run(a.matrix_sha,a.seed,a.seat,a.out,a.image))
    else:
        plan=check(a.matrix_sha)
        if a.op=="list":print(json.dumps({"include":[{"seed":r["seed"],"seat":r["seat"]} for r in plan["schedule"]]}))
        if a.op=="verify":
            v=verify_block(a.out,a.seed,a.seat);print(json.dumps(v,allow_nan=False));sys.exit(int(bool(v["verification_errors"])))

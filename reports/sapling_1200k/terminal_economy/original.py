"""Run same-map original-examiner development controls, separately from the proxy trials."""
import argparse
import json
import os
import subprocess
import sys

import run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm",choices=("both","candidate","parent"),default="both")
    parser.add_argument("--cpus",type=int,nargs=2,default=(0,1))
    args=parser.parse_args()
    env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    arms=("parent","candidate") if args.arm=="both" else (args.arm,)
    for label in arms:
        if label=="parent":
            pack=run.arena.PARENT
            weight=run.BASE/"delivery_parent_dev/build_manifest.json"
            pack_sha=run.arena.PARENT_SHA
        else:
            weight=run.HERE/"selection.json"
            freeze=json.loads(weight.read_bytes())
            if not freeze["admit_original_diagnostic"]:
                raise ValueError("no candidate passed the frozen training screen")
            pack=run.arena.ACTORS/"terminal_economy_agent.py"
            pack_sha=freeze["chosen_pack_sha256"]
            if run.arena.prior.digest(pack)!=pack_sha or pack.read_bytes()!=run.packed(run.ARMS[freeze["chosen"]]):
                raise ValueError("selected package identity differs")
        d=run.HERE/("original_"+label)
        subprocess.run([sys.executable,"-u",str(run.BASE/"tree_pack_arena.py"),"--pack",str(pack),
                        "--pack-sha256",pack_sha,"--weights",str(weight),"--maps",str(run.REAL),
                        "--cpus",*map(str,args.cpus),"--run",str(d)],env=env,check=True)
        if not (d/"official_replay.json").exists():
            subprocess.run([sys.executable,str(run.BASE/"official_game_replay.py"),str(d),
                            "--out",str(d/"official_replay.json")],env=env,check=True)


if __name__=="__main__": main()

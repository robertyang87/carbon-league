"""Recompute original and revised role aggregation from the identical frozen command dictionaries."""
from common import *
import subprocess

VERSION="d74e33630"
def main():
    old_rev=subprocess.check_output(["git","-C",str(ROOT),"rev-parse",VERSION+"^{commit}"],text=True).strip()
    rel=(HERE/"evaluate.py").relative_to(ROOT)
    frozen=subprocess.check_output(["git","-C",str(ROOT),"show",old_rev+":"+str(rel)])
    old={"__file__":str(HERE/"evaluate.py"),"__name__":"frozen_eval_reader"}
    exec(compile(frozen,str(rel)+"@"+VERSION,"exec"),old)
    from evaluate import split_summary,qualify
    raw=read(HERE/"held_commands.json.gz")
    rule=read(HERE/"protocol.json")
    test=[r for r in raw["rows"] if r["split"]=="test"]
    current=split_summary(test,rule["seed"]);original=old["split_summary"](test,rule["seed"])
    other=old["qualify"](original);now=qualify(current)
    if current!=original or now!=read(HERE/"held.json")["qualification"]:
        raise ValueError("criteria recomputation failed")
    changes={arm:other[arm]["passed"]!=now[arm]["passed"] for arm in now}
    save(HERE/"gate_audit.json",dict(commands_sha=sha(HERE/"held_commands.json.gz"),source=old_rev,
        frozen_evaluator_sha256=hashlib.sha256(frozen).hexdigest(),final_evaluator_sha256=sha(HERE/"evaluate.py"),
        script_sha256=sha(Path(__file__)),original_pooled=other,final_equal_terrain=now,qualification_flipped=changes,
        timing="The contract registered a -0.02 per-role guard and map-equal primary worker accuracy. Its first implementation at d74e33630 pooled all held workers within each role for the guard. The final evaluator changed only the role-gate weighting to equal terrain after that evaluator was frozen and before the final three-package held replay began or its actions/results were read. Both gates are reapplied independently to the same exact released action dictionaries."))
if __name__=="__main__":main()

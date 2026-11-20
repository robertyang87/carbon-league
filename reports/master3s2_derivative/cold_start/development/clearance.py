"""Reproduce both V2 development openings and compare seeds and initial carbon maps with the frozen corpus."""
import importlib.util
import json
from pathlib import Path
import re
import types

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/".git").exists())
STRATEGY = ROOT/"reports/sapling_1200k/master6_actor_dev/strategic_destination_learning"
DERIV = HERE.parents[1]
BASE = "4c33c48d40d23821d7cf2d6c87aa9ec641396833"
SEEDS = [621758590, 621758591]
TOKEN = re.compile(rb"(?<![.\da-zA-Z])62175859[01](?![.\da-zA-Z])")
spec = importlib.util.spec_from_file_location("shared_census",STRATEGY/"native_action_policy_dev/clearance.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
prior = shared.prior


def generate():
    prior.old.SEEDS = SEEDS
    prior.old.SOURCE = BASE
    boards = prior.old.boards()
    names = prior.git("ls-tree","-r","--name-only",BASE).decode().splitlines()
    top, proof, faults = prior.old.top_ledgers(names)
    raw = prior.frozen(prior.TEMPLATE,BASE)
    program = raw.decode()
    for before, after in (("len(top5)==1637",f"len(top5)=={len(top)}"),
                          ("if b'6217584' in b and CANDIDATE.search(b)",
                           "if b'62175859' in b and CANDIDATE.search(b)")):
        if program.count(before)!=1:
            raise ValueError("Historical census engine changed")
        program = program.replace(before,after)
    scanner=types.ModuleType("cold_start_v2_archive_census")
    scanner.__file__=str(prior.TEMPLATE)
    exec(compile(program,str(prior.TEMPLATE),"exec"),scanner.__dict__)
    scanner.SNAPSHOT=BASE
    scanner.SEEDS=SEEDS
    scanner.CANDIDATE=TOKEN
    scanner.GROUPS={"v2_development_only":SEEDS}
    census=scanner.scan()
    if census["by_seed"]!=boards or census["top5_unique_openings"]!=len(top):
        raise ValueError("Independent official board or ladder ledger differs")

    # Explicit checks for the source-based line cover both map hashes and train/holdout labels.
    protected={}
    for name in ("root_policy/protocol.json","opponent_policy/protocol.json"):
        path=DERIV/name
        data=prior.frozen(path,BASE)
        rows=json.loads(data)["games"]
        layouts={r.get("terrain_sha256",r.get("terrain")) for r in rows}
        if None in layouts or not layouts or layouts & set(boards.values()) or {r["seed"] for r in rows}&set(SEEDS):
            raise ValueError("Overlap with source training/holdout "+name)
        protected[name]=dict(rows=len(rows),maps=len(layouts),sha256=prior.sha(data))
    old=prior.frozen(DERIV/"dev1/clearance_at_freeze.json",BASE)
    prev=json.loads(old)
    prior.old.BASE=prev["through"]
    delta=prior.old.score_delta()
    if delta["unresolved_incremental_replays"]:
        raise ValueError("Missing remote replay among newly listed scores")
    census.update(schema="master3s2-cold-start/development-clearance/1",purpose="Two previously unused development maps for V2 vs the original, each played in both seats. Exclude from training and the independent examination.",
                  source_policy_exclusions=protected,top5_ledger_sha256=proof,reviewed_ladder_faults=faults,
                  official_score_delta=delta,previous_final_clearance_sha256=prior.sha(old),
                  archive_scanner_sha256=prior.sha(raw),verifier_sha256=prior.sha(Path(__file__).read_bytes()))
    return census


if __name__=="__main__":
    result=generate()
    print(shared.save(HERE/"clearance.json",result),result["by_seed"],flush=True)

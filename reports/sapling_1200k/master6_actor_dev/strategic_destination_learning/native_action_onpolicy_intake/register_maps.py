"""Reserve already permitted training maps and prove disjointness with value and evaluation maps."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
STRATEGY = HERE.parent
CARGO = STRATEGY.parent / "cargo_tree_action_learning_dev"
LEAGUE = CARGO / "native_actor_corpus/league_policy"
VALUE = STRATEGY / "native_action_return_intake/continuation_value"
SEEDS = list(range(621758420, 621758428))
BASE = "7b951a9d1"
ROOT = next(p for p in HERE.parents if (p / ".git").exists())


def blob(path):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", BASE+":" + str(path.relative_to(ROOT))])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    frozen = json.loads(blob(LEAGUE / "clearance.json"))
    original = json.loads(blob(LEAGUE / "protocol.json"))
    assert set(SEEDS) <= set(frozen["groups"]["train"])
    physical = {str(s):frozen["by_seed"][str(s)] for s in SEEDS}
    ancestry = {"clearance": digest(blob(LEAGUE / "clearance.json")), "prior_train_protocol":digest(blob(LEAGUE / "protocol.json"))}
    fit = json.loads(blob(VALUE / "fit.json"))
    fit_seeds = {r["seed"] for r in fit["held_by_map"]}
    excluded = {}
    for relative in ("development/protocol.json","replication/protocol.json","elite_direct_policy/development/protocol.json",
                     "midgame_economic_policy/development/design.json","native_action_policy_dev/design.json","policy_window_returns/protocol.json"):
        path=STRATEGY/relative
        data=blob(path)
        ancestor=json.loads(data)
        excluded.update(ancestor["physical_boards"])
        ancestry[relative]=digest(data)
    require = lambda b,e: None if b else (_ for _ in ()).throw(ValueError(e))
    require(len(fit_seeds)==10 and fit_seeds <= set(map(int,excluded)), "Unknown fitted critic maps")
    require(len(set(physical.values()))==8 and not (set(excluded)&set(physical) or set(excluded.values())&set(physical.values())), "Value/development layout collision")
    require(all(not frozen['formal_seed_half_open'][0]<=s<frozen['formal_seed_half_open'][1] for s in SEEDS), "Formal seed collision")
    require(not set(SEEDS)&set(frozen["groups"]["development"]+frozen["groups"]["original_development_reserve"]), "Training permission collision")
    spec = importlib.util.spec_from_file_location("training_terrain_census", STRATEGY / "elite_direct_policy/development/clearance.py")
    terrain = importlib.util.module_from_spec(spec);spec.loader.exec_module(terrain)
    terrain.SEEDS=SEEDS
    require(terrain.boards()==physical, "Official and fast terrains differ")
    terrain.SOURCE=BASE
    names=terrain.git("ls-tree","-r","--name-only",BASE).decode().splitlines()
    top,ledgers,faults=terrain.top_ledgers(names)
    require(not {h for s,h in top.values()}&set(physical.values()) and not {s for s,h in top.values()}&set(SEEDS), "Ladder map overlaps")
    doc=dict(schema="sapling-native-action-policy-training/registered-maps/1",
             source=terrain.git("rev-parse",BASE+"^{commit}").decode().strip(),seeds=SEEDS,physical_boards=physical,
             use="Eight existing training maps, one natural complete original-examiner game per map and alternating seat. For on-policy training only; none may count toward development or formal wins.",
             schedule=[dict(seed=s,seat=i%2,order=["collector"]) for i,s in enumerate(SEEDS)],
             expected_receipts=8,seat_counts=[4,4],budget="One original game per distinct registered map. No branch rollout, replay after role initialization, replacement, or outcome-dependent truncation.",
             fit_value_seeds=sorted(fit_seeds),fit_value_sha256=digest(blob(VALUE/"fit.json")),excluded_maps=excluded,
             prior_fingerprints=ancestry,training_group=frozen["groups"]["train"],training_scope=original["schema"],
             top5_maps=len(top),top5_faults=faults,top5_ledger_hashes=ledgers,formal_range=frozen['formal_seed_half_open'],
             disposition="Keep actual time forfeits in training provenance. Only official numeric terminal returns with verifiable trajectories are eligible for actor updates; no zero substitution for missing returns.")
    raw=(json.dumps(doc,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    dest=HERE/"design.json"
    require(not dest.exists() or dest.read_bytes()==raw, "Existing registration differs")
    dest.write_bytes(raw)
    print(digest(raw),len(top),physical)


if __name__=="__main__":
    main()

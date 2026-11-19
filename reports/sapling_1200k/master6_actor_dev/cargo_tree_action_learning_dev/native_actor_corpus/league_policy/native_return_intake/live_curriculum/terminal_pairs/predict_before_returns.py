"""Seal the frozen all-six-map research head's probabilities using only the pre-action prefix."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

HERE = Path(__file__).resolve().parent
INTAKE = HERE.parent.parent
HEAD = INTAKE / "order_reranker"
PACKAGE = HEAD / "research_candidate.py"
FULL_SHA = "d0985f023769c9c06e75ce74a0d33166c3e26c3b846f72e3eaf6f92632159232"
PARENT_SHA = "851c901aa561ce80bf04d03e7c2a8e7876bfa2f24dc6c57ffe22b501979320ee"
PLAN_SHA = "013c21283c1e48477ebb56c3668bbf79e4c019b86fd38dfb96f839018e5227bd"
sys.path.insert(0, str(HERE.parent))
import play

def digest(data):
    return hashlib.sha256(data).hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

class FollowMenu:
    """Read the probability actually used in the frozen package. Teacher-force prefix actions."""
    def __init__(self, historic):
        self.historic = historic
        self.event = None

    def random(self):
        loc = sys._getframe(1).f_locals
        require(self.event is None, "Multiple draws in same registered tick")
        p = list(map(float, loc["probs"]))
        orders = [dict(loc["base_cmd"]), dict(loc["alt_cmd"])]
        require(orders[0] != orders[1] and len(p) == 2 and
                all(0 < v < 1 for v in p) and abs(sum(p)-1) < 1e-12, "Invalid menu")
        idx = 0 if self.historic is None else orders.index(self.historic)
        self.event = dict(p=p, options=orders, native_logits=list(map(float, loc["logits"])),
                          uid=str(sys._getframe(1).f_globals["_FEAT"].own_uid[int(loc["slot"])]))
        return 0.0 if idx == 0 else p[0]

def act(actor, public, cfg, old):
    if public["step"] == 0:
        return actor.act(public, cfg), None
    stream = actor.mod._behavior_rng
    tap = FollowMenu(old)
    actor.mod._behavior_rng = tap
    try:
        return actor.act(public, cfg), tap.event
    finally:
        actor.mod._behavior_rng = stream

def predict(seed):
    require(digest((HERE/"protocol.json").read_bytes()) == PLAN_SHA, "Pair protocol changed")
    require(digest(PACKAGE.read_bytes()) == FULL_SHA, "Frozen full 272-parameter deployment changed")
    plan = play.read(HERE/"protocol.json")
    require(plan["student_sha256"] == PARENT_SHA, "Wrong natural parent")
    spec, = [s for s in plan["fixtures"] if s["seed"] == seed]
    seat, terrain = spec["seat"], spec["board_sha256"]
    folder = HERE/"outputs"/f"{seed}_{seat}"
    prefix_path = folder/"prefix.json"
    meta = play.read(prefix_path)
    require((meta["seed"], meta["seat"], meta["terrain"]) == (seed, seat, terrain), "Wrong prefix")
    tape = (folder/"prefix.jsonl.gz").read_bytes()
    tape_sha = digest(tape)
    require(tape_sha == meta["turns_sha256"], "Prefix tape digest changed")
    result = dict(schema="sapling-frozen-272-root-prediction/1", implementation_sha256=digest(Path(__file__).read_bytes()),
                  protocol_sha256=PLAN_SHA, package_sha256=FULL_SHA, parent_sha256=PARENT_SHA,
                  seed=seed, seat=seat, terrain=terrain, prefix_metadata_sha256=digest(prefix_path.read_bytes()),
                  prefix_tape_sha256=tape_sha, root_step=meta["root_step"],
                  scope="Training maps included in the old all-six-map head. New natural prefix, prior-to-branch inputs only. Probabilities of issuing each complete order, not win probabilities.",
                  status=meta["status"])
    if meta["status"] == "no_registered_supported_root":
        return result
    step = meta["root_step"]
    require(meta["status"] == "root" and step in plan["probes"] and step == meta["pre_end_step"], "Invalid root")
    env, cfg, live = play.official(seed, terrain, seat)
    actor = play.audit_menu.collect.Player(PACKAGE)
    raw = gzip.decompress(tape)
    require(digest(raw) == meta["raw_turns_sha256"], "Invalid raw prefix digest")
    lines = raw.splitlines()
    require(len(lines) == step, "Only complete, strictly pre-root transitions are permitted")
    bank = [5., 5.]
    for t, line in enumerate(lines):
        frame = json.loads(line)
        require(not env.done and frame["step"] == t, "Invalid prefix sequence")
        public = live.observe(env, seat, bank[seat])
        require(digest(live.encoded(public)) == frame["public_before_by_seat"][seat], "Public prefix mismatch")
        old = frame["orders"][seat]
        issued, _ = act(actor, public, cfg, old)
        require(issued == old, "Frozen head could not teacher-force actual prefix order")
        env.step(frame["orders"])
        bank = frame["bank_after"]
    public = live.observe(env, seat, bank[seat])
    public_sha = digest(live.encoded(public))
    require(not env.done and public["step"] == step and "full_carbon" not in public and
            public_sha == meta["root_public_by_seat"][seat], "Wrong or privileged root observation")
    issued, actual = act(actor, public, cfg, None)
    old = meta["root_menu"]
    require(actual is not None and actual["options"] == old["options"] == meta["root_orders"] and
            issued == actual["options"][0] and actual["uid"] == old["uid"] and
            max(abs(x-y) for x,y in zip(actual["native_logits"],old["native_logits"])) < 2e-6,
            "Different physical menu, focal worker or parent logits")
    native_gap = actual["native_logits"][0] - actual["native_logits"][1]
    parent = list(map(float, old["p"]))
    require(len(parent) == 2 and abs(sum(parent)-1.) < 1e-12 and
            abs(1./(1.+math.exp(-native_gap))-parent[0]) < 2e-6, "Wrong baseline probabilities")
    p = actual["p"]
    def preference(a,b):
        return "A" if a>b else "B" if b>a else "tie"
    result.update(status="probabilities_sealed", root_public_sha256=public_sha, worker=actual["uid"],
                  options={"A":actual["options"][0],"B":actual["options"][1]},
                  parent_p={"A":parent[0],"B":parent[1]}, full_head_p={"A":p[0],"B":p[1]},
                  full_minus_parent_p={"A":p[0]-parent[0],"B":p[1]-parent[1]},
                  parent_prefers=preference(*parent), full_head_prefers=preference(*p),
                  increased_relative_probability=preference(p[0]-parent[0],p[1]-parent[1]))
    return result

def seal(folder, result):
    path = folder/"prediction.json"
    if path.exists():
        record = play.read(path)
        record.pop("sealed_at_utc")
        require(record == result, "A different prediction is already sealed")
        return path
    stamped = dict(result, sealed_at_utc=datetime.now(timezone.utc).isoformat())
    with NamedTemporaryFile(dir=folder, delete=False) as f:
        stage = Path(f.name)
        f.write(play.encode(stamped))
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(stage,path)
    finally:
        stage.unlink()
    return path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",required=True,type=int,choices=(621758460,621758461))
    args = parser.parse_args()
    result = predict(args.seed)
    out = seal(HERE/"outputs"/f"{args.seed}_{result['seat']}", result)
    print(out, digest(out.read_bytes()), result.get("full_minus_parent_p"))

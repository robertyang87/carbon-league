"""Replay six completed training tapes; inject known uniform deviates only in the verifier process."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from export_behavior import build, reference_package, FOLDER, HERE, ROOT, STEPS


class ReplayDraw:
    def __init__(self, uniform):
        self.uniform = uniform
        self.menu = None

    def random(self):
        # Taken at the actual RNG call in the untouched final single file.
        f = sys._getframe(1).f_locals
        if self.menu is not None:
            raise ValueError("A scheduled step used multiple draws")
        self.menu = dict(probabilities=f["probs"].tolist(), options=[f["base_cmd"], f["alt_cmd"]],
                         slot=int(f["slot"]), actions=[int(f["best"]), int(f["second"])], route=bool(f["route"]))
        return self.uniform


def run(root, out, arm, reference):
    sys.path.insert(0, str(root / FOLDER))
    import collect as c
    import numpy as np
    sign = build(root, arm, out)
    report = {"deployment": sign, "games": [],
              "evidence": "First three ascending training maps, each played in both seats against an archived policy. Original joint histories fixed. Registered training-only instrumentation computes fresh menus and probabilities for each tested actor; final deployment is observed at its actual draw instruction in this separate Python process. Terminal actions are from the existing training tape, so no online win rate or ladder certification is inferred."}
    rows = json.loads((root/FOLDER/"collection.json").read_bytes())["games"]
    seeds = sorted({r["seed"] for r in rows})[:3]
    for r in rows:
        if r["arm"] != "sampled" or r["seed"] not in seeds:
            continue
        raw_path = root/r["path"]
        if c.sha_file(raw_path) != r["sha256"]: raise ValueError("Training tape changed")
        raw = json.loads(gzip.decompress(raw_path.read_bytes()))
        seed, seat = r["seed"], r["seat"]
        obslog = {e["step"]:e for e in raw["events"]}
        os.environ["SAPLING_NATIVE_STREAM"] = str(seed)
        os.environ.pop("SAPLING_NATIVE_TRACE", None)
        env = c.begin(seed,raw["game_configuration"])
        native = c.Player(root/sign["source"])
        sampled = c.Player(out/sign["sample_file"])
        ref = c.Player(reference)
        ref.mod._SAMPLE_STEPS = STEPS
        cfg = raw["game_configuration"]
        real = []
        for t,joint in enumerate(raw["actions"]):
            obs = c.public(env,seat)
            # Evaluate every scheduled draw and every 23rd ordinary command plus both boundaries.
            if t in STEPS or t%23==0 or t==raw["steps"]-1:
                default = native.act(obs,cfg)
                expected = ref.act(obs,cfg)
                event = ref.mod.last_training_event
                if t:
                    known_uniform = float(np.random.default_rng(np.random.SeedSequence([9173951,seed,seat,t])).random())
                    probe = ReplayDraw(known_uniform)
                    sampled.mod._behavior_rng = probe
                actual = sampled.act(obs,cfg)
                if t==0:
                    generator = sampled.mod._behavior_rng
                    if not isinstance(generator,np.random.Generator):
                        raise ValueError("Production default did not instantiate a game-local natural RNG")
                if actual != expected:
                    raise ValueError(f"Training/serving public orders differ at {seed}/{seat}/{t}")
                if t in STEPS:
                    if hashlib.sha256(c.canonical_bytes(obs)).hexdigest()!=obslog[t]["public_sha256"]:
                        raise ValueError("Public pre-action history differs from the recorded training history")
                    if arm=="parent":
                        if expected != joint[seat]:
                            raise ValueError("Frozen parent does not reproduce the original recorded event")
                        for k in ("issued","supported","probabilities","route","options","slot"):
                            if event.get(k)!=obslog[t].get(k):
                                raise ValueError("Archived menu/probability changed")
                    if event["supported"]:
                        if probe.menu is None or any(probe.menu[k]!=event[k] for k in probe.menu):
                            raise ValueError("Deployed support/probabilities/prefix/route differ")
                        if default!=event["options"][0] or actual!=event["options"][event["choice"]]:
                            raise ValueError("Default and selected physical orders differ from the decoded options")
                    elif probe.menu is not None or actual!=default:
                        raise ValueError("Unsupported step changed the default or consumed entropy")
                    real.append(dict(t=t,supported=event["supported"],choice=event.get("choice"),changed=actual!=default))
                elif expected!=default or actual!=default or (t and probe.menu is not None):
                    raise ValueError("Ordinary command changed or consumed entropy")
                else:
                    real.append(dict(t=t,ordinary=True))
            else:
                for p in (native,ref,sampled):
                    with c.attached(p.modules): p.mod._FEAT.observe(obs)
            env.step(joint)
        if not env.done or [a.reward for a in env.states]!=raw["official_score_by_seat"]:
            raise ValueError("Official training-tape terminal mismatch")
        report["games"].append(dict(seed=seed,seat=seat,opponent=r["opponent"],tape_sha256=r["sha256"],
            supported=sum(i.get("supported",False) for i in real),alternate=sum(i.get("changed",False) for i in real),
            ordinary=sum(i.get("ordinary",False) for i in real),scheduled=sum("supported" in i for i in real),
            rows=real,diagnostic_max_deploy_s=max(sampled.wall)))
    if len(report["games"])!=6 or len(set(g["opponent"] for g in report["games"]))!=3:
        raise ValueError("Missing registered training map/seat")
    report["totals"] = {k:sum(g[k] for g in report["games"]) for k in ("scheduled","supported","alternate","ordinary")}
    (out/(arm+"_audit.json")).write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(arm,report["totals"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--out",type=Path,default=HERE)
    parser.add_argument("--arm",choices=("parent","candidate"),required=True)
    args=parser.parse_args()
    with reference_package(args.root.resolve(), args.arm) as reference:
        run(args.root.resolve(), args.out, args.arm, reference)

"""Repeat a completed training root at the original decision head with the release's natural RNG."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
from export_behavior import reference_package, FOLDER, HERE, ROOT, STEPS


def check(root,out,arm,reference):
    sys.path.insert(0,str(root/FOLDER))
    import collect as c
    audit=json.loads((out/(arm+"_audit.json")).read_bytes())
    package=out/audit["deployment"]["sample_file"]
    if c.sha_file(package)!=audit["deployment"]["sha256"]: raise ValueError("Changed actor")
    rows=[]
    for seat in (0,1):
        seed=621758420
        game=json.loads(gzip.decompress((root/FOLDER/"train_games"/f"{seed}_{seat}_sampled.json.gz").read_bytes()))
        env=c.begin(seed,game["game_configuration"])
        actor=c.Player(package); ref=c.Player(reference); ref.mod._SAMPLE_STEPS=STEPS
        c.os.environ["SAPLING_NATIVE_STREAM"]=str(seed)
        for t,pair in enumerate(game["actions"][:13]):
            obs=c.public(env,seat)
            if t==12:
                original_mixed=actor.mod._mixed_actions
                captured=[]
                def also_capture(x,aux):
                    captured.append((x,aux))
                    return original_mixed(x,aux)
                actor.mod._mixed_actions=also_capture
            ref_order=ref.act(obs,game["game_configuration"])
            order=actor.act(obs,game["game_configuration"])
            if t==0:
                rng_hash=hashlib.sha256(c.canonical_bytes(actor.mod._behavior_rng.bit_generator.state)).hexdigest()
            if t==12: break
            env.step(pair)
        event=ref.mod.last_training_event
        if not event["supported"] or len(captured)!=1:raise ValueError("Diagnostic root lost support")
        options=event["options"]
        observed=[options.index(order)]
        with c.attached(actor.modules):
            for _ in range(47):
                order=dict(actor.mod._FEAT.to_actions(original_mixed(*captured[0])))
                observed.append(options.index(order))
        rows.append(dict(seed=seed,seat=seat,step=12,tape_sha256=c.sha_file(
              root/FOLDER/"train_games"/f"{seed}_{seat}_sampled.json.gz"),
              trials=len(observed),counts=[observed.count(i) for i in (0,1)],
              training_reference_probabilities=event["probabilities"],options=options,
              first_rng_state_sha256=rng_hash,actual_decoded_choice_sequence=observed))
    if rows[0]["first_rng_state_sha256"]==rows[1]["first_rng_state_sha256"]:
        raise ValueError("Reused entropy from independent game instances")
    doc={"package_sha256":c.sha_file(package),"training_roots":rows,
         "scope":"One unchanged exposed decision per seat, 48 independent successive draws from the deployed RNG at each root. First draw uses normal public entry, remaining draws use its exact frozen pre-action inputs and complete native decoder. Frequencies are diagnostic; these are not full games or an estimate of competitive strength."}
    (out/(arm+"_natural.json")).write_text(json.dumps(doc,sort_keys=True,indent=2)+"\n")
    print(arm,[r["counts"] for r in rows])


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=ROOT);p.add_argument("--out",type=Path,default=HERE);p.add_argument("--arm",required=True,choices=("parent","candidate"))
    a=p.parse_args()
    with reference_package(a.root.resolve(),a.arm) as reference:
        check(a.root.resolve(),a.out,a.arm,reference)

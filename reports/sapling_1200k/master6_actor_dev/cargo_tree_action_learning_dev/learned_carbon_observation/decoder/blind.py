"""Seal uninstrumented public decoder commands without decoding beyond the four frozen roots."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
CASES = HERE.parent / "active_roots" / "roots.json"
ROOTS_SHA = "126e372073ba2c38066eb3cd1d2c579888d81d2a24e4ce80654b84c34b361354"
PROTOCOL_SHA = "af39fdbd8e200906cf90dbf868aa9f717601b626c16e0ae4a3d5bb493cd734b1"
PACKS = {"decoder_student": "candidate.py", "decoder_constant": "constant.py"}
PUBLIC_KEYS = {"step","player","players","trees","carbon","remainingOverageTime"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",",":")).encode()


def worker(pack_file, expected_sha):
    assert sha(pack_file.read_bytes()) == expected_sha
    spec = importlib.util.spec_from_file_location("blind_public_agent", pack_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    data = json.load(sys.stdin)
    commands = []
    for t, obs in enumerate(data["public_history"]):
        assert set(obs) == PUBLIC_KEYS and obs["step"] == t
        action = mod.agent(obs, data["configuration"])
        assert isinstance(action, dict) and all(isinstance(k, str) and isinstance(v,str) for k,v in action.items())
        commands.append(action)
    print(json.dumps({"commands":commands[-1], "all_public_output_sha256":sha(canonical(commands)), "calls":len(commands)}, allow_nan=False))


def public_prefix(case):
    sys.path.insert(0, str(HERE.parent / "active_roots"))
    from select_roots import streaming_archive, begin, public
    source = ROOT / case["source"]
    assert sha(source.read_bytes()) == case["source_sha256"]
    cfg, actions = streaming_archive(source)
    assert cfg["seed"] == case["seed"] and cfg["initial_terrain_u8_sha256"] == case["terrain"]
    env = begin(cfg)
    history, prefix = [], []
    try:
        for t in range(case["t"] + 1):
            obs = public(env, case["seat"])
            assert obs["step"] == t and set(obs) == PUBLIC_KEYS
            # Copy the interpreter's mutable state before advancing the referee.
            history.append(json.loads(canonical(obs)))
            if t < case["t"]:
                action = next(actions)
                prefix.append(action)
                env.step(action)
    finally:
        actions.close()
    assert sha(canonical(prefix)) == case["prefix_sha256"]
    obs_hash = sha(canonical({k:v for k,v in history[-1].items() if k!="remainingOverageTime"}))
    assert obs_hash == case["public_obs_sha256"]
    return history, {**cfg["game_configuration"], "randomSeed":None}


def main():
    assert sha(CASES.read_bytes()) == ROOTS_SHA
    assert sha((HERE / "protocol.json").read_bytes()) == PROTOCOL_SHA
    cases = json.loads(CASES.read_bytes())["cases"]
    assert len(cases) == 4
    signatures = {name:sha((HERE / path).read_bytes()) for name,path in PACKS.items()}
    results = []
    for case in cases:
        history, cfg = public_prefix(case)
        row = {k:case[k] for k in ("seed","seat","t","source_sha256","terrain","prefix_sha256","public_obs_sha256","incumbent","alternative")}
        row["public_history_sha256"] = sha(canonical(history))
        row["outputs"] = {}
        assert case["incumbent"] != case["alternative"]
        for name,file in PACKS.items():
            args=[sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--worker", str(HERE / file), signatures[name]]
            worker_out=subprocess.check_output(args,input=canonical({"public_history":history,"configuration":cfg}))
            outcome=json.loads(worker_out)
            assert outcome["calls"] == case["t"]+1
            act=outcome["commands"]
            outcome["whole_team_match"] = "parent" if act==case["incumbent"] else "alternative" if act==case["alternative"] else "other"
            row["outputs"][name] = outcome
        results.append(row)
    seal = dict(protocol_sha256=PROTOCOL_SHA, registered_roots_sha256=ROOTS_SHA, runner_sha256=sha(Path(__file__).read_bytes()),
                pack_sha256=signatures,
                method="Fresh process for each map/seat/pack. Official interpreter advanced solely on the frozen prefix; the root and subsequent archive commands were never decoded. Each package received only the six public observation fields at every step, with randomSeed removed. Full physical commands compared to the two registered options; other has no measured return.",
                cases=results)
    dest=HERE/"blind_commands.json"
    raw=(json.dumps(seal, allow_nan=False, sort_keys=True, indent=2)+"\n").encode()
    if dest.exists() and dest.read_bytes()!=raw:
        raise RuntimeError("Existing sealed commands differ")
    if not dest.exists():
        with dest.open("xb") as f:
            f.write(raw)
    print(sha(raw))
    for r in results:
        print(r["seed"],r["seat"],r["t"],{name:v["whole_team_match"] for name,v in r["outputs"].items()})


if __name__ == "__main__":
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"
    os.sched_setaffinity(0,{3})
    if len(sys.argv)>1 and sys.argv[1]=="--worker":
        worker(Path(sys.argv[2]),sys.argv[3])
    else:
        main()

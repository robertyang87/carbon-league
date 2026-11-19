"""Build a single-file development candidate with the complete unchanged v1 network payloads."""
import ast
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import zlib

import actions
import policy

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parent
ROOT = ACTOR.parents[2]
PARENT = ROOT/"data/weights/sapling_1200k/terminal_return_fit_v1.py"
VALIDATOR = ACTOR/"late_local_residual_dev/deploy.py"
RULES = ROOT/"code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json"
PARENT_SHA = "9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0"
VALIDATOR_SHA = "db27b1832f1b1140484987411afcaff687c0f8b1c112df55c659416aff387ba8"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def weights_ok(p):
    if (not isinstance(p["enabled"],bool) or tuple(p["features"])!=actions.FEATURES
            or tuple(p["kinds"])!=actions.KINDS or set(p["heads"])!=set(policy.TARGETS)):
        raise ValueError("unexpected feature, branch or target identity")
    size = len(actions.FEATURES) + len(actions.KINDS)
    for row in p["heads"].values():
        if len(row)!=size or not all(isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v) for v in row):
            raise ValueError("incomplete or nonfinite fitted weights")


def embed(raw, var):
    blob = base64.b85encode(zlib.compress(raw,9))
    return f"{var}=types.ModuleType({var!r})\nexec(zlib.decompress(base64.b85decode({blob!r})),{var}.__dict__)\n".encode()


def atomic(path,raw):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)


def compose(weights):
    weights_ok(weights)
    parent = PARENT.read_bytes()
    guard = VALIDATOR.read_bytes()
    if sha(parent)!=PARENT_SHA or sha(guard)!=VALIDATOR_SHA:
        raise ValueError("frozen actor or prefix validator identity changed")
    guard_source = guard.decode()
    node, = [n for n in ast.parse(guard_source).body if isinstance(n,ast.FunctionDef) and n.name=="_local_edit"]
    exact = ast.get_source_segment(guard_source,node).encode()
    action_code = (HERE/"actions.py").read_bytes()
    policy_code = (HERE/"policy.py").read_bytes()
    glue = (HERE/"candidate_wrapper.py").read_bytes()
    rules = json.loads(RULES.read_bytes())["configuration"]
    cfg = {k:v["default"] for k,v in rules.items() if isinstance(v,dict) and "default" in v and k!="randomSeed"}
    cfg.setdefault("episodeSteps",300)
    serialized = parent + b"\n" + exact + b"\n"+embed(action_code,"_VALUE_ACTIONS")+embed(policy_code,"_VALUE_POLICY")
    serialized += f"_VALUE_DEFAULT_CFG={cfg!r}\n_VALUE_PARAMS={weights!r}\n".encode()+glue
    compile(serialized,"candidate.py","exec")
    if len(serialized)>10_000_000:
        raise ValueError("platform single-file limit")
    return serialized,{"candidate_sha256":sha(serialized),"bytes":len(serialized),"parent_sha256":sha(parent),
                       "validator_file_sha256":sha(guard),"validator_function_sha256":sha(exact),
                       "actions_sha256":sha(action_code),"policy_sha256":sha(policy_code),"wrapper_sha256":sha(glue),
                       "rules_sha256":sha(RULES.read_bytes()),"build_sha256":sha(Path(__file__).read_bytes())}


def main():
    weights=(HERE/"weights.json").read_bytes()
    candidate,receipt=compose(json.loads(weights))
    protocol=(HERE/"protocol.json").read_bytes()
    registration = json.loads(protocol)
    if (registration["parent_sha256"]!=receipt["parent_sha256"]
            or registration["frozen_method"]["actions.py"]!=receipt["actions_sha256"]
            or registration["frozen_method"]["policy.py"]!=receipt["policy_sha256"]):
        raise ValueError("actor, features or action choice differ from registration")
    receipt.update(weights_sha256=sha(weights),protocol_sha256=sha(protocol),purpose="development")
    atomic(HERE/"candidate.py",candidate)
    atomic(HERE/"package.json",(json.dumps(receipt,indent=2,allow_nan=False)+"\n").encode())
    print(json.dumps(receipt))


if __name__=="__main__":
    main()

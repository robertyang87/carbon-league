"""Build a self-contained development actor while retaining the two frozen parent network payloads."""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import zlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PARENT = ROOT/"data/weights/sapling_1200k/terminal_return_fit_v1.py"
PARENT_HASH = "9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0"
RULES = ROOT/"code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def compose(runtime, weights):
    parent = PARENT.read_bytes()
    if sha(parent) != PARENT_HASH:
        raise ValueError("combined parent identity changed")
    rules = json.loads(RULES.read_bytes())["configuration"]
    config = {k:v["default"] for k,v in rules.items() if isinstance(v,dict) and "default" in v and k!="randomSeed"}
    config.setdefault("episodeSteps", 300)
    module = base64.b85encode(zlib.compress(runtime,9))
    metadata = ("\n_LOCAL_RUNTIME = types.ModuleType('sapling_public_local_residual')\n"
                + f"exec(zlib.decompress(base64.b85decode({module!r})), _LOCAL_RUNTIME.__dict__)\n"
                + f"_LOCAL_DEFAULT_CFG = {config!r}\n_LOCAL_WEIGHTS = {weights!r}\n")
    deploy = (HERE/"deploy.py").read_bytes()
    package = parent + metadata.encode() + deploy
    compile(package,"candidate.py","exec")
    if len(package)>10_000_000:
        raise ValueError("single-file size limit")
    return package, {"candidate_sha256":sha(package),"bytes":len(package),"frozen_parent_sha256":sha(parent),
                     "runtime_sha256":sha(runtime),"deploy_sha256":sha(deploy),"rules_sha256":sha(RULES.read_bytes()),
                     "build_sha256":sha(Path(__file__).read_bytes())}


def atomic(path, raw):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    tmp.write_bytes(raw)
    tmp.replace(path)


def main():
    runtime = (HERE/"runtime.py").read_bytes()
    weights = (HERE/"weights.json").read_bytes()
    protocol = (HERE/"protocol.json").read_bytes()
    if sha(runtime) != json.loads(protocol)["proposal_sha256"]:
        raise ValueError("public proposal differs from registration")
    params = json.loads(weights)
    if not isinstance(params.get("enabled"), bool):
        raise ValueError("residual enabled status missing")
    if params["enabled"]:
        import runtime as shared
        if tuple(params["features"]) != shared.FEATURES or set(params["heads"]) != {"margin", "principal", "own_delivery"}:
            raise ValueError("unexpected head identity")
        for coef in params["heads"].values():
            if len(coef) != len(shared.FEATURES) or not all(math.isfinite(x) for x in coef):
                raise ValueError("incomplete or nonfinite learned coefficients")
    package, receipt = compose(runtime, params)
    receipt.update(weights_sha256=sha(weights), protocol_sha256=sha(protocol), enabled=params["enabled"])
    atomic(HERE/"candidate.py",package)
    atomic(HERE/"package.json",(json.dumps(receipt,indent=2,allow_nan=False)+"\n").encode())
    return receipt


if __name__=="__main__":
    print(json.dumps(main()))

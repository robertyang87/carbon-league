"""Package the original two-network interpreter, its recorded prefix and the independent cargo rule."""
import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import zlib

import actions

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PARENT=ROOT/"data/weights/sapling_1200k/terminal_return_fit_v1.py"
VALIDATOR=HERE.parent/"late_local_residual_dev/deploy.py"
RULES=ROOT/"code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def function(raw,name):
    text=raw.decode()
    node,=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name]
    return ast.get_source_segment(text,node).encode()


def write(path,raw):
    tmp=path.with_name(f".{path.name}.{os.getpid()}.part")
    tmp.write_bytes(raw)
    tmp.replace(path)


def compose(enabled=True):
    parent=PARENT.read_bytes()
    valid=VALIDATOR.read_bytes()
    if sha(parent)!=actions.PARENT_SHA or sha(valid)!="db27b1832f1b1140484987411afcaff687c0f8b1c112df55c659416aff387ba8":
        raise ValueError("changed parent or validator")
    feat=(HERE/"actions.py").read_bytes()
    guard=function(valid,"_local_edit")
    decode=function(parent,"_return_decode")
    needle=b"        action = int(np.argmax(np.where(legal, combined, -np.inf)))"
    if decode.count(needle)!=1:
        raise ValueError("decoder identity")
    instrument=decode.replace(b"def _return_decode(",b"def _cargo_scored_decode(",1).replace(
        needle,b"        _cargo_record(slot,combined,legal)\n"+needle)
    scope=json.loads(RULES.read_bytes())["configuration"]
    cfg={k:v["default"] for k,v in scope.items() if isinstance(v,dict) and "default" in v and k!="randomSeed"}
    cfg.setdefault("episodeSteps",300)
    marker=base64.b85encode(zlib.compress(feat,9))
    prefix=(f"\n_CARGO_ENABLED={bool(enabled)!r}\n_CARGO_CFG={cfg!r}\n"
            f"_CARGO_ACTIONS=types.ModuleType('sapling_loaded_actions')\nexec(zlib.decompress(base64.b85decode({marker!r})), _CARGO_ACTIONS.__dict__)\n").encode()
    wrapper=(HERE/"pack.py").read_bytes()
    binary=parent+prefix+guard+b"\n"+instrument+b"\n"+wrapper
    compile(binary,"sapling_loaded.py","exec")
    if len(binary)>10_000_000:
        raise ValueError("size exceeds competition limit")
    return binary,{"sha256":sha(binary),"bytes":len(binary),"enabled":enabled,"parent_sha256":sha(parent),
                   "actions_sha256":sha(feat),"wrapper_sha256":sha(wrapper),"build_sha256":sha(Path(__file__).read_bytes()),
                   "guard_definition_sha256":sha(guard),"original_decoder_sha256":sha(decode),"rules_sha256":sha(RULES.read_bytes())}


def main():
    binary,report=compose()
    off,off_receipt=compose(False)
    protocol=(HERE/"protocol.json").read_bytes()
    registered=json.loads(protocol)
    if (registered["actions_sha256"]!=report["actions_sha256"] or
            registered["parent_sha256"]!=report["parent_sha256"] or
            registered["min_cargo"]!=actions.MIN_CARGO or registered["t"]!=[180,295]):
        raise ValueError("production candidate differs from its registration")
    report["protocol_sha256"]=sha(protocol)
    report["disabled_sha256"]=off_receipt["sha256"]
    write(HERE/"candidate.py",binary)
    write(HERE/"package.json",(json.dumps(report,indent=2)+"\n").encode())
    print(json.dumps(report))


if __name__=="__main__":
    main()

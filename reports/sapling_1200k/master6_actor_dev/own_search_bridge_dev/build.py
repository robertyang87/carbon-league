"""Rebuild deployable H8 self-rollout package from the registered opening Sapling and existing prefix code."""

import argparse
import ast
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import zlib
import numpy as np

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parent
ROOT = ACTOR.parents[2]
DEFAULT = Path("/workspace/sapling-h8-self-rollout.py")
PAY = ACTOR / "tree_tenure_dev/pay_policy"
OPEN = ACTOR / "blended_actor_dev/opening_training"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def functions(src):
    return {
        n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)
    }


def make(out):
    # The opening release builder writes release.json, so rebuild it in an isolated directory.
    import importlib.util, shutil

    b = importlib.util.spec_from_file_location("own_bridge_opening", OPEN / "build.py")
    m = importlib.util.module_from_spec(b)
    b.loader.exec_module(m)
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d)
        m.HERE = folder
        for name in ("protocol.json", "fit.json", "opening_weights.npz"):
            shutil.copyfile(OPEN / name, folder / name)
        parent = folder / "parent.py"
        m.build(parent)
        src = parent.read_text()
    proto = json.loads((HERE / "entry_protocol.json").read_text())
    assert sha(src.encode()) == proto["parent"]
    funcs = functions(src)
    dec = funcs["_return_decode"]
    dec = dec.replace(
        "_return_decode(x, aux, encoded, apply_residual)",
        "_pay_forced(x, aux, encoded, apply_residual, target_slot, target_action)",
    )
    off = dec.index("        if not apply_residual:")
    end = dec.index("        labels[slot] = action", off)
    dec = dec[:off] + """        if slot==target_slot:
            if slot>=10 or x["tok_kind"][slot]!=1 or not legal[target_action]:
                return None
            action=target_action
""" + dec[end:]
    mix = funcs["_mixed_actions"]
    assert mix.count("return baseline") == 3 and mix.count("return candidate") == 1
    mix = mix.replace("return baseline", "return _pay_select(x,aux,enc,baseline,False)").replace(
        "return candidate", "return _pay_select(x,aux,enc,candidate,True)"
    )
    overlay = (PAY / "overlay.py").read_text()
    needle = "    view, cfg = _PAY_VIEW"
    assert overlay.count(needle) == 1 and overlay.count("return other if old!='NONE' else original") == 1
    overlay = overlay.replace(needle, needle + "\n    _bridge_observe(view, original)")
    overlay = overlay.replace(
        "return other if old!='NONE' else original", "return _bridge_choose(view,cfg,original,other)"
    )
    overlay = overlay.replace(
        "    global _PAY_VIEW", "    global _PAY_VIEW,_BR_START\n    _BR_START=time.perf_counter()"
    )
    # The packaged encoder also has optional diagnostic reveals. Restrict its input to actual platform data.
    mark = "    _PAY_VIEW=(observation,{**_PAY_CFG,**(configuration or {})})"
    assert overlay.count(mark) == 1
    overlay = overlay.replace(
        mark,
        '    observation={k:observation[k] for k in ("step","player","players","carbon","trees","remainingOverageTime") if k in observation}\n'
        + mark,
    )
    extracted = functions((HERE / "bridge.py").read_text())
    helper = "\n".join(
        extracted[n]
        for n in (
            "require",
            "features",
            "predict",
            "decode",
            "public_enemy",
            "estimate_view",
            "jitter",
            "leaf",
            "play_public_pair",
        )
    )
    engine = (ROOT / "code/fastsim.py").read_bytes()
    engine_blob = base64.b64encode(zlib.compress(engine, 9)).decode()
    wt = base64.b64encode((HERE / "rollout.npz").read_bytes()).decode()
    inject = f"""
import copy, io, time
from statistics import mean
_BR_FAST=types.ModuleType('_sapling_bridge_sim')
exec(zlib.decompress(base64.b64decode({engine_blob!r})),_BR_FAST.__dict__)
FastCarbon=_BR_FAST.FastCarbon
F=sys.modules['cnet2_feat']
ONEHOT=np.eye(5,dtype='f')
_BR_WEIGHTS=dict(np.load(io.BytesIO(base64.b64decode({wt!r}))))
_PAY_CFG=FastCarbon({{}}).config
"""
    append = """
_BR_OP=None
_BR_X=None
_BR_LAST={}
def _bridge_observe(view,original):
    global _BR_OP,_BR_X,_BR_LAST
    s=view["player"]
    if _BR_OP is None or view["step"]==0 or _BR_OP.seat!=1-s:
        _BR_OP=F.Feat(1-s)
    _BR_X=_BR_OP.observe(public_enemy(view,_FEAT.est))
    _BR_LAST={"base":_FEAT.to_actions(original),"status":"outside_or_ineligible"}
def _bridge_choose(view,cfg,original,other):
    if time.perf_counter()-_BR_START>.45:
        _BR_LAST.update(status="budget_skip")
        return original
    s=view["player"];views=[None,None];views[s]=_FEAT;views[1-s]=_BR_OP
    cmds=[_FEAT.to_actions(original),_FEAT.to_actions(other)]
    verdict=play_public_pair(cfg,view,views,_BR_X,cmds,_BR_WEIGHTS)
    finish=time.perf_counter()-_BR_START
    use=finish<.80 and verdict["mean_delta"]["h8"]>0
    _BR_LAST.update(status="completed" if finish<.80 else "budget_fallback",alternate=cmds[1],
                    selected=int(use),comparison=verdict,entry_s=finish)
    return other if use else original
"""
    data = (
        src + "\n" + inject + "\n" + helper + "\n" + dec + "\n" + mix + "\n" + overlay + "\n" + append
    ).encode()
    compile(data, str(out), "exec")
    assert len(data) <= 10_000_000
    out.write_bytes(data)
    return dict(
        bytes=len(data),
        sha256=sha(data),
        parent_sha256=proto["parent"],
        source_sha256=sha((HERE / "bridge.py").read_bytes()),
        weights_sha256=sha((HERE / "rollout.npz").read_bytes()),
        fastsim_sha256=sha(engine),
        overlay_sha256=sha((PAY / "overlay.py").read_bytes()),
        build_sha256=sha(Path(__file__).read_bytes()),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT)
    arg = p.parse_args()
    print(json.dumps(make(arg.out), sort_keys=True))

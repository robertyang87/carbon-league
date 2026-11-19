"""Assemble a standalone policy only after independently rebuilding the registered training release."""

import argparse
import ast
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent
BASE = TRAINING.parents[1]
ROOT = BASE.parents[1]
PARENT = ROOT / "data/weights/sapling_1200k/terminal_return_fit_v1.py"
ROUTE_FEATURES = BASE / "planter_routes_dev/features.py"
ROUTE_MODEL = TRAINING.parent / "planter_option/value_weights.json"
SOURCES = {
    PARENT: "9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0",
    TRAINING
    / "select_roots.py": "e42204000073d7169a4381cd2da10507892a462f51eede3a128af7536ac050a6",
    TRAINING
    / "opportunities.py": "2c44731ccc2480f9d7b0ead116fe4d8d1e54a1c2acafce03919b84a8d5a607a5",
    ROUTE_FEATURES: "72034f67653f7125df468bd895a6c11379fab5ecfe3cbbdbda90d3668817926d",
    ROUTE_MODEL: "59726c09984f6d2210d28ee90712cf8f0c37033ff9635bd272bdc0ca6e3edaf7",
}


class TreeResponse:
    def __init__(self, parent, market, proposals, learned, weights, windows):
        self.parent = parent
        self.market_module = market
        self.proposals = proposals
        self.learned = learned
        self.weights = weights
        self.windows = windows
        self.last = {}
        self.seat = None
        self.step = -1
        self.used = False
        self.complete_history = False

    def observe(self, obs, cfg):
        seat, step = obs["player"], obs["step"]
        if step == 0 or self.seat != seat or step != self.step + 1:
            self.market = self.market_module.PlanterRouteState()
            self.complete_history = step == 0
            if step == 0 or self.seat != seat:
                self.used = False
        self.step, self.seat = step, seat
        self.last = {"step": step, "seat": seat, "changed": False}
        if self.complete_history:
            self.market.observe(obs, seat, obs["carbon"], cfg, step)

    def pair(self, obs, st, base, uid, commitments):
        p = self.proposals
        seat = obs["player"]
        pos = st["own"][2][uid][0]
        hires, destinations, payments = commitments
        amount = sum(c for _, c in hires.values()) + sum(
            c for u, c in payments.items() if u != uid
        )
        _, price = p.quote(st, pos)
        if price is None or max(0.0, st["cash"] - amount) < price:
            return None
        occupied = {cell for u, cell in destinations.items() if u != uid} | {
            cell for cell, _ in hires.values()
        }
        if pos in occupied:
            return None
        current = base.get(uid, "NONE")
        actions = {}
        for i, cmd in enumerate(p.MOVES):
            cell = int(self.market_module._ROUTE_DEST[pos, i])
            actions[cmd] = dict(
                cell=cell,
                friendly_conflict=cell in occupied,
                enemy_collector_distance=p.nearest(cell, st["ec"]),
                enemy_planter_distance=p.nearest(cell, st["ep"]),
            )
        item = dict(
            seat=seat, t=obs["step"], uid=uid, parent_command=current, actions=actions
        )
        # This function is the comparison block extracted from the source-locked training selector.
        return p.legal_pair(self.parent, self.market, obs, base, item)

    def consider(self, obs, cfg, base):
        self.observe(obs, cfg)
        if (
            self.used
            or not self.complete_history
            or not any(lo <= self.step <= hi for lo, hi in self.windows)
        ):
            return base
        p = self.proposals
        st = p.state(obs, cfg)
        commitments = p.commitments(st, base)
        for uid in self.parent._FEAT.own_uid[:10]:
            worker = st["own"][2].get(uid)
            if worker is None or worker[2] != "PLANTER":
                continue
            pair = self.pair(obs, st, base, uid, commitments)
            if pair is None:
                continue
            pay_is_v1 = pair["paid_arm"] == "parent"
            defer = pair["new"] if pay_is_v1 else pair["old"]
            x = self.learned.features(obs, cfg, uid, "NONE", defer, base)
            decision = self.learned.choose(self.weights, x, pay_is_v1)
            if decision not in ("pay", "defer"):
                raise ValueError("invalid trained tree decision")
            if (decision == "pay") != pay_is_v1:
                self.used = True
                self.last.update(
                    changed=True,
                    worker=uid,
                    old=pair["old"],
                    new=pair["new"],
                    features=list(x),
                    decision=decision,
                    true_prefix_slot=pair["true_prefix_slot"],
                )
                return pair["alternative"]
        return base


def selected_source(path, wanted):
    source = path.read_text()
    found = {
        n.name: ast.get_source_segment(source, n)
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name in wanted
    }
    return "\n\n".join(found[name] for name in wanted)


def runtime_source():
    """Build source of the real wrapper; the original v1 binary and fitted weights are supplied separately."""
    for path, digest in SOURCES.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"changed registered source: {path}")
    v1 = PARENT.read_text()
    mixed = v1[v1.rindex("\ndef _mixed_actions(") :]
    mixed = mixed.replace(
        "return baseline", "return _capture(x,aux,enc,baseline,False)"
    ).replace("return candidate", "return _capture(x,aux,enc,candidate,True)")
    decoder = (
        v1[v1.index("def _return_decode(") :].split("\n    return labels", 1)[0]
        + "\n    return labels"
    )
    decoder = decoder.replace(
        "_return_decode(x, aux, encoded, apply_residual)",
        "_forced_decode(x, aux, encoded, apply_residual, target_slot, target_action)",
    )
    a, b = decoder.index("        if not apply_residual:"), decoder.index(
        "        labels[slot] = action"
    )
    decoder = decoder[:a] + """        if slot == target_slot:
            if slot>=10 or x['tok_kind'][slot]!=1 or not legal[target_action]:
                return None
            action=target_action
""" + decoder[b:]
    helper = selected_source(
        TRAINING / "opportunities.py",
        ["norm", "neighbors", "nearest", "state", "quote", "commitments"],
    )
    training = selected_source(TRAINING / "select_roots.py", ["route"])
    block = training[training.index("    x,aux,enc,original,mode=net._root") :]
    shared = (
        helper
        + "\n\ndef legal_pair(net,market,obs,base,item):\n    s,t,uid=item['seat'],item['t'],item['uid']\n"
        + block
    )
    # Private modules keep the parent and the training helper namespaces disjoint.
    market = ROUTE_FEATURES.read_text()
    policy = (HERE / "policy.py").read_text()
    route_weights = json.loads(ROUTE_MODEL.read_bytes())
    windows = json.loads((TRAINING / "protocol.json").read_bytes())["windows"]
    return mixed + "\n" + decoder + "\n" + inspect.getsource(TreeResponse) + f"""
from collections import Counter
from types import SimpleNamespace

def _tree_module(name, source, bindings):
    module = types.ModuleType(name)
    module.__dict__.update(bindings)
    exec(compile(source, name, "exec"), module.__dict__)
    return module

_tree_market = _tree_module("public_planter_market",{market!r},{{}})
_tree_policy = _tree_module("public_terminal_response",{policy!r},{{}})
_tree_proposals = _tree_module("frozen_training_route",{shared!r},dict(
    Counter=Counter, np=np, sys=sys, MOVES=['NONE','UP','RIGHT','DOWN','LEFT'], BANK=('RECCOLLECTOR','RECPLANTER'),
    MAXDIST=15, _ROUTE_DEST=_tree_market._ROUTE_DEST, _ROUTE_DIST=_tree_market._ROUTE_DIST,
    clean=lambda a: {{k:v for k,v in a.items() if v and v!='NONE'}}, route_value=_tree_market.route_value, MODEL={route_weights!r}))
_tree_parent = SimpleNamespace(_forced_decode=_forced_decode,_return_conflicts=_return_conflicts)
def _capture(x,aux,enc,labels,mode):
    _tree_parent._root=(x,aux,enc,labels,mode)
    return labels

_tree_original = agent
_tree_head = TreeResponse(_tree_parent,_tree_market,_tree_proposals,_tree_policy,_tree_weights,{windows!r})

def agent(observation,configuration=None):
    base = _tree_original(observation,configuration)
    _tree_parent._FEAT = _FEAT
    if configuration is None:
        return base
    return _tree_head.consider(observation,configuration,base)
"""


def publish(out):
    from train import rebuild_release

    weights = rebuild_release(intake_root=TRAINING)
    if not isinstance(weights, dict) or weights != json.loads(
        (HERE / "weights.json").read_bytes()
    ):
        raise ValueError("rebuilt training release differs from weights")
    # The original package and both physical continuations must still match the frozen witnesses.
    subprocess.run(
        [
            sys.executable,
            str(HERE / "test_deploy.py"),
            "TestDeployment.test_eight_ordered_pairs_and_rejections",
        ],
        env={
            **os.environ,
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        check=True,
    )
    # Compile before exclusive creation so a failed build cannot replace an existing agent.
    data = (
        PARENT.read_bytes()
        + b"\n_tree_weights = "
        + repr(weights).encode()
        + b"\n"
        + runtime_source().encode()
    )
    compile(data, str(out), "exec")
    if len(data) > 10_000_000:
        raise ValueError("single-file size budget exceeded")
    import tempfile

    out = Path(out)
    with tempfile.NamedTemporaryFile(dir=out.parent, delete=False) as temp:
        temp.write(data)
        scratch = Path(temp.name)
    try:
        try:
            os.link(scratch, out)
        except FileExistsError:
            if out.read_bytes() != data:
                raise
    finally:
        scratch.unlink()
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    print(publish(parser.parse_args().out))

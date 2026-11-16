"""hybrid_m3s2search -- OUR policy pack proposes the move, the published Master3s2 per-turn search referees it
(diagnostic: does a joint-move rollout search of that design add strength on top of our actor?).

Each turn: the base pack (HYB_BASE, default packs/echo4_int8.py) returns its joint action; Master3s2's own
_act runs as published (belief tracker, its policy logits for candidate sampling), but the candidate-0 /
protected move handed to its _search is OUR action instead of its policy's sample. The search then plays its
usual candidates (temperature samples of ITS policy + single-worker deviations of candidate 0) and switches
only when a rollout value beats candidate 0. HYB_SEARCH=0 returns our action untouched (control = base pack).
Not a deployable artifact: it embeds another author's published strategy file; any use beyond local
diagnostics is the owner's decision.
"""
import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(path, tag):
    spec = importlib.util.spec_from_file_location(f"{tag}_{os.getpid()}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_BASE = _load(os.path.join(_HERE, "..", os.environ.get("HYB_BASE", "packs/echo4_int8.py")), "hyb_base")
_M = _load(os.path.join(_HERE, "..", "..", "sources", "master3s2", "master3s2.py"), "hyb_m3s2")
_ORIG_SEARCH = _M._search
_OURS = {}
STATS = {"turns": 0, "changed": 0}


def _patched(obs, configuration, est, act0, lw0, lb0, prev, t0, home=()):
    ours = _OURS.get("act")
    if ours is None:
        return _ORIG_SEARCH(obs, configuration, est, act0, lw0, lb0, prev, t0, home)
    out = _ORIG_SEARCH(obs, configuration, est, dict(ours), lw0, lb0, prev, t0, ())
    STATS["turns"] += 1
    STATS["changed"] += int(out != ours)
    return out


_M._search = _patched


def agent(observation, configuration):
    ours = dict(_BASE.agent(observation, configuration) or {})
    if os.environ.get("HYB_SEARCH", "1") == "0":
        return ours
    _OURS["act"] = ours
    t0 = _M._time.perf_counter()
    try:
        return _M._act(observation, configuration, t0)
    except Exception:
        if os.environ.get("CARBON_DEBUG"):
            raise
        return ours

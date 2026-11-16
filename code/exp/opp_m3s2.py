"""opp_m3s2 -- the published Master3s2 source (sources/master3s2/master3s2.py, md5 57f9a594) as a local
opponent, with ablation switches for diagnostics (never a training-pool artifact by itself):
  M3S2_SEARCH=0   policy only (per-turn search off)
  M3S2_HOME=0     go_home end-game rule off (collectors never forced home)
  M3S2_TEMP=<t>   sampling temperature of the policy (0 = argmax); default as published (1.0)
The module is loaded fresh per load (its belief tracker / rng state is per game anyway)."""
import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "..", "sources", "master3s2", "master3s2.py")
_spec = importlib.util.spec_from_file_location(f"m3s2_{os.getpid()}_{id(_HERE)}", _SRC)
_M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_M)
if os.environ.get("M3S2_SEARCH", "1") == "0":
    _M._SEARCH_OK = False
if os.environ.get("M3S2_HOME", "1") == "0":
    _M.go_home = lambda *a, **k: None
if os.environ.get("M3S2_TEMP") is not None:
    _M._TEMP = float(os.environ["M3S2_TEMP"])


def agent(observation, configuration):
    return _M.agent(observation, configuration)


def search_stats():
    return dict(getattr(_M, "_S_STATS", {}))

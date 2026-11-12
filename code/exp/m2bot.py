"""m2bot: whole-game rule bot fitted to RyanX-AIPowercore-M2 replays (code/m2_bot.py), no neural net.
cfg: json in env M2BOT_CFG (or base64 json in M2BOT_CFG_B64); default = m2_bot.DEFAULT"""
import base64, json, os, sys
_C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _C)
import m2_bot as _B

_cfg = os.environ.get("M2BOT_CFG_B64")
_cfg = json.loads(base64.b64decode(_cfg)) if _cfg else json.loads(os.environ.get("M2BOT_CFG", "{}"))
agent = _B.make_agent(_cfg)   # exp modules are loaded fresh per game

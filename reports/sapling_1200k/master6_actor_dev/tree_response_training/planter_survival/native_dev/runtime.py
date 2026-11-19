"""Overlay the fitted one-transition survival veto on frozen v1's actual decisions."""

from collections import deque
import hashlib

_SURV_ORIGINAL = agent
_SURV_MIXED = _mixed_actions
_SURV_CONTEXT = None
_SURV_HISTORY = deque(maxlen=3)
_SURV_TRACE = {}


def _mixed_actions(x, aux):
    global _SURV_CONTEXT
    _SURV_CONTEXT = x, aux
    return _SURV_MIXED(x, aux)


def _surv_proposals(obs, cfg, base):
    feat = _FEAT
    seat = obs["player"]
    own = obs["players"][seat]
    if not any(uid in own[2] and own[2][uid][2] == "PLANTER" and not base.get(uid) for uid in feat.own_uid):
        return []
    f = sys.modules["cnet2_feat"]
    st = _SURV_RULES.state(obs, cfg)
    hires, dest, required = _SURV_RULES.commitments(st, base)
    reserved_hires = sum(cost for _, cost in hires.values())
    pref = f.Prefix(*_SURV_CONTEXT)
    labels = feat.labels(base)
    options = []
    for slot, uid in enumerate(feat.own_uid):
        if uid is None:
            continue
        _, _, legal = pref.feats(slot)
        a = int(labels[slot])
        if not legal[a]:
            raise ValueError("v1 emitted an illegal prefix")
        if uid in required and not base.get(uid) and own[2][uid][2] == "PLANTER":
            origin = own[2][uid][0]
            available = max(
                0.0,
                st["cash"] - reserved_hires - sum(cost for other, cost in required.items() if other != uid),
            )
            blocked = {cell for other, cell in dest.items() if other != uid} | {
                cell for cell, _ in hires.values()
            }
            if required[uid] > 0 and required[uid] <= available and legal[0] and origin not in blocked:
                moves = []
                for i, cmd in enumerate(_SURV_RULES.MOVES[1:], 1):
                    canonical = f.CANON_MOVE[i] if seat else i
                    if legal[canonical] and _SURV_MODEL.destination(origin, cmd) not in blocked:
                        moves.append(cmd)
                if moves:
                    suffix = f"{seat}|{obs['step']}|{uid}|"
                    move = min(moves, key=lambda cmd: hashlib.sha256((suffix + cmd).encode()).digest())
                    hav = _SURV_MODEL.death_risk(_SURV_WEIGHTS, obs, list(_SURV_HISTORY), base, uid, "NONE")
                    mov = _SURV_MODEL.death_risk(_SURV_WEIGHTS, obs, list(_SURV_HISTORY), base, uid, move)
                    options.append(
                        {
                            "uid": uid,
                            "old": "NONE",
                            "move": move,
                            "p_stay": hav,
                            "p_move": mov,
                            "trigger": hav >= 0.20 and hav >= mov + 0.05,
                        }
                    )
        pref.advance(slot, a)
    return sorted(options, key=lambda item: item["uid"])


def agent(observation, configuration=None):
    global _SURV_TRACE
    base = _SURV_ORIGINAL(observation, configuration)
    seat, t = observation["player"], observation["step"]
    if t == 0 or _SURV_TRACE.get("seat") != seat or _SURV_TRACE.get("step") != t - 1:
        _SURV_HISTORY.clear()
        complete = t == 0
    else:
        complete = _SURV_TRACE["complete"]
    obs = {k: observation[k] for k in ("player", "step", "players", "carbon", "trees")}
    options = _surv_proposals(obs, configuration, base) if configuration is not None and complete else []
    choice = next((r for r in options if r["trigger"]), None)
    _SURV_TRACE = {
        "step": t,
        "seat": seat,
        "v1_commands": base.copy(),
        "opportunities": options,
        "selected": None if choice is None else choice["uid"],
        "complete": complete,
    }
    # Only visible worker locations are needed to reproduce the three-turn motion features.
    past = [[None, None, dict((u, tuple(w)) for u, w in p[2].items())] for p in obs["players"]]
    _SURV_HISTORY.append({"player": seat, "step": t, "players": past})
    if choice is None:
        return base
    return {**base, choice["uid"]: choice["move"]}

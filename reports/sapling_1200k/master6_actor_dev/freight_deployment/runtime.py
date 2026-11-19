"""Single public counterfactual layered over a fully decoded, frozen two-network candidate."""
from collections import deque
from copy import deepcopy


def public_slice(obs):
    return {"step": int(obs["step"]), "player": int(obs["player"]), "players": deepcopy(obs["players"])}


class FreightOverlay:
    def __init__(self, predictor, policy):
        self.predictor = predictor
        self.policy = policy
        self.history = deque(maxlen=5)
        self.last_step = -1
        self.seat = None
        self.used = False
        self.last = {}

    def consider(self, obs, cfg, base_actions, reselect):
        t, seat = int(obs["step"]), int(obs["player"])
        if t == 0 or t <= self.last_step or seat != self.seat:
            self.history.clear()
            self.used = False
        elif t != self.last_step + 1:
            self.history.clear()
        self.last_step, self.seat = t, seat
        old_history = tuple(self.history)
        current = public_slice(obs)
        self.history.append(current)
        self.last = {"step": t, "seat": seat, "changed": False, "public_predictions": {}}
        def keep(reason):
            self.last["reason"] = reason
            return base_actions
        if self.used or not 219 <= t <= 289:
            return keep("already_changed" if self.used else "outside_window")
        rule = {k: cfg.get(k, default) for k, default in (("size", 15), ("moveCost", 0), ("smeltCost", 0))}
        if rule["size"] != 15 or rule["moveCost"] != 0 or rule["smeltCost"] != 0:
            return keep("unvalidated_rules")
        forecasts = self.predictor.predict(current, old_history)
        self.last["public_predictions"] = forecasts
        offer = self.policy.choose_intervention(current, rule, seat, base_actions, forecasts)
        if offer is None:
            return keep("no_shared_offer")
        uid, action = offer["worker"], offer["alternative"]
        self.last["proposal"] = {k: offer[k] for k in ("worker", "target_worker", "alternative", "landing", "rank_score")}
        reason = self.veto(current, base_actions, offer)
        if reason:
            return keep(reason)
        # Reject a changed suffix even if every command would be individually legal.
        modified = dict(base_actions)
        if action == "NONE":
            modified.pop(uid, None)
        else:
            modified[uid] = action
        confirmed = reselect(uid, action)
        if confirmed != modified:
            return keep("changed_suffix_or_illegal_prefix")
        self.used = True
        self.last.update(changed=True, reason="committed")
        return modified

    def veto(self, obs, actions, offer):
        s = obs["player"]
        me, enemy = obs["players"][s], obs["players"][1-s]
        uid, alternative = offer["worker"], offer["alternative"]
        pack = self.policy
        origins = me[2]
        if uid not in origins or origins[uid][2] != "COLLECTOR" or alternative not in pack.MOVES:
            return "invalid_worker_or_action"
        pos, cargo, _ = origins[uid]
        old = actions.get(uid, "NONE")
        previous = pack.destination(pos, old, 15)
        landing = pack.destination(pos, alternative, 15)
        if old == alternative or landing != offer["landing"]:
            return "stale_command"
        other = {pack.destination(w[0], actions.get(u, "NONE"), 15) for u, w in origins.items() if u != uid}
        recruiting = {cell for base, cell in me[1].items() if actions.get(base) in ("RECCOLLECTOR", "RECPLANTER")}
        if landing in other or landing in recruiting or landing in enemy[1].values():
            return "reserved_landing"
        if cargo > 0 and previous in me[1].values() and landing not in me[1].values():
            return "forgone_immediate_bank"
        if (previous not in other and previous in {v[0] for v in me[3].values()}
                and any(w[2] == "PLANTER" and pack.D[previous][w[0]] <= 1 for w in enemy[2].values())):
            return "uncovered_contested_tree"
        return None

"""Estimate an opponent planter's stay probability using evidence from this game.

One instance per match and observer seat. Call observe before evaluating actions.
Only an enemy visible at both consecutive steps supplies an unambiguous label.
"""
import math

import numpy as np

PRIOR_WEIGHT = 6.0
HALF_LIFE = 40.0


def adjacent_or_same(cell, size):
    r, c = divmod(cell, size)
    return (cell, ((r - 1) % size) * size + c, ((r + 1) % size) * size + c,
            r * size + (c - 1) % size, r * size + (c + 1) % size)


def opportunities(obs, config):
    """uid -> (public pre-action cell, whether an existing defender is in reach)."""
    own, enemy = [obs["players"][s] for s in (obs["player"], 1 - obs["player"])]
    if enemy[0] < config["seizeCost"]:
        return {}
    trees = {pos for pos, age in own[3].values() if age < config["treeLifespan"]}
    ours = {w[0] for w in own[2].values()}
    return {uid: (pos, not ours.isdisjoint(adjacent_or_same(pos, config["size"])))
            for uid, (pos, _, role) in enemy[2].items() if role == "PLANTER" and pos in trees}


class AdaptiveTreeIntent:
    def __init__(self):
        self.reset()

    def reset(self):
        self.seat = None
        self.step = None
        self.pending = {}
        self.counts = np.zeros((2, 2), dtype=float)  # context x [stays, resolves]
        self.learned = 0
        self.censored = 0

    def observe(self, obs, cfg):
        seat, step = obs["player"], obs["step"]
        if self.step is not None and (seat != self.seat or step < self.step):
            self.reset()
        if step == self.step:
            return
        if self.step is not None:
            gap = step - self.step
            self.counts *= math.exp2(-gap / HALF_LIFE)
            if gap == 1:
                enemy = obs["players"][1 - seat][2]
                for uid, (old_position, covered) in self.pending.items():
                    if uid not in enemy:
                        self.censored += 1
                        continue
                    self.counts[int(covered)] += (int(enemy[uid][0] == old_position), 1)
                    self.learned += 1
        self.seat, self.step = seat, step
        self.pending = opportunities(obs, cfg)

    def probabilities(self, obs, cfg, frozen_map):
        if (obs["player"], obs["step"]) != (self.seat, self.step):
            raise ValueError("observe this seat and step before predicting")
        result = frozen_map.copy()
        for uid, (_, covered) in self.pending.items():
            v = np.asarray(frozen_map[uid], dtype=float)
            if v.shape != (5,) or not np.all(np.isfinite(v)) or np.min(v) < 0 or not np.isclose(v.sum(), 1):
                raise ValueError("invalid frozen action probabilities")
            stays, total = self.counts[int(covered)]
            if total == 0:
                continue
            p = (PRIOR_WEIGHT * v[0] + stays) / (PRIOR_WEIGHT + total)
            rest = v[1:].sum()
            directions = v[1:] / rest if rest else np.full(4, .25)
            result[uid] = np.r_[p, (1-p) * directions]
        return result

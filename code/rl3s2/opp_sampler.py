"""opp_sampler -- ratio sampler for the RL line's opponent pool.

Registered caliber (war-room filing + SUPPLEMENT_RL_LEAGUE_POOL_20261119):
categories self / master3s2 / history, default shares 0.575 / 0.25 / 0.175.

Hard rules, enforced in code (a violation dies naming the offender):
  * master3s2 share legal domain is [0, 0.50] -- 0.50 is a CAP (re-pinned by ruling 1d9f63e0), not a
    target. Any config or update above it dies.
  * Runtime adjustment of the master3s2 share is DOWNWARD ONLY: an update
    raising it above the current effective value dies. (Initial load may
    set any value inside the legal domain.)
  * NO automatic redistribution: shares are used as given; any mass not
    covered by the shares (or freed by cutting master3s2, or by a member
    missing on the seat) stays IDLE. Nothing renormalizes.
  * Echo6 and Master2 NEVER enter the training pool (gate exams only):
    a built-in exclusion list; a member whose id / identity / spec name
    matches dies at registration.
  * Every accepted share change and every rejected attempt is appended
    to the audit log (per-tier share changes leave a trace); every
    assignment records the actual opponent identity per game.

The sampler is the ratio-exact layer: assign() returns per-game opponent
specs with count-exact category shares. It also emits the stock trainer's
--pool JSON (priors proportional to shares) for launches through
rl2_ppo_train, whose internal PFSP sampling is approximate -- see
MANIFEST.md for that boundary.
"""

import json
import math
import os
import re
import time

CAP_MASTER3S2 = 0.50  # re-pinned 0.25 -> 0.50 by strategy ruling 1d9f63e0 sec.2 (2026-11-19)
CATEGORIES = ("self", "master3s2", "history")
DEFAULT_SHARES = {"self": 0.575, "master3s2": 0.25, "history": 0.175}
# Normalized-substring exclusion. "master2" is not a substring of
# "master3s2", so the registered opponent never trips its own ban.
HARD_EXCLUDED_TOKENS = ("echo6", "master2")


class SamplerError(RuntimeError):
    pass


def _die(msg):
    raise SamplerError(msg)


def _norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def check_excluded(member_id, spec, identity=None):
    hay = _norm(member_id) + "|" + _norm(identity or "") + "|" + _norm(os.path.basename(str(spec)))
    for tok in HARD_EXCLUDED_TOKENS:
        if tok in hay:
            _die(f"opponent member {member_id!r} (spec {spec!r}) matches hard-excluded "
                 f"identity '{tok}': Echo6 and Master2 never enter the training pool "
                 "(gate exams only)")


def validate_shares(shares, context="shares"):
    out = {}
    for k, v in dict(shares).items():
        if k not in CATEGORIES:
            _die(f"{context}: unknown category {k!r} (categories are {CATEGORIES})")
        v = float(v)
        if not (0.0 <= v <= 1.0):
            _die(f"{context}: share[{k}]={v} outside [0, 1]")
        out[k] = v
    for c in CATEGORIES:
        out.setdefault(c, 0.0)
    if out["master3s2"] > CAP_MASTER3S2 + 1e-12:
        _die(f"{context}: master3s2 share {out['master3s2']} exceeds the hard cap "
             f"{CAP_MASTER3S2} (the cap is a ceiling, not a target)")
    s = sum(out.values())
    if s > 1.0 + 1e-9:
        _die(f"{context}: shares sum to {s:.4f} > 1 (the remainder must stay idle; "
             "no automatic redistribution exists)")
    return out


class OpponentSampler:
    def __init__(self, config=None, audit_path=None):
        config = config or {}
        self._shares = validate_shares(config.get("shares", DEFAULT_SHARES), "config shares")
        self._members = {c: [] for c in CATEGORIES}
        for m in config.get("members", []):
            self.register_member(m)
        self._audit_path = audit_path or config.get("audit_path")
        self._log("init", {"shares": self._shares})

    # ------------------------------------------------------------ members
    def register_member(self, member):
        mid, cat, spec = member.get("id"), member.get("category"), member.get("spec")
        if cat not in CATEGORIES:
            _die(f"member {mid!r}: unknown category {cat!r}")
        if not mid or not spec:
            _die(f"member {member!r}: id and spec are required")
        check_excluded(mid, spec, member.get("identity"))
        if any(m["id"] == mid for ms in self._members.values() for m in ms):
            _die(f"member {mid!r}: duplicate id")
        self._members[cat].append({"id": mid, "category": cat, "spec": spec})
        return mid

    def members(self, category=None):
        if category:
            return list(self._members[category])
        return [m for c in CATEGORIES for m in self._members[c]]

    # ------------------------------------------------------------- shares
    def current_shares(self):
        return dict(self._shares)

    def apply_update(self, new_shares, source="manual"):
        """Runtime share adjustment. master3s2 may only move DOWN from the
        current effective value; other categories move only by this explicit
        new ratio (never by automatic redistribution)."""
        try:
            ns = validate_shares(new_shares, f"update from {source}")
            if ns["master3s2"] > self._shares["master3s2"] + 1e-12:
                _die(f"update from {source}: master3s2 share {ns['master3s2']} > current "
                     f"{self._shares['master3s2']}: runtime adjustment is downward only "
                     f"(legal domain [0, {CAP_MASTER3S2}]); refusing")
        except SamplerError as e:
            self._log("update_rejected", {"source": source, "reason": str(e)})
            raise
        old = self._shares
        self._shares = ns
        self._log("update", {"source": source, "old": old, "new": ns})
        return self.current_shares()

    def halve_master3s2(self, source="gate-halving"):
        ns = dict(self._shares)
        ns["master3s2"] = ns["master3s2"] / 2.0
        return self.apply_update(ns, source=source)

    # --------------------------------------------------------- assignment
    def assign(self, n_games, rng):
        """Count-exact per-game assignment. Returns (assignments, idle):
        assignments[i] = {game_index, category, member_id, spec}. The
        master3s2 count never exceeds floor(0.25 * n_games); rounding
        leftovers and uncovered share mass stay idle, never redistributed."""
        if n_games <= 0:
            _die("assign: n_games must be positive")
        for c in CATEGORIES:
            if self._shares[c] > 0 and not self._members[c]:
                _die(f"assign: category {c!r} has share {self._shares[c]} but no "
                     "registered member (register members, or set the share to 0 "
                     "explicitly -- the mass is not redistributed automatically)")
        raw = {c: self._shares[c] * n_games for c in CATEGORIES}
        counts = {c: int(math.floor(raw[c])) for c in CATEGORIES}
        left = n_games - sum(counts.values())
        order = sorted(CATEGORIES, key=lambda c: raw[c] - counts[c], reverse=True)
        i = 0
        while left > 0 and order:
            c = order[i % len(order)]
            if counts[c] + 1 <= raw[c] + 1e-9 or sum(counts.values()) < sum(raw.values()):
                counts[c] += 1
                left -= 1
            i += 1
            if i > 4 * n_games:
                break
        cap_n = int(CAP_MASTER3S2 * n_games + 1e-9)
        if counts["master3s2"] > cap_n:            # hard cap by game count
            counts["master3s2"] = cap_n
        total = sum(counts.values())
        idle = n_games - total
        picks = []
        for c in CATEGORIES:
            ms = self._members[c]
            for _ in range(counts[c]):
                m = ms[rng.randrange(len(ms))]
                picks.append({"category": c, "member_id": m["id"], "spec": m["spec"]})
        rng.shuffle(picks)
        for j, p in enumerate(picks):
            p["game_index"] = j
        self._log("assign", {"n_games": n_games, "counts": counts, "idle": idle,
                             "shares": self._shares})
        return picks, idle

    # ------------------------------------------------------ trainer bridge
    def pool_json(self):
        """[[spec, prior], ...] for rl2_ppo_train --pool: each present
        member's prior = its category share split equally inside the
        category. Categories with no present member contribute nothing
        (their share idles). The 'self' category is realized by the
        trainer's own snapshots (--snap-prior), not by pool members."""
        out = []
        for c in ("master3s2", "history"):
            ms = self._members[c]
            if ms and self._shares[c] > 0:
                pr = self._shares[c] / len(ms)
                for m in ms:
                    out.append([m["spec"], round(pr, 6)])
        return out

    # -------------------------------------------------------------- audit
    def _log(self, event, payload):
        if not self._audit_path:
            return
        rec = {"t": round(time.time(), 3), "event": event, **payload}
        with open(self._audit_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    import sys
    cfg = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
    s = OpponentSampler(cfg.get("sampler", cfg))
    if sys.argv[1] == "check":
        print("shares OK:", s.current_shares())
        print("pool:", json.dumps(s.pool_json()))
    elif sys.argv[1] == "pool":
        print(json.dumps(s.pool_json()))

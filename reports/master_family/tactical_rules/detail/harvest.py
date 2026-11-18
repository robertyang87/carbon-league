"""Visible-ground harvest, continuation and collision-avoidance choices; run via mine.py.

All explanatory variables are from decision t or the same worker's earlier action.
The submitted action lives in t+1; disappearance after a move never removes a
decision. The field has no tree or depot at t. Pre-action cargo comparisons use
the logged default zero move cost. No hidden map or earlier terminal selection.
"""
from collections import Counter
import hashlib
import statistics

SIZE = 15
MOVES = {None: (0, 0), "UP": (-1, 0), "RIGHT": (0, 1), "DOWN": (1, 0), "LEFT": (0, -1)}
CARDINAL = ("UP", "RIGHT", "DOWN", "LEFT")


def shift(pos, dr, dc):
    row, col = divmod(pos, SIZE)
    return ((row + dr) % SIZE) * SIZE + (col + dc) % SIZE


DEST = [{a: shift(pos, *offset) for a, offset in MOVES.items()} for pos in range(SIZE ** 2)]
ADJ = [tuple(row[a] for a in CARDINAL) for row in DEST]
NINE = [tuple(shift(pos, dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
        for pos in range(SIZE ** 2)]
INF = float("inf")
PAIRINGS = {
    "rich_minus_lean": ("ground_rich", "ground_lean"),
    "cargo_heavy_minus_light_on_rich": ("cargo_heavy", "cargo_light"),
    "threat_minus_no_reachable_collector": ("threat_lighter_or_tie", "threat_none"),
    "threat_lighter_minus_only_heavier": ("threat_lighter_or_tie", "threat_only_heavier"),
    "repeat_rich_minus_repeat_lean": ("repeat_rich", "repeat_lean"),
}


def record(metrics, name, decision):
    count = metrics.setdefault(name, [0, 0])
    count[0] += bool(decision)
    count[1] += 1


def observe(replay, seat, key):
    cfg = replay["configuration"]
    if (cfg["size"], cfg["moveCost"], cfg["initialCollectRate"], cfg["collectDecreaseRate"]) != (15, 0, .3, 0):
        raise ValueError(f"Unaccounted game configuration {key}")
    steps = replay["steps"]
    start = steps[0][0]["observation"]["players"]
    own_bases = set(start[seat][1].values())
    enemy_bases = set(start[1-seat][1].values())
    bases = own_bases | enemy_bases
    near_home = own_bases | {pos for home in own_bases for pos in ADJ[home]}
    metrics, example = {}, {}
    take_examples = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 31 == 0
    evade = [0, 0, 0]  # selected safe, opportunities, total safe cardinal destinations
    accounting = Counter()
    previous_stays = set()
    for t, (current, following) in enumerate(zip(steps, steps[1:])):
        shared = current[0]["observation"]
        assert shared["step"] == t
        own, enemy = shared["players"][seat], shared["players"][1-seat]
        # Collector always observes its own cell, including a true zero.
        visible_carbon = current[seat]["observation"]["carbon"]
        actions = following[seat]["action"] or {}
        following_players = following[0]["observation"]["players"]
        after = following_players[seat][2]
        enemy_trees_after = {tree[0] for tree in following_players[1-seat][3].values()}
        trees = {tree[0] for player in (own, enemy) for tree in player[3].values()}
        occupied = Counter(w[0] for player in (own, enemy) for w in player[2].values())
        min_enemy = {}
        for cell, cargo, kind in enemy[2].values():
            if kind == "COLLECTOR":
                for dst in DEST[cell].values():
                    if cargo < min_enemy.get(dst, INF):
                        min_enemy[dst] = cargo
        staying = set()
        for uid, (pos, cargo, kind) in own[2].items():
            if kind != "COLLECTOR":
                continue
            action = actions.get(uid) or None
            if action not in MOVES:
                raise ValueError(f"Invalid worker action {key}, step {t}")
            stay = action is None
            if stay:
                staying.add(uid)
            if pos in bases or pos in trees:
                continue
            ground = visible_carbon[pos]
            least = min_enemy.get(pos, INF)
            threatened = least <= cargo
            rich = ground >= 60
            lean = ground < 10
            # Fix game stage, home proximity and cargo when comparing resource density.
            if t < 200 and pos not in near_home:
                if not threatened and cargo < 30:
                    band = "lean" if lean else "rich" if rich else "mid" if ground >= 30 else "low"
                    record(metrics, "ground_" + band, stay)
                if not threatened and rich:
                    band = "heavy" if cargo >= 100 else "light" if cargo < 30 else "medium"
                    record(metrics, "cargo_" + band, stay)
                if 0 < cargo < 100 and ground >= 30:
                    condition = "lighter_or_tie" if threatened else "none" if least == INF else "only_heavier"
                    record(metrics, "threat_" + condition, stay)
                if not threatened and uid in previous_stays:
                    if lean or rich:
                        record(metrics, "repeat_lean" if lean else "repeat_rich", stay)
            # Attacker is one step from the current cell. Restrict to genuine
            # choices with both safe and dangerous exits, away from enemy depots
            # where a new enemy collector could spawn. Compare to a uniform move.
            if threatened and cargo > 0 and not stay and enemy_bases.isdisjoint(ADJ[pos]):
                safe_exits = [dst for dst in ADJ[pos] if min_enemy.get(dst, INF) > cargo]
                if 0 < len(safe_exits) < 4:
                    safe = DEST[pos][action] in safe_exits
                    evade[0] += safe
                    evade[1] += 1
                    evade[2] += len(safe_exits)
                    if take_examples and safe and "safe_exit" not in example:
                        example["safe_exit"] = dict(t=t, uid=uid, cell=pos, action=action,
                                                    cargo=cargo, lowest_reachable_enemy_cargo=least,
                                                    safe_exits=len(safe_exits))
            if stay and ground > 0:
                # No collision can add cargo; no surviving enemy tree can levy
                # a toll. At positive carbon, observed gain should be 0.3*C
                # up to replay rounding. Both outcomes and exceptions counted.
                if occupied[pos] == 1 and all(occupied[dst] == 0 for dst in ADJ[pos]) and enemy_trees_after.isdisjoint(NINE[pos]):
                    accounting["clean_positive_ground_stays"] += 1
                    if uid in after:
                        gain = after[uid][1] - cargo
                        accounting["survived"] += 1
                        accounting["positive_cargo_delta"] += gain > 0
                        accounting["rule_gain_within_002"] += abs(gain - .3*ground) <= .02
                        if take_examples and rich and "observable_harvest" not in example:
                            example["observable_harvest"] = dict(t=t, uid=uid, cell=pos, ground=ground,
                                                                 cargo_before=cargo, cargo_after=after[uid][1])
        previous_stays = staying
    return {"choices": metrics, "escape": evade, "accounting": dict(accounting), "examples": example}


def fraction(hit, n):
    return round(hit / n, 5) if n else None


def aggregate(rows):
    names = sorted({m for row in rows for m in row["observation"]["choices"]})
    counts = {}
    for name in names:
        pairs = [row["observation"]["choices"].get(name, (0, 0)) for row in rows]
        hit, n = map(sum, zip(*pairs))
        counts[name] = {"stay": hit, "decisions": n, "games_exposed": sum(t > 0 for _, t in pairs),
                        "rate": fraction(hit, n)}
    paired = {}
    for label, (high, low) in PAIRINGS.items():
        diffs = []
        for row in rows:
            stats = row["observation"]["choices"]
            a, na = stats.get(high, (0, 0))
            b, nb = stats.get(low, (0, 0))
            if na and nb:
                diffs.append(a / na - b / nb)
        paired[label] = {"eligible_appearances": len(diffs), "above": sum(d > 0 for d in diffs),
                         "below": sum(d < 0 for d in diffs),
                         "mean_game_delta": round(statistics.mean(diffs), 5) if diffs else None}
    hit, n, credit = map(sum, zip(*(r["observation"]["escape"] for r in rows)))
    accounting = sum((Counter(r["observation"]["accounting"]) for r in rows), Counter())
    examples = {}
    for row in rows:
        for label, details in row["observation"]["examples"].items():
            if label not in examples:
                examples[label] = {"key": row["key"], "seat": row["seat"], **details}
    return {"appearances": len(rows), "stay_choices": counts, "paired_contrasts": paired,
            "escape": {"safe_chosen": hit, "decisions": n, "pooled_rate": fraction(hit, n),
                       "uniform_cardinal_reference": fraction(credit, n * 4)},
            "observed_clean_harvest": dict(accounting), "examples": examples}


def summarize(rows):
    return {
        "definitions": {
            "choices": "Submitted STAY among existing collectors on cells with no tree or depot. Ground is from the collector's own seat's masked carbon at the current cell. Stage t<200. Distance to own nearest base >=2.",
            "ground": "Own cargo <30, no reachable enemy collector lighter than or tied with ours. Carbon bands [0,10),[10,30),[30,60),[60,infinity).",
            "cargo": "Ground >=60, no reachable lighter or equal enemy collector. Own cargo [0,30),[30,100),[100,infinity).",
            "threat": "0<cargo<100, ground>=30. No reachable enemy collector, only strictly heavier ones, or at least one lighter/tied; reach includes the present cell and cardinal neighbors.",
            "repeat": "Same uid also submitted STAY at previous decision, with current lighter/equal enemy reach absent. Contrast field carbon <10 versus >=60; cargo unrestricted.",
            "escape": "Any stage, loaded collector leaving a presently threatened field; 1–3 of four cardinal exits safe from all reachable lighter/equal enemy collectors, no adjacent enemy depot. Compare actual chosen safe-exit rate to uniform over four exits on same decision. Excludes stayed, no-safe and all-safe cases. Ignores friendly conflicts and economic value.",
            "accounting": "All stages; positive-ground stay on a field with no other standing worker at or within one step of it before action and no live enemy tree in the post-state 3x3. Survivors' observed cargo delta compared with the rule 0.3*C to tolerance 0.02, not inferred by counting all waits.",
            "units": "Repeated decisions inside games. Paired contrasts use equal weight among model appearances with both observed conditions; choice does not establish causal utility.",
        },
        "models": {model: aggregate([r for r in rows if r["model"] == model])
                   for model in sorted({r["model"] for r in rows})},
    }

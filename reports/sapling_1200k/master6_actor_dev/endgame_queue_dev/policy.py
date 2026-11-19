"""Terminal joint cash router. The decision interface accepts one public seat and v1's offer."""
from functools import lru_cache
from itertools import combinations, product
from math import inf
from time import process_time

from fastsim import DEFAULT_CONFIG, FastCarbon


NAMES = ("NONE", "UP", "RIGHT", "DOWN", "LEFT")
OFFSETS = ((0, 0), (-1, 0), (0, 1), (1, 0), (0, -1))


@lru_cache(maxsize=1024)
def moves(cell, size):
    r, c = divmod(cell, size)
    return {k: ((r + dr) % size) * size + (c + dc) % size
            for k, (dr, dc) in zip(NAMES, OFFSETS)}


def dist(a, b, n):
    dy, dx = abs(a // n - b // n), abs(a % n - b % n)
    return min(dy, n-dy) + min(dx, n-dx)


def home_dist(cell, bases, n):
    return min((dist(cell, home, n) for home in bases), default=inf)


def _previous(past, t, player):
    obj = past.get(t - 1, {}) if past else {}
    roster = obj.get("players", ())
    return roster[player][2] if roster and obj.get("step") == t - 1 else {}


def _heading(uid, cell, earlier, size):
    previous = earlier.get(uid)
    if previous:
        return next((act for act, dest in moves(previous[0], size).items() if dest == cell), None)
    return None


def orders(obs, past, cfg, variant):
    """Cheap opponent/continuation policies. No identity, seed or opposing commands."""
    assert variant in (0, 1, 2)
    n, seat, t = cfg["size"], obs["player"], obs["step"]
    me, rival = obs["players"][seat], obs["players"][1-seat]
    homes = tuple(me[1].values())
    enemy_homes = tuple(rival[1].values())
    earlier = _previous(past, t, seat)
    directions = NAMES[1:] if variant != 1 else NAMES[1:][::-1]
    result = {}
    for uid, (cell, cargo, kind) in me[2].items():
        dest = moves(cell, n)
        heading = _heading(uid, cell, earlier, n)
        sequence = list(directions)
        if variant == 1 and heading in sequence:
            sequence.remove(heading)
            sequence.insert(0, heading)
        action = None
        if variant == 2 and kind == "COLLECTOR":
            targets = []
            for enemy, (ec, load, role) in rival[2].items():
                if role != "COLLECTOR" or load < 100 or cargo >= load or cell in homes:
                    continue
                d = home_dist(ec, enemy_homes, n)
                if d not in (1, 2):
                    continue
                for cmd in (*sequence, "NONE"):
                    x = dest[cmd]
                    if (x not in homes+enemy_homes and dist(ec,x,n) == 1
                            and home_dist(x,enemy_homes,n) == d-1):
                        targets.append((-load, enemy, (*sequence, "NONE").index(cmd), cmd))
            if targets:
                action = min(targets)[-1]
        if action is None and kind == "COLLECTOR" and cargo > 0:
            distance = home_dist(cell, homes, n)
            slack = cfg["episodeSteps"] - 1 - t - distance
            if variant == 2 and slack >= 1 and distance and obs["carbon"][cell] >= 20:
                action = "NONE"
            elif distance == 0:
                action = "NONE"
            else:
                action = min(sequence, key=lambda x: home_dist(dest[x],homes,n))
        if action is None:
            action = heading if heading in directions else sequence[0]
        if action != "NONE":
            result[uid] = action
    return result


def _operator_grid(obs, own, cfg, horizon):
    seat, n = obs["player"], cfg["size"]
    me, them = obs["players"][seat], obs["players"][1-seat]
    homes, enemy_homes = tuple(me[1].values()), tuple(them[1].values())
    home_set = set(homes + enemy_homes)
    returners, blockers = [], []
    for uid, (cell, cargo, role) in me[2].items():
        if role != "COLLECTOR":
            continue
        old = own.get(uid) or "NONE"
        dest = moves(cell, n)
        hd = home_dist(cell, homes, n)
        if cargo >= 1 and 0 < hd <= horizon:
            choices = [cmd for cmd,x in dest.items()
                       if home_dist(x,homes,n) < hd or (cmd == "NONE" and hd < horizon)]
            returners.append((-cargo, uid, [old] + [x for x in choices if x != old]))
        if cargo > 35:
            continue
        for eid, (ec,load,kind) in them[2].items():
            d = home_dist(ec, enemy_homes, n)
            if kind != "COLLECTOR" or load < 100 or d not in (1,2):
                continue
            # Landing on the rival's bank pays the rival, even when we win the collision.
            choices = [cmd for cmd,x in dest.items()
                       if x not in home_set and dist(ec,x,n) + home_dist(x,enemy_homes,n) == d
                       and dist(ec,x,n) == 1]
            extra = [cmd for cmd in choices if cmd != old]
            if extra:
                blockers.append((-load,cargo,uid,eid,[old]+extra))
    returners.sort()
    selected = returners[:2]
    used = {u for _,u,_ in selected}
    blocker = next((b for b in sorted(blockers) if b[2] not in used), None)
    chosen = {u:cmds for _,u,cmds in selected}
    if blocker:
        chosen[blocker[2]] = blocker[-1]
    elif len(returners) >= 3:
        _,uid,options = returners[2]
        chosen[uid] = options
    return chosen, {
        "homebound_eligible": len(returners), "homebound_workers": [u for _,u,_ in returners],
        "blocker_eligible": len({b[2] for b in blockers}),
        "blocker": blocker[2] if blocker else None,
        "enemy_target": blocker[3] if blocker else None, "operators": list(chosen),
    }


def _traffic(obs, commands, cfg):
    seat,n = obs["player"], cfg["size"]
    _,bases,workers,_ = obs["players"][seat]
    arrivals = {}
    for uid,(pos,cargo,kind) in workers.items():
        cmd = commands.get(uid) or "NONE"
        cell = moves(pos,n)[cmd]
        arrivals.setdefault(cell,[]).append((uid, float(cargo), pos != cell))
    for uid, cell in bases.items():
        if commands.get(uid) in ("RECCOLLECTOR","RECPLANTER"):
            arrivals.setdefault(cell,[]).append((uid,0.,False))
    conflicts, banks = set(), []
    for cell,wave in arrivals.items():
        if len(wave)<2:
            continue
        if cell in bases.values():
            banks.append({"cell":cell,"workers":[w[0] for w in wave],"incoming_cargo":sum(w[1] for w in wave),
                          "inbound_loaded":sum(w[1]>0 and w[2] for w in wave)})
        else:
            conflicts.update((cell,*sorted((a[0],b[0]))) for a,b in combinations(wave,2))
    return conflicts,banks


def _public_sim(obs, cfg):
    # Rebuild a seat's visibility so even an accidental unmasked verifier input cannot disclose hidden cells.
    seat,n = obs["player"],cfg["size"]
    visible=set()
    for cell,_,kind in obs["players"][seat][2].values():
        if kind=="COLLECTOR":
            visible.add(cell)
        else:
            visible.update(x for cmd,x in moves(cell,n).items() if cmd!="NONE")
    allowed={key:cfg[key] for key in DEFAULT_CONFIG if key in cfg and key!="randomSeed"}
    sim=FastCarbon(allowed)
    sim.load_observation({"carbon":[v if i in visible else 0. for i,v in enumerate(obs["carbon"])],
                          "step":obs["step"],"player":seat,"players":obs["players"],"trees":obs["trees"]})
    return sim


def _extinct(sim):
    threshold=min(sim.config["recCollectorCost"],sim.config["recPlanterCost"])
    return any(not sim.player_workers[i] and not sim.player_trees[i] and
               (not sim.bases[i] or sim.cash[i]<threshold) for i in (0,1))


def choose(obs, past, own, cfg):
    """Return complete candidate commands and a decision audit. Time is diagnostic only."""
    started=process_time()
    base=dict(own)
    horizon = cfg["episodeSteps"] - 1 - obs["step"]
    audit={"horizon":horizon,"edits":{},"per_scenario_gain":[],"candidate_count":0,
           "rejected_friendly_conflict":0,"evaluated":0,"new_branch_extinct":0}
    def finish(action, reason):
        audit["reason"]=reason
        audit["cpu_ms"]=(process_time()-started)*1000
        return action,audit
    if horizon not in (1,2):
        return finish(base,"outside_window")
    candidates, metadata = _operator_grid(obs,base,cfg,horizon)
    audit.update(metadata)
    base_conflicts, audit["parent_shared_bank"] = _traffic(obs,base,cfg)
    audit["selected_shared_bank"]=audit["parent_shared_bank"]
    if not any(len(options)>1 for options in candidates.values()):
        return finish(base,"no_alternative")
    seat,t = obs["player"],obs["step"]
    sim=_public_sim(obs,cfg)
    initial=sim.get_state()
    other=sim.obs(1-seat)
    first_enemy=[orders(other,past,cfg,k) for k in range(3)]
    # Only the pre-action public rosters can influence movement heading; endpoint forecasts see t's public rosters.
    previous={t:{"step":t,"players":obs["players"]}}
    def project(action):
        scores=[]
        for k,enemy in enumerate(first_enemy):
            sim.set_state(initial)
            joint=[None,None]; joint[seat]=action; joint[1-seat]=enemy
            sim.step(*joint)
            if _extinct(sim):
                return None
            if horizon==2:
                joint=[orders(sim.obs(s),previous,cfg,k) for s in (0,1)]
                sim.step(*joint)
                if _extinct(sim):
                    return None
            scores.append(sim.cash[seat]-sim.cash[1-seat])
        return scores
    baseline=project(base)
    audit["parent_projected_margins"]=baseline
    if baseline is None:
        return finish(base,"projected_parent_extinction")
    selected=base
    best=None
    keys=list(candidates)
    for commands in product(*(candidates[x] for x in keys)):
        edits={uid:cmd for uid,cmd in zip(keys,commands) if cmd!=(base.get(uid) or "NONE")}
        if not edits:
            continue
        audit["candidate_count"]+=1
        trial=dict(base)
        for uid,cmd in edits.items():
            if cmd=="NONE":
                trial.pop(uid,None)
            else:
                trial[uid]=cmd
        conflicts,banks=_traffic(obs,trial,cfg)
        if not conflicts.issubset(base_conflicts):
            audit["rejected_friendly_conflict"]+=1
            continue
        audit["evaluated"]+=1
        scores=project(trial)
        if scores is None:
            audit["new_branch_extinct"]+=1
            continue
        gain=[round(x-y,6) for x,y in zip(scores,baseline)]
        avg=sum(gain)/len(gain)
        key=(avg,-len(edits))
        if min(gain)>.01 and avg>=5 and (best is None or key>best):
            selected=trial;best=key
            audit.update(edits=edits,per_scenario_gain=gain,selected_shared_bank=banks)
    return finish(selected,"selected" if best is not None else "below_gain_gate")

"""Five-transition paired action values on frozen Top Five training and development replays."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

import cnet2_feat as F
from fastsim import FastCarbon

ELITE_SHA = "566bd7437a997078a751de6a7f0c9d0f8af1f41a901d6c0ea66b9f6192b16a9b"
HORIZON = 5
ROOT = Path(__file__).resolve().parents[1]


def sha(value):
    return hashlib.sha256(value).hexdigest()


def selected(games):
    groups = defaultdict(list)
    for g in games:
        groups[g["split"], tuple(sorted(g["models"]))].append(g)
    if len(groups) != 20:
        raise ValueError("missing Top Five pair/split")
    return [g for (split, _), group in sorted(groups.items())
            for g in sorted(group, key=lambda row: sha(("courier-v1|" + row["id"]).encode()))[:3 if split == "train" else 1]]


def roll(sim, root, trail, seat, focal, first):
    sim.set_state(root)
    transfer = [0., 0.]
    stale = 0
    extinct = [False, False]
    for j, commands in enumerate(trail):
        both = [dict(c) for c in commands]
        if j == 0:
            both[seat].pop(focal, None)
            if first:
                both[seat][focal] = F.MOVES[first]
        for p, cmd in enumerate(both):
            legal_ids = set(sim.player_workers[p]) | {i for i, _ in sim.bases[p]}
            for key in cmd.keys() - legal_ids:
                del cmd[key]
                stale += 1
        sim.step(*both)
        for _, victim, carbon, paid_to, _ in sim.last_collisions:
            if paid_to == 1 - victim:
                transfer[paid_to] += carbon
        extinct = [not sim.player_workers[p] and not sim.player_trees[p] and
                   (not sim.bases[p] or sim.cash[p] < min(sim.config["recCollectorCost"], sim.config["recPlanterCost"]))
                   for p in (0, 1)]
        if any(extinct):
            break
    cargo = [sum(sim.workers[w].carbon for w in sim.player_workers[p]) for p in (0, 1)]
    # Actual settled cash, plus a partial salvage value for cargo that was retained.
    # No geometric reward for moving toward a base. Both seats' transfers remain auditable.
    other = 1-seat
    q = sim.cash[seat] - sim.cash[other] + .30 * (cargo[seat] - cargo[other])
    return dict(q=round(q,5), cash=[round(float(x),4) for x in sim.cash],
                cargo=[round(float(x),4) for x in cargo], transfers=[round(float(x),4) for x in transfer],
                ignored_commands=stale, extinct=extinct)


def game_data(g, elite, replay_dir, archive):
    path = replay_dir / (g["id"]+".json.gz")
    compressed = path.read_bytes()
    env = json.loads(gzip.decompress(compressed))
    if sha(compressed) != archive[g["id"]]["sha256"]:
        canonical = json.dumps(env,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()
        if sha(canonical) != g["canonical_sha256"]:
            raise ValueError("replay content mismatch: " + g["id"])
    frames = env["steps"]
    tmax = len(frames) - 1
    if env["configuration"]["randomSeed"] != g["seed"] or g["rows"] != 2*tmax:
        raise ValueError("game identity or length mismatch")
    shard = elite.parent / g["shard"]
    if sha(shard.read_bytes()) != g["sha256"]:
        raise ValueError("feature shard hash mismatch")
    with np.load(shard, allow_pickle=False) as data:
        position, legal, actions, step = (data[k] for k in ("tok_pos", "legal", "y", "step"))
    wanted = set(range(39 + int(g["id"][:2], 16)%7, min(246, tmax-HORIZON), 7))
    rows, checks = [], defaultdict(int)
    # Only read historical observations up through each root for entity-ID reconstruction.
    # Full carbon and the other seat's next actions are exclusively used for labels.
    wc, tc = [0,0], [0,0]
    sim = FastCarbon(env["configuration"])
    for t in range(max(wanted)+1):
        obs = frames[t][0]["observation"]
        for player in obs["players"]:
            for table, counts in ((player[2], wc), (player[3], tc)):
                for ident in table:
                    _, creator, _, suffix = ident.split("-")
                    counts[int(creator)] = max(counts[int(creator)], int(suffix)+1)
        if t not in wanted:
            continue
        sim.load_observation(dict(obs, wid_counter=wc, tid_counter=tc))
        state = sim.get_state()
        trail = [[frames[k][p]["action"] or {} for p in (0,1)] for k in range(t+1,t+HORIZON+1)]
        end = frames[t+HORIZON][0]["observation"]["players"]
        expected_cash = np.array([pl[0] for pl in end])
        expected_cargo = np.array([sum(w[1] for w in pl[2].values()) for pl in end])
        for p, own in enumerate(obs["players"]):
            row = p*tmax+t
            if step[row] != t:
                raise ValueError("shard time mismatch")
            slots = {int(F.CANON[cell]) if p else int(cell): s for s,cell in enumerate(position[row,:10]) if actions[row,s]>=0}
            eligible=[]
            for wid, (cell, cargo, kind) in own[2].items():
                if kind != "COLLECTOR" or cargo < 25:
                    continue
                slot = slots[cell]
                if legal[row,slot].sum() >= 2:
                    eligible.append((wid,slot,cargo))
            if not eligible:
                continue
            # Outcome-independent unit selection, including pickups and contested routes far from base.
            wid,slot,load = min(eligible, key=lambda u: sha(f"{g['id']}/{t}/{p}/{u[0]}".encode()))
            labels = {}
            for physical_move in range(5):
                m = int(F.CANON_MOVE[physical_move]) if p else physical_move
                if legal[row,slot,m]:
                    labels[m] = roll(sim, state, trail, p, wid, physical_move)
            expert = int(actions[row,slot])
            observed_action = F.MOVES.index(trail[0][p].get(wid,"NONE"))
            if expert != (int(F.CANON_MOVE[observed_action]) if p else observed_action):
                raise ValueError("shard action mismatch")
            checked = labels[expert]
            checks["eligible"] += 1
            if (checked["ignored_commands"] or np.max(np.abs(checked["cash"]-expected_cash)) > .05
                    or np.max(np.abs(checked["cargo"]-expected_cargo)) > .05
                    or any(any(o["extinct"]) for o in labels.values())):
                checks["rejected_control_or_terminal"] += 1
                continue
            rows.append(dict(row=row,step=t,seat=p,unit=wid,slot=slot,load=round(float(load),4),expert=expert,
                             choices=labels))
    return dict(id=g["id"],seed=g["seed"],split=g["split"],models=g["models"],shard_sha256=g["sha256"],
                replay_sha256=sha(compressed),checks=dict(checks),rows=rows)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--elite",type=Path,required=True)
    cli.add_argument("--replays",type=Path,required=True)
    cli.add_argument("--out",type=Path,required=True)
    args=cli.parse_args()
    if sha(args.elite.read_bytes()) != ELITE_SHA:
        raise ValueError("wrong split")
    inventory=json.loads((ROOT/"reports/master_family/tactical_rules/top5_inventory.json").read_bytes())["replays"]
    parts=[]
    for g in selected(json.loads(args.elite.read_bytes())["games"]):
        result=game_data(g,args.elite,args.replays,inventory)
        parts.append(result)
        print(g["split"],g["id"],len(result["rows"]),result["checks"],flush=True)
    out=dict(horizon=HORIZON, parent_corpus_sha256=ELITE_SHA, builder_sha256=sha(Path(__file__).read_bytes()),
             utility="Own minus opponent bank cash plus 0.30 times net retained cargo, in carbon units, five transitions. Fixed succeeding actions for surviving UIDs, stale commands counted. Both seats' cross-player cargo transfers in diagnostics.",
             eligibility="Both seats; t~40..245 every seven, one uniformly hash-selected loaded collector (>=25) with >=2 prefix-legal actions per seat/root. Exact logged branch cash and cargo to 0.05 required. Exclude roots with a terminal branch.",
             games=parts)
    with args.out.open("xb") as h:
        h.write(gzip.compress(json.dumps(out,separators=(",",":"),allow_nan=False).encode(),mtime=0))


if __name__ == "__main__":
    main()

"""Audit autoregressive commands and replay the first late collector edit per development game."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from pydantic_core import from_json

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parent
BASE = ACTOR.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT / "code"))
import cnet2_feat as feat
import cnet2_np
import cnet2_pack
from fastsim import FastCarbon
from lots import measure

NAMES = ("parent", "control", "candidate")
CACHE = Path("/tmp/sapling-late-adv-v1")
REPLAYS = Path("/workspace/sapling-all-top5-20261119/replays")
GROUPS = ("all", "late_collector", "early", "planter", "base", "protected")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, obj):
    raw = (json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if path.suffix == ".gz":
        raw = gzip.compress(raw, mtime=0)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)


def load_actor(path):
    if cnet2_pack.QMODE != "int8":
        raise ValueError("requires production quantization")
    with np.load(path, allow_pickle=False) as archive:
        _, packed, deq = cnet2_pack.quantize(archive)
        deq["meta"] = json.loads(str(archive["meta"]))
    return cnet2_np.Policy(deq), {"weights_sha256": digest(path.read_bytes()), "packed_sha256": digest(packed)}


def views(frame, seat):
    shared = frame[0]["observation"]
    return {**{k: shared[k] for k in ("step","trees","players")}, "player":seat, "carbon":frame[seat]["observation"]["carbon"]}


def discover(doc, game, polys):
    card = game["card"]
    seat = card["seat"]
    path = CACHE / (card["id"] + ".npz")
    if digest(path.read_bytes()) != game["shard_sha256"]:
        raise ValueError("source-shard fingerprint differs")
    with np.load(path, allow_pickle=False) as shard:
        slots = {int(t): i for i,t in enumerate(shard["step"])}
        hist = feat.Feat(seat)
        selected = defaultdict(set)
        for row in game["sample"]:
            selected[row["t"]].add(row["slot"])
        counter = defaultdict(Counter)
        roots, first = [], None
        frames = doc["steps"]
        for t in range(max(slots) + 1):
            x = hist.observe(views(frames[t],seat))
            if t not in slots:
                continue
            row = slots[t]
            for key in ("tok_pos","tok_kind","tok_mask","hist_cell","hist_move"):
                if not np.array_equal(x[key],shard[key][row]):
                    raise ValueError(f"public history differs for {key} at {t}")
            outputs = [policy.act(x, hist.aux, teacher=2) for policy in polys]
            acts = [hist.to_actions(z[0]) for z in outputs]
            for (labels,logits,_), orders in zip(outputs, acts):
                if not np.array_equal(hist.labels(orders), labels):
                    raise ValueError("decoder seat/slot round trip")
                if not np.isfinite(logits).all():
                    raise ValueError("nonfinite policy logits")
            occ = x["tok_mask"][:feat.OWN]
            kinds = x["tok_kind"][:feat.OWN]
            role = {"all":occ, "early":occ & (t<225),
                    "late_collector":occ & (kinds==0) & (t>=225),
                    "planter":occ & (kinds==1), "base":occ & (np.arange(feat.OWN)>=10)}
            role["protected"] = role["early"] | role["planter"] | role["base"]
            byname = {}
            for i,j in ((0,2),(0,1),(1,2)):
                key = NAMES[i]+"_vs_"+NAMES[j]
                dif = outputs[i][0] != outputs[j][0]
                both = np.logical_and(*[(outputs[k][1] > -5e8).sum(1)>1 for k in (i,j)])
                byname[key] = [int(s) for s in np.flatnonzero(occ & dif)]
                for group,mask in role.items():
                    c = counter[(key, group)]
                    c["slots"] += int(mask.sum())
                    c["changed"] += int((mask & dif).sum())
                    c["both_nonforced"] += int((mask & both).sum())
                    c["changed_both_nonforced"] += int((mask & both & dif).sum())
                    c["roots"] += int(mask.any())
                    c["changed_roots"] += int((mask & dif).any())
            roots.append({"t":t,"occupied_slots":np.flatnonzero(occ).tolist(),"commands":dict(zip(NAMES,acts)),
                          "changed_slots":byname})
            changed = sorted(selected[t] & set(byname["control_vs_candidate"]))
            if first is None and changed:
                first = {"t":t,"commands":dict(zip(NAMES, acts)),"changed_collectors":[hist.own_uid[s] for s in changed]}
    return roots, counter, first


def native_prefix(doc, t):
    frames = doc["steps"]
    sim = FastCarbon(doc["configuration"]).load_observation(frames[0][0]["observation"])
    for frame in frames[1:t+1]:
        sim.step(*[a["action"] or {} for a in frame])
    original = frames[t][0]["observation"]
    actual = sim.full_observation()
    if actual["players"] != original["players"] or actual["trees"] != original["trees"] or actual["carbon"] != original["full_carbon"]:
        raise ValueError("complete replay prefix failed to reconstruct the root")
    return sim.get_state()


def extinct(player, cfg):
    cash, bases, workers, trees = player
    return not workers and not trees and (not bases or cash < min(cfg["recCollectorCost"],cfg["recPlanterCost"]))


def fixed_continuation(doc, t, seat, root_command, focus, snapshot, original=False):
    cfg = doc["configuration"]
    sim = FastCarbon(cfg)
    sim.set_state(snapshot)
    frames = [[{"observation":sim.full_observation(),"action":{}},{"action":{}}]]
    ignored = Counter()
    for i, recorded in enumerate(doc["steps"][t+1:]):
        opts = [dict(r["action"] or {}) for r in recorded]
        if i==0:
            opts[seat] = dict(root_command)
        pre = sim.full_observation()["players"]
        for s in (0,1):
            live = set(pre[s][1]) | set(pre[s][2])
            ignored[str(s)] += len(set(opts[s])-live)
            opts[s] = {u:a for u,a in opts[s].items() if u in live}
        sim.step(*opts)
        obs = sim.full_observation()
        frames.append([{"observation":obs, "action":opts[0]},{"action":opts[1]}])
        if original and (obs["players"] != recorded[0]["observation"]["players"]
                         or obs["carbon"] != recorded[0]["observation"]["full_carbon"]):
            raise ValueError("registered original log deviated on replay")
        dead = [extinct(player,cfg) for player in obs["players"]]
        if any(dead):
            break
    n = len(frames)-1
    terminated = any(dead) or t+n == int(cfg["episodeSteps"])-1
    scores = [t+n-int(cfg["episodeSteps"])-1 if dead[k] else pre[k][0] if any(dead) else obs["players"][k][0] for k in (0,1)]
    if original and (not terminated or not np.allclose(scores,doc["rewards"],atol=.015,rtol=0)):
        raise ValueError("original terminal score did not reproduce")
    lot, totals = measure({"steps":frames, "configuration":cfg},seat,lo=0)
    def settlement(workers):
        entries = [v for (tick,u),v in lot.items() if tick<4 and u in workers]
        return {"fresh_gross":sum(v["gross"] for v in entries),"fresh_direct":sum(v["direct"] for v in entries),
                "starting_cargo_direct":sum(v["existing_direct"] for (tick,u),v in lot.items() if tick==0 and u in workers)}
    own = frames[0][0]["observation"]["players"][seat][2]
    return {"turns":n,"reached_terminal":terminated,"cash_score_if_terminal":scores if terminated else None,
            "official_rule_margin_if_terminal":scores[seat]-scores[1-seat] if terminated else None,
            "ignored_dead_uid_commands":dict(ignored),"focus":settlement(set(focus)),
            "root_collectors":settlement({u for u,w in own.items() if w[2]=="COLLECTOR"}),
            "team_own_deposits_to_replay_end":totals["official_own_worker_deposit"]}


def main():
    protocol = json.loads((HERE/"protocol.json").read_bytes())
    paths = [BASE.parents[1]/"data/weights/sapling_1200k/master6_actor_v1.npz",HERE/"control.npz",HERE/"candidate.npz"]
    loaded = [load_actor(path) for path in paths]
    actor_ids = {name:identity for name,(_,identity) in zip(NAMES,loaded)}
    if actor_ids["parent"]["weights_sha256"] != protocol["parent"] or len({v["packed_sha256"] for v in actor_ids.values()})<3:
        raise ValueError("serving actors not distinct or wrong parent")
    games = [g for g in from_json(gzip.decompress((HERE/"settlements.json.gz").read_bytes()))["games"] if g["card"]["split"]=="held"]
    if len(games)!=37 or len({g["card"]["terrain_sha256"] for g in games})!=len(games):
        raise ValueError("unexpected tested map panel")
    counts, maps = defaultdict(Counter), []
    for i,game in enumerate(games):
        card = game["card"]; seat = card["seat"]
        packed = (REPLAYS/(card["id"]+".json.gz")).read_bytes()
        if digest(packed)!=card["sha256"]:
            raise ValueError("source replay differs")
        doc = from_json(gzip.decompress(packed))
        roots, local, first = discover(doc,game,[p for p,_ in loaded])
        for k,v in local.items(): counts[k].update(v)
        branch = None
        if first:
            t = first["t"]; snap = native_prefix(doc,t)
            arms = {"recorded":doc["steps"][t+1][seat]["action"] or {}, **{k:first["commands"][k] for k in ("control","candidate")}}
            branch = {"root":first, "arms":{k:fixed_continuation(doc,t,seat,v,first["changed_collectors"],snap,k=="recorded") for k,v in arms.items()}}
        maps.append({"id":card["id"],"terrain":card["terrain_sha256"],"wdl":card["wdl"],"opponent":card["opponent"],"seat":seat,
                     "source_sha256":card["sha256"],"counts":{"/".join(k):v for k,v in local.items()},
                     "roots":roots,"first_changed_late_collector":branch})
        print(i+1,len(roots),None if first is None else first["t"],flush=True)
    out = HERE/"serving_evidence.json.gz"
    save(out,{"actors":actor_ids,"groups":GROUPS,"maps":maps})
    bygroup = {"/".join(k):{**v,"changed_rate":v["changed"]/v["slots"] if v["slots"] else None}
               for k,v in counts.items()}
    paired = []
    for game in maps:
        b=game["first_changed_late_collector"]
        if b:
            x,y = (b["arms"][k] for k in ("candidate","control"))
            paired.append({"id":game["id"],"t":b["root"]["t"],"new_mining_direct_delta_focus":x["focus"]["fresh_direct"]-y["focus"]["fresh_direct"],
                           "starting_cargo_direct_delta_focus":x["focus"]["starting_cargo_direct"]-y["focus"]["starting_cargo_direct"],
                           "all_original_collectors_new_mining_direct_delta":x["root_collectors"]["fresh_direct"]-y["root_collectors"]["fresh_direct"],
                           "team_own_deposits_delta":x["team_own_deposits_to_replay_end"]-y["team_own_deposits_to_replay_end"],
                           "terminal_margin_delta":(x["official_rule_margin_if_terminal"]-y["official_rule_margin_if_terminal"]
                                                    if x["reached_terminal"] and y["reached_terminal"] else None)})
    phases={k:bygroup["control_vs_candidate/"+k]["changed_rate"] for k in GROUPS}
    save(HERE/"serving.json",{"maps":len(games),"roots":sum(len(m["roots"]) for m in maps),"actors":actor_ids,
         "protocol_sha256":digest((HERE/"protocol.json").read_bytes()),"evidence_sha256":digest(out.read_bytes()),
         "method":"Each network decodes from its own prefix. All models replay every public observation; no enforced expert command or policy guard. Group counts are occupied physical command slots, plus subset nonforced in both policies.",
         "branch_selection":"First chronological registered nonoverlapping late root per map with a changed labeled collector. Opponent and all continuation orders fixed from the source, missing worker command removed. These conditional alternative states are simulated, without a responding opponent.",
         "counts":bygroup, "selected_maps":len(paired),"map_differences":paired,
         "gate_late_collector":phases["late_collector"]>=.005,"gate_protected_pooled":phases["protected"]<=.02,
         "gate_each_early_base_planter":all(phases[k]<=.02 for k in ("early","base","planter"))})


if __name__=="__main__":
    main()

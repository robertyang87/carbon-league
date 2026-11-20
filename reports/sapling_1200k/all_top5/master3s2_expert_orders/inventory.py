"""Build a read-only card/command inventory and stream public teacher-forced slot labels."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
REPO = TOP.parents[2]
ACTOR = TOP.parent / "master6_actor_dev"
REV = "7574f70c0ae34f9314e9aa5da0e967af3b437c7d"
EXAM = "claude-AlphaCarbon-Master3s2"
EXAM_ID = "d679ed45ab595e8b7bb1ea51c3451154"
EXAM_SOURCE = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
sys.path.insert(0, str(TOP))
sys.path.insert(0, str(REPO / "code"))
import context
import cnet2_feat as F


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), "-c", "protocol.allow=never", *args],
                                   env={**os.environ, "GIT_NO_LAZY_FETCH":"1", "GIT_TERMINAL_PROMPT":"0"})


def rel(path):
    return str(path.relative_to(REPO))


def blob(path):
    return git("cat-file", "blob", f"{REV}:{path}")


def read(path):
    return json.loads(blob(rel(path)))


def paths(folder):
    return git("ls-tree", "-r", "--name-only", REV, "--", folder).decode().splitlines()


@lru_cache(maxsize=1)
def archive(path):
    return blob(path)


def raw_bytes(location):
    if "repo_path" in location:
        return blob(location["repo_path"])
    with tarfile.open(fileobj=io.BytesIO(archive(location["archive"])), mode="r:*") as tar:
        return tar.extractfile(location["member"]).read()


def source_cards():
    for module in (context, F):
        local = Path(module.__file__).resolve()
        require(local.read_bytes() == blob(rel(local)), "public or prefix implementation differs from frozen code")
    cards, sources, geography = {}, {}, {}
    top_names = set()
    for path in paths(rel(TOP)):
        p = REPO / path
        if p.name != "manifest.json" or not (p.parent == TOP or p.parent.parent == TOP and p.parent.name.startswith("appendix_")):
            continue
        body = blob(path)
        batch = json.loads(body)
        map_file = rel(p.with_name("validation_maps.jsonl.gz"))
        map_data = blob(map_file)
        maps = {v["id"]:v for v in map(json.loads, gzip.decompress(map_data).splitlines())}
        names = {v["name"]:v["id"] for v in batch["top5"]}
        if top_names:
            require(set(names) == top_names, "registered top five changed")
        top_names = set(names)
        sources[path] = {"sha256":sha(body), "geography":map_file, "geography_sha256":sha(map_data), "games":len(batch["games"])}
        for c in batch["games"]:
            k = c["id"]
            require(k not in geography and maps[k]["seed"] == c["seed"], "duplicate game or mismatched seed")
            geography[k] = maps[k]["initial_terrain_u8_sha256"]
            if EXAM not in c["seat_names"]:
                continue
            seat = 1 - c["seat_names"].index(EXAM)
            name = c["seat_names"][seat]
            require(c["seat_strategy_ids"][seat] == names[name] and c["clean_terminal"] and c["frame_verified"], "source seat or clean terminal")
            cards[k] = dict(id=k, seed=c["seed"], terrain=geography[k], expert=name, expert_strategy_id=names[name],
                            examiner_registration=EXAM_ID, seat=seat, score_by_seat=c["score_by_seat"], source_batch=path,
                            archive=c["archive_location"], replay_sha256=c["sha256"], uuid=c["environment_uuid"],
                            steps=c["transitions"], source_method=c["source"]["method"], score_indexes=[])
    require(len(geography) == len(set(geography.values())) == 1647 and len(cards) == 30, "frozen corpus changed")
    require(EXAM_ID.encode() in blob("sources/master3s2/README.md") and sha(blob("sources/master3s2/master3s2.py")) == EXAM_SOURCE,
            "published reference identity changed")
    found = set()
    for index in paths("data"):
        if not index.endswith("_matches.json"):
            continue
        data = json.loads(blob(index))
        if not isinstance(data, list):
            continue
        for c in data:
            if not isinstance(c, dict) or c.get("state") != "done":
                continue
            ns = (c.get("name_a"), c.get("name_b"))
            if EXAM not in ns or not top_names.intersection(ns):
                continue
            k = f"{c['id']}-{c['game']}"
            require(k in cards, f"missing scored top-five game {k}")
            r = cards[k]
            order = c["order"]
            physical = [c[f"name_{s}"] for s in order]
            require(physical[r["seat"]] == r["expert"] and physical[1-r["seat"]] == EXAM
                    and [c["scores"][s] for s in order] == r["score_by_seat"], "score, seat or registration mismatch")
            if index not in r["score_indexes"]:
                r["score_indexes"].append(index)
            found.add(k)
    require(found == cards.keys(), "score index coverage changed")
    return cards, sources


def own_orders(card, raw):
    game = context.Game(raw)
    seat = card["seat"]
    featurizer = F.Feat(seat)
    own_prefix = hashlib.sha256()
    for t in range(game.turns):
        obs = game._obs(t, seat)
        require(obs["step"] == t and obs["player"] == seat and set(obs) == context.OBS_FIELDS, "unexpected public input")
        x = featurizer.observe(obs)
        full = raw["steps"][t+1][seat]["action"] or {}
        order = {k:v for k,v in full.items() if v not in (None, "NONE")}
        labels = featurizer.labels(order)
        representable = featurizer.to_actions(labels) == order
        prefix = F.Prefix(x, featurizer.aux)
        prior, slots = [], []
        for s, uid in enumerate(featurizer.own_uid):
            if uid is None:
                continue
            _, _, legal = prefix.feats(s)
            action = int(labels[s])
            role = obs["players"][seat][2][uid][2] if s < 10 else "BASE"
            slots.append(dict(uid=uid,role=role,slot=s,observed=order.get(uid,"NONE"),canonical_action=action,
                              legal=[bool(b) for b in legal],prefix=prior.copy(),allowed=bool(legal[action])))
            prefix.advance(s,action)
            prior.append([s,action])
        yield dict(t=t,public=obs,prior_own_orders_sha256=own_prefix.hexdigest(),
                   label=dict(expert_order=order,slots=slots,round_trip=representable,
                              all_slots_legal=all(s["allowed"] for s in slots)))
        own_prefix.update(canonical(order)+b"\n")


def ledger(card):
    packed = raw_bytes(card["archive"])
    require(sha(packed) == card["replay_sha256"], "archive member hash mismatch")
    raw = json.loads(gzip.decompress(packed))
    frames = raw["steps"]
    require(raw["id"] == card["uuid"] and raw["configuration"]["randomSeed"] == card["seed"]
            and sha(bytes(frames[0][0]["observation"]["full_carbon"])) == card["terrain"]
            and raw["rewards"] == [s["reward"] for s in frames[-1]] == card["score_by_seat"]
            and raw["statuses"] == ["DONE","DONE"] and len(frames)-1 == card["steps"], "official identity or reward changed")
    joint, expert, pubs = (hashlib.sha256() for _ in range(3))
    counters = Counter()
    misses = Counter()
    for entry in own_orders(card, raw):
        t, label = entry["t"], entry["label"]
        pair = [s["action"] or {} for s in frames[t+1]]
        require(all(s["status"] == "ACTIVE" for s in frames[t]), "inactive turn mislabeled as an ordinary action")
        joint.update(canonical(pair)+b"\n")
        expert.update(canonical(label["expert_order"])+b"\n")
        pubs.update(canonical(entry["public"])+b"\n")
        counters["turns"] += 1
        counters["worker_slots"] += sum(s["role"] != "BASE" for s in label["slots"])
        counters["base_slots"] += sum(s["role"] == "BASE" for s in label["slots"])
        if not label["round_trip"]:
            misses["unrepresentable_turns"] += 1
        if not label["all_slots_legal"]:
            misses["decoder_mask_conflict_turns"] += 1
        if label["round_trip"] and label["all_slots_legal"]:
            counters["uncorrected_prefix_turns"] += 1
            counters["uncorrected_worker_slots"] += sum(s["role"] != "BASE" for s in label["slots"])
    return dict(**counters,**misses,joint_orders_sha256=joint.hexdigest(),
                observed_expert_orders_sha256=expert.hexdigest(),public_observations_sha256=pubs.hexdigest())


def registered_uses(cards):
    uses = defaultdict(list)
    names = ["data_split.json","broader_elite_backbone_dev/panel.json","strategic_destination_learning/data.json"]
    for name in names:
        d = read(ACTOR / name)
        by_map = defaultdict(list)
        for r in d.get("games",d.get("shards",[])):
            by_map[r.get("terrain_sha256",r.get("terrain"))].append(r)
        for k, card in cards.items():
            for s in sorted({r["split"] for r in by_map[card["terrain"]]}):
                uses[k].append(dict(project=rel(ACTOR/name),split=s,role="expert actor network",terrain=card["terrain"]))
    return uses


def build():
    cards, batches = source_cards()
    major = registered_uses(cards)
    from prior_use import prior_uses
    other = prior_uses(list(cards.values()))
    status_by_role = Counter()
    validate_code_sha = sha((HERE/"prior_use.py").read_bytes())
    grouped = defaultdict(lambda: Counter())
    for card in sorted(cards.values(),key=lambda r:(r["archive"].get("archive",""),r["id"])):
        k, s = card["id"], card["seat"]
        cash = card["score_by_seat"]
        margin = round(cash[s] - cash[1-s], 2)
        card.update(margin=margin,result="win" if margin>0 else "loss" if margin<0 else "draw",ledger=ledger(card),
                    three_policy_panels=major[k],additional_uses=other.get(k,[]))
        roles = {r["split"] for r in card["three_policy_panels"]+card["additional_uses"]}
        # Keep previous map isolation. Observed diagnostics without a split do not confer training rights.
        protected = roles.intersection({"held","holdout","test","val","valid","validation","calibration","dev"})
        protected.update(role for role in roles if "holdout" in role or "held" in role or "new_map" in role)
        actor_train = any(r["split"]=="train" for r in major[k])
        if protected:
            status = "protected"
        elif actor_train:
            status = "known_actor_train"
        elif "train" in roles:
            status = "other_historical_train_only"
        else:
            status = "unassigned_to_actor_training"
        card["disposition"] = status
        status_by_role[status]+=1
        grouped[card["expert"]][card["result"]]+=1
    return dict(frozen_commit=REV,batches=batches,registered_examiner=EXAM,examiner_reference_sha256=EXAM_SOURCE,
                source_games=1647,direct_games=30,distinct_maps=30,three_policy_panel_overlap=sum(bool(major[k]) for k in cards),
                disposition_counts=dict(status_by_role),by_expert={k:dict(v) for k,v in grouped.items()},
                total_ledger=dict(sum([Counter({k:v for k,v in c["ledger"].items() if type(v) is int}) for c in cards.values()],Counter())),
                code_sha256=sha(Path(__file__).read_bytes()), prior_registry_code_sha256=validate_code_sha,
                game_context_code_sha256=sha(Path(context.__file__).read_bytes()), decoder_code_sha256=sha(Path(F.__file__).read_bytes()),
                games=sorted(cards.values(),key=lambda r:r["id"]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders",metavar="GAME")
    args=parser.parse_args()
    if not args.orders:
        print(json.dumps(build(), indent=2, sort_keys=True, ensure_ascii=False,allow_nan=False))
        return
    manifest=json.loads((HERE/"manifest.json").read_bytes())
    require(manifest["frozen_commit"] == REV and sha(Path(F.__file__).read_bytes()) == manifest["decoder_code_sha256"]
            and sha(Path(context.__file__).read_bytes()) == manifest["game_context_code_sha256"], "decoder or public projection changed")
    card=next(r for r in manifest["games"] if r["id"]==args.orders)
    require(card["disposition"]=="known_actor_train","registered expert actor training maps only")
    packed=raw_bytes(card["archive"])
    require(sha(packed)==card["replay_sha256"],"replay changed")
    for row in own_orders(card,json.loads(gzip.decompress(packed))):
        label=row["label"]
        if label["round_trip"] and label["all_slots_legal"]:
            print(canonical(row).decode())


if __name__=="__main__":
    main()

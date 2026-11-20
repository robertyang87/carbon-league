"""Recover ten registered two-command terminal contrasts and replay only the acting seat's history."""
import gzip
import hashlib
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DER = HERE.parents[1]
ORIGINAL = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
V3 = "016bc047554dde98844ddf4b7e8be03f52fcbe474017a0c5732d83f97485e31d"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def file_sha(path):
    return digest(path.read_bytes())


def build(group, ticket, row, arms, report, folder, source, source_sha, source_arm, policy_sha):
    seed, seat, step = (row.get(key, ticket.get(key)) for key in ("seed", "seat", "step"))
    orders, margins, enemies, requests, refs = [], [], [], [], {}
    for arm in arms:
        entry = row["arms"][arm]
        if group == "tree":
            path = folder / f"{seed}_{arm}.json.gz"
        elif group == "block":
            path = folder / f"{seed}_{seat}_{step}_{arm}.json.gz"
        else:
            path = folder / entry["evidence"]
        sha = entry.get("sha256", entry.get("evidence_sha256"))
        require(file_sha(path) == sha, path)
        evidence = load(path)
        if group == "tree":
            tape = evidence["rows"]
            require(evidence["arm"] == arm and evidence["protocol"] == row["protocol"], path)
            require(evidence["root_enemy"] == tape[0]["actions"][1-seat], path)
            require(tape[0]["actions"][seat] == ticket[arm], path)
        elif group == "block":
            tape = evidence["rows"]
            require(evidence["choices"][arm] == tape[0]["orders"][seat], path)
            require(evidence["root_shared_original_order"] == tape[0]["orders"][1-seat], path)
        else:
            tape = evidence["tape"]
            require(evidence["root_order"] == arm and evidence["ticket"] == ticket, path)
            require(evidence["continuation_by_seat"][1-seat] == ORIGINAL, path)
        first, last = tape[0], tape[-1]
        pair = first.get("actions", first.get("orders"))
        orders.append(pair[seat])
        enemies.append(pair[1-seat])
        requests.append(first.get("public_sha256", first.get("public_sha", first.get("input_sha256"))))
        require([r.get("step", r.get("t")) for r in tape] == list(range(step, 299)), path)
        result = evidence.get("outcome", evidence)
        scores = result["scores"]
        require(scores == entry["scores"] == last.get("wallets", last.get("end_cash")), path)
        margin = round(scores[seat] - scores[1-seat], 2)
        reported = entry.get("margin", entry.get("official_margin"))
        require(abs(margin - reported) < 1e-7, path)
        if group in ("tree", "block"):
            require(entry.get("status", entry.get("statuses")) == ["DONE", "DONE"], path)
        margins.append(margin)
        refs[arm] = {"path": str(path.relative_to(DER)), "sha256": sha}
    require(enemies[0] == enemies[1] and requests[0] == requests[1] and orders[0] != orders[1], report)
    refs["result"] = {"path": str(report.relative_to(DER)), "sha256": file_sha(report)}
    return dict(group=group, seed=seed, seat=seat, step=step, terrain=ticket["terrain"],
                source_sha=source_sha, source_path=source, source_arm=source_arm,
                policy_sha=policy_sha, own_observation_sha=requests[0][seat],
                arms=list(arms), orders=orders, margins=margins, references=refs)


def cards():
    registry = load(HERE / "protocol.json")
    for path, sha in registry["source_sha256"].items():
        require(file_sha(DER / path) == sha, path)
    rows = []
    folder = DER / "tree_action_outcomes"
    tree = load(folder / "protocol.json")
    for t in tree["tickets"]:
        report = folder / f"{t['seed']}_result.json"
        row = load(report)
        require(row["protocol"] == file_sha(folder / "protocol.json"), report)
        require(row["root_sha256"] == file_sha(folder / f"{t['seed']}_root.json.gz"), report)
        rows.append(build("tree", t, row, ("leave", "stay"), report, folder,
                          t["archive_relpath"], t["archive_sha256"], "parent", tree["candidate"]))
    base = DER / "parallel_responses"
    registry2 = load(base / "protocol.json")
    tickets = {(t["seed"], t["seat"]): t for t in registry2["tickets"]}
    for group, folder, key, arms in [("width", base, "t293_responses", ("M1", "M2")),
                                      ("mean", base / "equal_work", "selected", ("single0", "mean2"))]:
        report = folder / "terminal.json"
        settled = load(report)
        require(settled.get("protocol_sha", settled.get("protocol")) == file_sha(folder / "protocol.json"), report)
        for row in settled[key]:
            t = tickets[row["seed"], row["seat"]]
            rows.append(build(group, t, row, arms, report, folder, t["source"], t["sha256"],
                              "parent", tree["candidate"] if t["source"].startswith("identity_replication_dev/") else registry2["parent"]))
    folder = DER / "v3_release/development/return_turns"
    host = folder.parent / "hosted"
    settled = load(host / "settlement.json")
    for seat, step in [(0, 290), (1, 271)]:
        seed = 621758607
        tape = next(t for t in settled["archives"] if (t["seed"], t["seat"]) == (seed, seat))
        t = next(t for t in settled["games"] if (t["seed"], t["seat"], t["arm"]) == (seed, seat, "challenger"))
        report = folder / f"{seed}_{seat}_{step}_result.json"
        row = load(report)
        require(row["root"] == [seed, seat, step] and row["selected_sha256"] == file_sha(folder / "locked_roots.json"), report)
        require(row["protocol"] == file_sha(folder / "intervention_protocol.json"), report)
        source = str((host / f"{tape['id']}.zip").relative_to(DER))
        rows.append(build("block", {**t, "step": step}, row, ("parent", "block"), report, folder,
                          source, tape["sha256"], "challenger", V3))
    require([[r[k] for k in ("seed", "seat", "step", "group")] for r in rows] == registry["keys"][-10:], "ten registered roots")
    return rows


def context(card):
    """Only own public requests and executed pre-root commands, retaining original dictionary order."""
    seat, step = card["seat"], card["step"]
    path = DER / card["source_path"]
    require(file_sha(path) == card["source_sha"], path)
    with zipfile.ZipFile(path) as z:
        def read(name):
            return z.read(card["source_arm"] + "/" + name)
        registration, receipt = read("registration.json"), json.loads(read("receipt.json"))
        reg = json.loads(registration)
        turns = read("turns.jsonl")
        require(digest(registration) == receipt["protocol_sha256"] and digest(turns) == receipt["turns_sha256"], path)
        require((reg["seed"], reg["candidate_seat"], reg["physical_map_sha256"]) ==
                (card["seed"], seat, card["terrain"]), path)
        require(reg["files"]["opponent"]["sha256"] == ORIGINAL and
                reg["files"]["candidate"]["sha256"] == card["policy_sha"], path)
        lines = iter(turns.splitlines())
        history = []
        for n, raw in enumerate(gzip.decompress(read("observations.jsonl.gz")).splitlines()):
            frame = json.loads(raw)
            obs = frame["observations"][seat]
            require(set(obs) == {"step", "player", "carbon", "players", "trees", "remainingOverageTime"}, path)
            require((frame["step"], obs["step"], obs["player"]) == (n, n, seat), path)
            check = digest((json.dumps(obs, separators=(",", ":"), allow_nan=False) + "\n").encode())
            require(check == frame["public_observation_sha256"][seat], path)
            if n == step:
                require(check == card["own_observation_sha"], (path, step, check, card["own_observation_sha"]))
                break
            action = json.loads(next(lines))
            require(action["step"] == n and action["public_observation_sha256"][seat] == check, path)
            call = action["calls"][seat]
            require(call["status"] == "ok", path)
            history.append((obs, call["response"]["action"]))
        else:
            raise ValueError("Missing registered root")
    cfg_source = DER.parent / "sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev/direct_original/games/candidate/raw_621758346_1.json.gz"
    require(file_sha(cfg_source) == "7a5967383c78324d248f4b228541d7eb30f3ac3d39bf18e74f81a011a78c4631", cfg_source)
    config = load(cfg_source)["game_configuration"]
    config.pop("randomSeed")
    config.update(agentTimeout=20, actTimeout=1, runTimeout=1500)
    return history, obs, config

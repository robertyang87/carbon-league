"""Select physical maps using stored identities and construct pre-action cash projection rows."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REV = "7acd7b8c1136904aa828708e3d288dcf097f0109"
PREFIX = "reports/master3s2_derivative/"
TOP = "reports/sapling_1200k/all_top5/"
TIMES = tuple(range(249, 295, 5))
CACHE = Path("/workspace/sapling-all-top5-20261119/replays")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def blob(path):
    return git("show", f"{REV}:{path}")


def registry():
    paths = git("ls-tree", "-r", "--name-only", REV).decode().splitlines()
    ledger = json.loads(blob(PREFIX+"ladder/exposed_ladder_maps.json"))
    used = {item["opening_carbon_bytes_sha256"] for item in ledger}
    blocked_seeds = {item[k] for item in ledger for k in ("scorecard_seed","replay_random_seed")}
    for item in json.loads(blob(PREFIX+"ladder/tree_payback/maps.json"))["tickets"]:
        used.add(item["terrain"])
    for path in paths:
        if path.startswith(PREFIX) and path.endswith("/clearance.json"):
            prior = json.loads(blob(path))
            used.update(prior["by_seed"].values())
            blocked_seeds.update(map(int,prior["by_seed"]))
    pool = {}
    for path in sorted(p for p in paths if p.startswith(TOP) and p.endswith("/validation_maps.jsonl.gz")):
        for line in gzip.decompress(blob(path)).splitlines():
            item = json.loads(line)
            terrain, seed = item["initial_terrain_u8_sha256"], item["seed"]
            if terrain in used or seed in blocked_seeds or 1887000000 <= seed < 1888000000:
                continue
            if terrain in pool:
                raise ValueError(f"Repeated opening {terrain}")
            pool[terrain] = (item,path)
    ranked = sorted(pool, key=lambda key: digest(b"ds-pro/terminal-cash-projection/v1/"+bytes.fromhex(key)))
    manifests = {}
    tickets = []
    for i,terrain in enumerate(ranked[:200]):
        row,path = pool[terrain]
        directory = path.rsplit("/",1)[0]
        if directory not in manifests:
            manifests[directory] = {x["id"]:x for x in json.loads(blob(directory+"/manifest.json"))["games"]}
        entry = manifests[directory][row["id"]]
        assert entry["frame_verified"] and row["seed"] == entry["seed"]
        tickets.append(dict(split="train" if i<160 else "held", id=row["id"], seed=row["seed"],
                            terrain=terrain, compressed_sha256=entry["sha256"], source=entry["archive_location"]))
    assert len({x["id"] for x in tickets})==len({x["terrain"] for x in tickets})==200
    return dict(revision=REV, eligible=len(pool), tickets=tickets)


def replay(ticket, cache=CACHE):
    local = cache / (ticket["id"]+".json.gz")
    if local.exists():
        raw = local.read_bytes()
    elif "repo_path" in ticket["source"]:
        raw = blob(ticket["source"]["repo_path"])
    else:
        source = ticket["source"]
        with tarfile.open(fileobj=io.BytesIO(blob(source["archive"])),mode="r:*") as tar:
            raw = tar.extractfile(source["member"]).read()
    if digest(raw) != ticket["compressed_sha256"]:
        raise ValueError("Replay hash mismatch "+ticket["id"])
    data = json.loads(gzip.decompress(raw))
    assert digest(bytes(data["steps"][0][0]["observation"]["full_carbon"])) == ticket["terrain"]
    return data


def examples(data):
    """One seat orientation per map. Future frames/rewards are label-only."""
    frames = data["steps"]
    terminal = frames[-1]
    rewards = data.get("rewards")
    if [v.get("status") for v in terminal] != ["DONE","DONE"]:
        return [], "non-DONE terminal"
    if not isinstance(rewards,list) or len(rewards)!=2 or not all(isinstance(v,(float,int)) for v in rewards):
        return [], "no numeric official reward"
    rows=[]
    for t in TIMES:
        if t >= len(frames)-1:
            continue
        obs=frames[t][0]["observation"]
        assert obs["step"]==t
        rows.append(dict(step=t, players=obs["players"], seat=0, final_margin=rewards[0]-rewards[1]))
    return rows, "ok" if rows else "ended before study window"


if __name__=="__main__":
    target=HERE/"maps.json"
    raw=(json.dumps(registry(),ensure_ascii=False,indent=2)+"\n").encode()
    if target.exists() and raw!=target.read_bytes():
        raise ValueError("Registration changed")
    target.write_bytes(raw)
    print(digest(raw))

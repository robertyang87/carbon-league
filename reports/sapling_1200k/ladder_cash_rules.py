"""Audit end-game tree changes on public replays, and replay live games in the original Board."""
import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/env_src/carbon_challenge-master"))
from zerosum_env.envs.carbon.helpers import Board
from zerosum_env.envs.carbon import idgen


def trees(observation):
    return {tid:(p, *tree) for p, player in enumerate(observation["players"]) for tid, tree in player[3].items()}


def audit(path, official):
    env = json.loads(gzip.decompress(path.read_bytes()))
    cfg, steps = env["configuration"], env["steps"]
    root = steps[0][0]["observation"]
    actions = Counter()
    late = Counter()
    prev = trees(root)
    total_checked = 0
    if official:
        idgen.reset()
        state = copy.deepcopy(root)
        state["carbon"] = state["full_carbon"]
    for t, pair in enumerate(steps[1:],1):
        obs = pair[0]["observation"]
        if obs["step"] != t:
            raise ValueError("nonconsecutive states")
        joint = [entry["action"] or {} for entry in pair]
        for commands in joint:
            actions.update(commands.values())
        cur = trees(obs)
        if t >= cfg["episodeSteps"] - 30:
            late["transitions"] += 1
            for tid in prev.keys() - cur.keys():
                late["disappeared"] += 1
                late["at_lifespan" if prev[tid][2]>=cfg["treeLifespan"] else "earlier_than_lifespan"] += 1
            late["changed_owner"] += sum(prev[k][0] != cur[k][0] for k in prev.keys() & cur.keys())
            late["planted"] += len(cur.keys()-prev.keys())
        if official:
            state = Board(state, cfg, joint).next().observation
            if state["players"] != obs["players"] or list(state["carbon"]) != list(obs["full_carbon"]):
                raise ValueError(f"original Board diverged from published replay {path.name} at {t}")
            total_checked += 1
        prev = cur
    last = steps[-1][0]["observation"]
    cash = [p[0] for p in last["players"]]
    return {"game":path.name.split('.')[0], "replay_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "transitions":len(steps)-1, "official_board_transitions_checked":total_checked, "actions":dict(actions),
            "last30":dict(late), "end_trees_by_seat":[len(p[3]) for p in last["players"]],
            "end_cargo_by_seat":[round(sum(w[1] for w in p[2].values()),2) for p in last["players"]],
            "end_cash":cash, "official_rewards":env["rewards"], "statuses":env["statuses"]}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live",required=True,type=Path,help="directory containing the two new ladder replays")
    parser.add_argument("--corpus",required=True,type=Path,help="root of the restored 108-match Top Five corpus")
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    online=[audit(path,True) for path in sorted(args.live.glob("*.json.gz"))]
    archived=[audit(path,False) for path in sorted(args.corpus.rglob("*.json.gz"))]
    if len(online)!=2 or len(archived)!=108 or len({g["game"] for g in online+archived})!=110:
        raise ValueError("fixed roster of two live plus 108 archived distinct games required")
    report={"live":online,"historical_top5":archived,
            "total_actions":dict(sum((Counter(r["actions"]) for r in online+archived),Counter())),
            "total_last30":dict(sum((Counter(r["last30"]) for r in online+archived),Counter()))}
    with args.out.open('x') as f:
        json.dump(report,f,indent=2,allow_nan=False)
        f.write('\n')

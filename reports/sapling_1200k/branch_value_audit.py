"""Check whether the frozen 30-step cash forecaster can rank actual legal alternatives."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path

from official_game_replay import make
from zerosum_env.utils import structify
from state_value.value import CashMargin30, History, HORIZON


HERE = Path(__file__).resolve().parent


def visible(frame, seat):
    shared = frame[0]["observation"]
    own = frame[seat]["observation"]
    return {"player": seat, "step": shared["step"], "players": shared["players"],
            "trees": shared["trees"], "carbon": own["carbon"]}


def margin(obs, seat):
    return obs["players"][seat][0] - obs["players"][1-seat][0]


def evaluate_arm(replay, case, arm, model):
    frames, config = replay["steps"], replay["configuration"]
    root, seat = case["root"]["step"], case["card"]["seat"]
    env = make("carbon", configuration=config)
    env.states = structify(copy.deepcopy(frames[0]))
    env.steps = [env.states]
    history = History()
    history.update(visible(env.states, seat))
    for turn in frames[1:root + 1]:
        env.step([side["action"] or {} for side in turn])
        observation = visible(env.states, seat)
        if observation != visible(turn, seat):
            raise ValueError("official root history diverged")
        history.update(observation)
    start = margin(visible(env.states, seat), seat)
    first_pred, first_cash, future_cash = None, None, None
    for n, orders in enumerate(case[arm]["actions"], start=1):
        if env.done:
            raise ValueError("action after terminal")
        env.step(orders)
        obs = visible(env.states, seat)
        if n == 1:
            history.update(obs)
            first_cash = margin(obs, seat)
            first_pred = model.predict(history, obs, config)
        if n == 1 + HORIZON:
            future_cash = margin(obs, seat)
    scores = [p.reward for p in env.states]
    if not env.done or scores != case[arm]["score_by_seat"]:
        raise ValueError("official terminal mismatch")
    return {"next_cash_margin": first_cash, "predicted_change_next30": first_pred,
            "projected_from_root": first_cash - start + first_pred,
            "actual_from_root_at31": None if future_cash is None else future_cash - start,
            "official_terminal_margin": scores[seat] - scores[1-seat]}


def run(source, checkpoint):
    model = CashMargin30(checkpoint)
    rows = []
    for folder in ["orchard_branches", "orchard_high_branches"]:
        plan = json.loads((HERE / folder / "plan.json").read_bytes())
        cards = {c["id"]: c for c in plan["maps"]}
        for file in sorted((HERE / folder).glob("*.json.gz")):
            case = json.loads(gzip.decompress(file.read_bytes()))
            if "root" not in case:
                continue
            card = cards[case["card"]["id"]]
            replay = json.loads(gzip.decompress((source / (card["id"] + ".json.gz")).read_bytes()))
            canonical = json.dumps(replay, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
            if hashlib.sha256(canonical).hexdigest() != card["canonical_sha256"] or card != case["card"]:
                raise ValueError("unbound source")
            move, buy = [evaluate_arm(replay, case, arm, model) for arm in ("move", "buy")]
            key = "actual_from_root_at31"
            if move[key] is None or buy[key] is None:
                actual30 = None
            else:
                actual30 = buy[key] - move[key]
            rows.append({"id": card["id"], "seed": card["seed"], "seat": card["seat"],
                         "step":case["root"]["step"], "branch_cohort": folder, "group":card["group"],
                         "actions_sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "move":move,"buy":buy,
                         "predicted_buy_minus_move":buy["projected_from_root"] - move["projected_from_root"],
                         "actual_buy_minus_move_at31":actual30,
                         "actual_buy_minus_move_at_terminal":buy["official_terminal_margin"]-move["official_terminal_margin"]})
    if len(rows) != len({row["id"] for row in rows}):
        raise ValueError("duplicate root map")
    summary = {}
    for target in ("actual_buy_minus_move_at31", "actual_buy_minus_move_at_terminal"):
        relevant = [r for r in rows if r[target] is not None and r[target] != 0]
        summary[target] = {"root_count": len(relevant),
                           "preferred_arm_correct": sum(r["predicted_buy_minus_move"] * r[target] > 0 for r in relevant),
                           "always_move_correct": sum(r[target] < 0 for r in relevant),
                           "always_buy_correct": sum(r[target] > 0 for r in relevant)}
    return {"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "summary":summary,
            "scope":"First successor only; both alternatives faced the same opposing root action. Six ordinary and two high-scoring branch roots on the value head's historical training-map pool. Unseen branch futures; descriptive in-pool diagnostic, never a formal strength estimate.",
            "rows":rows}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replays",type=Path,required=True)
    p.add_argument("--weights",type=Path,default=HERE/"state_value/cash_margin_30.npz")
    p.add_argument("--out",type=Path,default=HERE/"branch_value_audit.json")
    args=p.parse_args()
    result=run(args.replays,args.weights)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(result["summary"])

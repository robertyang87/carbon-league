"""Reconcile both full-map arms, and verify each issued intervention against its actual joint command."""

from collections import Counter
import gzip
import importlib.util
import json
import math
import tempfile
from pathlib import Path
import shutil
import sys

from transfer import BASE, DEV, HERE, inputs, sha

sys.path.insert(0, str(DEV / "freight_horizon"))
from data import FastCarbon, FreightCash, add_step, target


def load_pack(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def commands(values):
    return {key: value for key, value in values.items() if value not in (None, "NONE")}


def audit_pack(path, candidate, raw, terrain):
    # Each session starts with a fresh module and the recorded map. The opponent commands
    # are historical facts; replay only establishes what the actor issued on that trajectory.
    with tempfile.TemporaryDirectory() as temp:
        pack = Path(temp) / "submission.py"
        shutil.copyfile(path, pack)
        mod = load_pack(pack, "same_tape_transfer")
        captured = []
        if candidate:
            original = mod._original_agent

            def capture(*args):
                result = original(*args)
                captured[:] = [commands(result)]
                return result

            mod._original_agent = capture
        seat = raw["candidate_seat"]
        cfg = raw["game_configuration"]
        sim = FastCarbon(cfg).reset(seed=raw["seed"])
        from map_identity import terrain_sha256

        if terrain_sha256(sim.carbon) != terrain:
            raise ValueError("actor replay opening mismatch")
        sim.cash = FreightCash(sim.cash)
        events, interventions = [], []
        reasons = Counter()
        for t, joint in enumerate(raw["actions"]):
            public = sim.obs(seat)
            issued = commands(mod.agent(public, cfg))
            if issued != commands(joint[seat]):
                raise ValueError(
                    f"actual pack does not reproduce issued joint command at {t}/{seat}"
                )
            if candidate:
                before = captured[0]
                memo = mod._FREIGHT.last
                reasons[memo["reason"]] += 1
                changed = [
                    key
                    for key in set(issued) | set(before)
                    if issued.get(key) != before.get(key)
                ]
                if bool(changed) != memo["changed"]:
                    raise ValueError("runtime count differs from issued changes")
                if changed:
                    offer = memo["proposal"]
                    uid = offer["worker"]
                    if (
                        changed != [uid]
                        or issued.get(uid, "NONE") != offer["alternative"]
                    ):
                        raise ValueError("intervention changed another command")
                    target_id = offer["target_worker"]
                    cargo = sim.workers[target_id].carbon
                    interventions.append(
                        dict(
                            t=t,
                            seat=seat,
                            own_worker=uid,
                            target=target_id,
                            cargo=cargo,
                            before=before.get(uid, "NONE"),
                            after=issued.get(uid, "NONE"),
                            prediction=memo["public_predictions"][target_id],
                        )
                    )
            events.append(add_step(sim, joint))
        if len(interventions) > 1 or list(sim.cash) != raw["last_post_cash"]:
            raise ValueError("intervention cap or terminal cash failed")
        for case in interventions:
            uid, cargo, t = case["target"], case["cargo"], case["t"]
            frac, deposited, gross, principal, first = target(
                uid, cargo, 1 - seat, t, events
            )
            receipts = [
                dict(t=e["t"], value=e["payments"][uid][1])
                for e in events[t:]
                if uid in e["payments"] and e["payments"][uid][0] == 1 - seat
            ]
            case.update(
                target_paid_within_10=bool(deposited),
                target_10_gross=gross,
                target_10_initial_cargo_paid=principal,
                initial_cargo_paid_fraction=frac,
                first_payment_t=first if deposited else None,
                all_remaining_target_receipts=receipts,
            )
        return dict(
            exact_replayed_joint_commands=len(raw["actions"]),
            interventions=interventions,
            root_reason_counts=dict(sorted(reasons.items())),
            network_counters=dict(mod._COUNTS),
        )


def read_arm(arm):
    seed, terrain, pack, _ = inputs(arm)
    folder = HERE / ("transfer_" + arm)
    official = json.loads((folder / "official_check.json").read_bytes())
    results = json.loads((folder / "results.json").read_bytes())
    if official["native_results_sha256"] != sha(folder / "results.json") or official[
        "pack_sha256"
    ] != sha(pack):
        raise ValueError("official receipt or pack identity mismatch")
    rows = []
    for row in results["games"]:
        seat = row["candidate_seat"]
        truth = next(g for g in official["games"] if g["seat"] == seat)
        raw_path = folder / row["raw_file"]
        raw = json.loads(gzip.decompress(raw_path.read_bytes()))
        if (
            sha(raw_path) != truth["transcript_sha256"]
            or raw["seed"] != seed
            or truth["map_sha256"] != terrain
            or row["audit_messages"]
            or not row["valid"]
        ):
            raise ValueError("unexpected development evidence")
        scores = truth["official_reward_by_seat"]
        margin = scores[seat] - scores[1 - seat]
        wins = (margin > 0) + 0.5 * (margin == 0)
        utility = 0.5 * wins + 0.5 / (1 + math.exp(-margin / 1000))
        economics = truth["income"]["cash_entries_by_seat"]
        rows.append(
            dict(
                seat=seat,
                result=truth["result"],
                margin=round(margin, 2),
                u=utility,
                official_by_seat=scores,
                end=truth["steps"],
                termination=row["termination"],
                examiner_search=row["search"],
                examiner_call_types=row["search_return_types"],
                examiner_wall_seconds=row["call_seconds_by_seat"][1 - seat],
                own_economy=economics[seat],
                opponent_economy=economics[1 - seat],
                **audit_pack(pack, arm == "candidate", raw, terrain),
            )
        )
    if sorted(r["seat"] for r in rows) != [0, 1]:
        raise ValueError("unexpected seat coverage")
    return sorted(rows, key=lambda r: r["seat"])


def compare_before_intervention(candidate):
    seat = candidate["seat"]
    seed = inputs("candidate")[0]
    runs = []
    for arm in ("parent", "candidate"):
        path = HERE / ("transfer_" + arm) / f"raw_{seed}_{seat}.json.gz"
        raw = json.loads(gzip.decompress(path.read_bytes()))
        sim = FastCarbon(raw["game_configuration"]).reset(seed=seed)
        runs.append((raw["actions"], sim))
    cutoff = (
        candidate["interventions"][0]["t"]
        if candidate["interventions"]
        else min(len(acts) for acts, sim in runs)
    )
    first_own = first_opp = None
    for t in range(cutoff):
        left, right = [acts[t] for acts, sim in runs]
        if first_own is None and commands(left[seat]) != commands(right[seat]):
            first_own = t
        if first_opp is None and commands(left[1 - seat]) != commands(right[1 - seat]):
            first_opp = t
        for acts, sim in runs:
            sim.step(*acts[t])
    margins = [round(sim.cash[seat] - sim.cash[1 - seat], 2) for acts, sim in runs]
    return dict(
        seat=seat,
        before_t=cutoff,
        first_own_command_divergence=first_own,
        first_examiner_command_divergence=first_opp,
        parent_cash_margin=margins[0],
        candidate_cash_margin=margins[1],
        delta_margin_before=round(margins[1] - margins[0], 2),
    )


def write_report(summary):
    seed, terrain, _, _ = inputs("candidate")
    lines = [
        f"# 首次拦运策略在开发图 {seed} 的双席迁移结果",
        "",
        "四局均使用仓内原版 Master3s2 的搜索、温度与官方结算。每局保留双方逐步指令，官方解释器重放整局，再按实际现金入账拆分运货和树收益。",
        "",
        "| 席 | 开发臂 | 胜平负 | 步数 | 我方终分 | 对方终分 | 分差 | U | 我方回库 | 敌方回库 | 我方树吸收 | 敌方树吸收 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seat in (0, 1):
        for arm, name in (("parent", "双网亲本"), ("candidate", "首次拦运")):
            row = summary["arms"][arm][seat]
            ours, theirs = row["own_economy"], row["opponent_economy"]
            scores = row["official_by_seat"]
            wdl = {"win": "胜", "draw": "平", "loss": "负"}[row["result"]]
            lines.append(
                f"| {seat} | {name} | {wdl} | {row['end']} | {scores[seat]:.2f} | {scores[1-seat]:.2f} | {row['margin']:+.2f} | {row['u']:.4f} | {ours.get('own_delivery',0):.2f} | {theirs.get('own_delivery',0):.2f} | {ours.get('tree_absorption',0):.2f} | {theirs.get('tree_absorption',0):.2f} |"
            )
    d0, d1 = summary["paired_candidate_minus_parent"]
    pat = sum(len(x["interventions"]) for x in summary["arms"]["candidate"])
    lines += [
        "",
        f"候选相对同席亲本的分差分别为 {d0['margin']:+.2f} 与 {d1['margin']:+.2f}，单图双席均差 {summary['delta_mean_margin']:+.2f}。两席合计实际下达 {pat} 次拦运改令。",
        "",
        "### 已下达的改令与后续现金",
        "",
    ]
    for row in summary["arms"]["candidate"]:
        if not row["interventions"]:
            lines.append(
                f"- 席 {row['seat']} 在 {row['end']} 个决策中沿用亲本动作。候选在这条轨迹的逐步输出已与行动带核实。"
            )
        for cut in row["interventions"]:
            receipts = cut["all_remaining_target_receipts"]
            paid = sum(z["value"] for z in receipts)
            left = (
                "、".join(f"t{z['t']} 入库 {z['value']:.2f}" for z in receipts)
                or "余局未入库"
            )
            lines.append(
                f"- 席 {row['seat']} 的 t{cut['t']}，{cut['own_worker']} 从 {cut['before']} 改为 {cut['after']}，其余联合指令一致。目标 {cut['target']} 当时载货 {cut['cargo']:.2f}，模型预测十步入库概率 {cut['prediction']['p_deposit_10']:.3f}。实际十步入库 {cut['target_10_gross']:.2f}，其中根时已有货兑现 {cut['target_10_initial_cargo_paid']:.2f}。该目标至终局总入库 {paid:.2f}（{left}）。"
            )
    lines += [
        "",
        "本次只有一张独立开发图。两臂考官分别运行原生采样和墙钟搜索，局面可在拦运前就分化。表中双席分差只能描述整包实战，目标现金只描述相应实发轨迹；跨地图区间和单次拦运的因果收益均无法从这四局估计。",
        "",
        f"初始地图 225 格碳量 SHA-256 为 `{terrain}`。",
        "",
        "[预登记](transfer_plan.json) · [地图收据](map_preflight.json) · [亲本官方核账](transfer_parent/official_check.json) · [候选官方核账](transfer_candidate/official_check.json) · [逐根改令与现金明细](transfer_summary.json)",
    ]
    evidence = []
    for row in summary["pre_intervention_trajectories"]:
        if row["first_examiner_command_divergence"] is None:
            note = "两条轨迹的考官指令到此时仍然相同"
        else:
            note = f"两条轨迹的考官指令最早在 t{row['first_examiner_command_divergence']} 就分开"
        evidence.append(
            f"- 席 {row['seat']}，{note}。在 t{row['before_t']} 决策前，亲本局本方现金差为 {row['parent_cash_margin']:+.2f}，候选局为 {row['candidate_cash_margin']:+.2f}，当时两局已相差 {row['delta_margin_before']:+.2f}。"
        )
    pos = next(i for i, x in enumerate(lines) if x.startswith("本次只有"))
    lines[pos:pos] = ["### 改令前的分歧", "", *evidence, ""]
    (HERE / "TRANSFER.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parent, candidate = read_arm("parent"), read_arm("candidate")
    changes = [
        dict(
            seat=p["seat"],
            margin=round(c["margin"] - p["margin"], 2),
            u=c["u"] - p["u"],
        )
        for p, c in zip(parent, candidate)
    ]
    summary = dict(
        arms={"parent": parent, "candidate": candidate},
        paired_candidate_minus_parent=changes,
        pre_intervention_trajectories=[
            compare_before_intervention(c) for c in candidate
        ],
        delta_mean_margin=round(sum(d["margin"] for d in changes) / 2, 2),
        independent_maps=1,
        across_map_interval=None,
        interpretation="Native opponent sampling and wall-clock search can differ between arms. A full-map margin difference cannot identify the causal effect of a particular issued command.",
    )
    (HERE / "transfer_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    write_report(summary)

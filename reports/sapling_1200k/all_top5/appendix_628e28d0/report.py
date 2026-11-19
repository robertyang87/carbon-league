"""Summarize independently reconstructed play at the frozen 628e28d0 ladder cutoff."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
ROOT = TOP.parents[2]
sys.path.insert(0, str(TOP))
from defense_summary import ranking_lookup
from master6_pressure import MASTER6


def read(name):
    data = (HERE / name).read_bytes()
    return json.loads(gzip.decompress(data) if name.endswith(".gz") else data)


def write_json(name, content):
    (HERE / name).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n")


def build():
    manifest = read("manifest.json")
    delta = read("score_delta.json")
    validation = read("validation.json")
    context = read("context_receipt.json")
    exposure = read("exposure.json")
    dev = read("development_maps.json")
    cases = {r["id"]: r for r in read("cases.json.gz")["rows"]}
    cash = {r["id"]: r for r in read("turn_cash.json.gz")["games"]}
    n = sum(len(r["turns"]) for r in cash.values())
    assert set(cash) == set(cases) == {g["id"] for g in manifest["games"]}
    assert validation["fastsim_transitions"] == validation["original_board_transitions"] == n
    assert validation["score_cards_agree"] == context["games"] == len(cash)
    assert context["counts"]["decisions"] == 2*n and exposure["complete_scored_games"] == delta["union_games"]
    assert not dev["seen_225_cell_layout_matches"] and not dev["seen_seed_matches"]
    rank_source = TOP.parent / "latest5_inventory_sources.json.gz"
    rank_bytes = subprocess.check_output(["git", "-C", str(ROOT), "show",
                                          f'{manifest["revision"]}:{rank_source.relative_to(ROOT)}'])
    assert rank_source.read_bytes() == rank_bytes
    tier_for, rank_date = ranking_lookup()
    frozen_ids = {p["id"] for p in manifest["top5"]}
    rows, masters = [], []
    for g in manifest["games"]:
        gid = g["id"]
        seats = [i for i, pid in enumerate(g["seat_strategy_ids"]) if pid in frozen_ids]
        assert len(seats) == 1
        seat = seats[0]
        opp = 1-seat
        stats = cases[gid]["economy"]
        swing = stats["swing"]
        snaps = {snap["t"]: snap["seats"] for snap in swing["positions"]}
        lookback = []
        for offset in (10, 5, 3, 0):
            t = max(0, swing["a"]-offset)
            lookback.append(dict(t=t, offset=offset, top_seat_economy=snaps[t][seat], opponent_economy=snaps[t][opp]))
        top_score, opp_score = (g["score_by_seat"][i] for i in (seat, opp))
        margin = round(top_score-opp_score, 2)
        row = dict(game=gid, seed=g["seed"], top_seat=seat, names=[g["seat_names"][i] for i in (seat,opp)],
                   scores=[top_score,opp_score], terminal_margin=margin, steps=g["transitions"],
                   largest_ten_step_window=[swing["a"],swing["b"]], top_margin_delta=(1-2*seat)*swing["margin0_delta"],
                   delivery=[swing["seats"][i]["bank"] for i in (seat,opp)],
                   tree_cash=[swing["seats"][i]["tree"] for i in (seat,opp)], lookback=lookback)
        rows.append(row)
        # The shared case renderer links to the base batch's economy shard. Bind this page to its own ledger.
        page = HERE / (gid + ".md")
        content = re.sub(r"\[同批机器可读证据\]\([^)]+\)",
                         "[本批逐拍收入账本](turn_cash.json.gz)", page.read_text())
        assert "[本批逐拍收入账本](turn_cash.json.gz)" in content
        page.write_text(content)
        if g["seat_strategy_ids"][seat] == MASTER6:
            masters.append(dict(game=gid, seed=g["seed"], master_seat=seat, opponent=g["seat_names"][opp],
                                opponent_tier=tier_for(g["seat_strategy_ids"][opp], g["seat_names"][opp]),
                                result="win" if margin>0 else "loss" if margin<0 else "draw", terminal_margin=margin,
                                last30_margin_change=(1-2*seat)*stats["last30"]["margin0_delta"],
                                last30_delivery=[stats["last30"]["seats"][i]["bank"] for i in (seat,opp)],
                                engine_score=g["score_by_seat"]))
    write_json("turning_points.json", {"seat_order":"top strategy, opponent", "games":rows})
    middle = [r for r in masters if r["opponent_tier"] == "21–50"]
    unknown = [r for r in masters if r["opponent_tier"] == "unknown"]
    write_json("master6.json", dict(frozen_heavy_board_utc=rank_date, source=str(rank_source.relative_to(ROOT)),
               source_sha256=hashlib.sha256(rank_bytes).hexdigest(), middle_tier_definition="heavy ranks 21–50",
               wins=[r["game"] for r in masters if r["result"]=="win"], losses=[r["game"] for r in masters if r["result"]=="loss"],
               draws=[r["game"] for r in masters if r["result"]=="draw"], confirmed_middle_tier=[r["game"] for r in middle],
               unknown_tier=[r["game"] for r in unknown], games=masters))
    lines = ["# 截至 628e28d0 的前五新增六场", "",
             f"从 `{delta['previous_score_revision'][:8]}` 到 `{manifest['revision'][:8]}`，在截止点全仓确认 347 个来源文件。本次有变化的来源所含比赛引用从 "
             f"{delta['old_reference_count']:,} 条增至 {delta['new_reference_count']:,} 条。其中 1 条补记已有局，24 条双方都未进入冻结前五，另有 10 条指向本批 6 局。"
             f"6 局均为对外比赛，新增故障 0。累计 {delta['union_games']:,} 局有双方官方终分，其中互打 121 局、对外 1,515 局。另计 1 局历史故障。"
             " [成绩逐条去重](score_delta.json) · [来源与回放身份](manifest.json)", "",
             "| 前五策略 / 物理席 | 对手 | 官方终分（前五 / 对手） | 前五终场分差 | 最大十步窗口 | 窗口分差变化 | 窗口正常交仓（前五 / 对手） | 窗口树收入（前五 / 对手） |",
             "|---|---|---:|---:|---|---:|---:|---:|"]
    for r in rows:
        a,b = r["scores"]
        t,u = r["largest_ten_step_window"]
        c,d = r["delivery"]
        e,f = r["tree_cash"]
        lines.append(f"| [{r['names'][0]} / {r['top_seat']}]({r['game']}.md) | {r['names'][1]} | "
                     f"{a:,.2f} / {b:,.2f} | {r['terminal_margin']:+,.2f} | {t}–{u} | {r['top_margin_delta']:+,.2f} | "
                     f"{c:,.2f} / {d:,.2f} | {e:,.2f} / {f:,.2f} |")
    lines.extend(["", "## Master6 的胜、负和中游对手", ""])
    label = {"win":"胜局", "loss":"负局", "draw":"平局"}
    for r in sorted(masters, key=lambda v:v["terminal_margin"], reverse=True):
        a,b = r["last30_delivery"]
        lines.append(f"- **{label[r['result']]}**　种子 {r['seed']}，Master6 席 {r['master_seat']}。最后 30 拍，"
                     f"Master6 和对方各交仓 {a:,.2f}、{b:,.2f}，现金分差变化 {r['last30_margin_change']:+,.2f}，"
                     f"最终分差 {r['terminal_margin']:+,.2f}。[逐拍与收入]({r['game']}.md)")
    lines.extend([f"- **对中游**　按 {rank_date} 冻结 heavy 榜的第 21–50 名计算，本批有 {len(middle)} 局。"
                  f"同一对手 it920-evo31 的 {len(unknown)} 局在该快照均无可核验档位。[分档与原始名次表指纹](master6.json)", "",
                  "每个逐局页面列出现金差绝对变化最大的十拍窗口，并回到窗前 3、5、10 拍，查看双方的工人、树、基地、在途货、可见状态与实际指令。"
                  "这些窗口按赛后现金变化选取，用于追溯收入构成。[双方状态](turning_points.json) · [逐拍八项现金](turn_cash.json.gz) · [逐拍指令](turn_actions.json.gz) · [进攻防守与树权](cases.json.gz) · [后续 10、30 拍收入抽核](future_economy_check.json)", "",
                  f"官方 Board 与 FastCarbon 各复算 {n:,} 个联合回合，六场官方终分全部吻合。"
                  f"双方共 {2*n:,} 个行动前决策通过上下文核验。[双引擎收据](validation.json) · [双席上下文](context_receipt.json) · [完整官方回放](replays.tar)", "",
                  f"6 个新增初盘按录像内 225 格碳量计算 SHA-256；历史全部 {exposure['distinct_archived_terrains_including_fault']:,} 个初盘（含故障）各异。"
                  f"六局与冻结时已登记或排除的 {exposure['known_local_or_conservatively_excluded_physical_boards']:,} 个物理图及开发图 621758400、621758401 在种子和地形上均无碰撞。"
                  " [地形逐局指纹](development_maps.json) · [暴露与预留范围](exposure.json)", "",
                  "逐局读取状态、双方可见信息与结果可用 `context.Corpus(manifest=Path('reports/sapling_1200k/all_top5/appendix_628e28d0/manifest.json'))`。本批独立索引与收据放在同一附录内。", "",
                  "复现时在仓库根目录运行 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 taskset -c 2 nice -n 15 /workspace/carbon-review-venv/bin/python reports/sapling_1200k/all_top5/appendix_628e28d0/prepare.py`。程序可从本批归档恢复官方回放。", ""])
    (HERE / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    build()

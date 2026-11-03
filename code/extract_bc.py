#!/usr/bin/env python3
"""Extract behavior-cloning datasets from Carbon League public replays.

Inputs (local files only, no network):
  replays/index.json          - one record per game: strategies a/b (name,id),
                                scores, winner ('a'/'b'), order, file.
  replays/replays/*.json.gz   - Kaggle-style env JSON per game.

Key format facts (verified against the files, see bc_data/README.md):
  * steps[t] is a per-player list of 2 entries. Entry p holds player p's
    `action` for step t and player p's *own* partial observation.
  * Only entry 0's observation carries the shared fields (players, trees,
    step). Entry 1's observation has only carbon/player/remainingOverageTime.
    `players` and `trees` are shared (identical for both players), so a
    player-1 sample combines entry 1's own `carbon` with entry 0's shared
    `players`/`trees`.
  * order[p] is the strategy slot ('a' or 'b') that replay player p played
    as. index.winner is a slot, not a player index.
  * observation['carbon'] is that player's *visible* carbon (225 cells),
    not the full map. observation also contains 'full_carbon' (hidden info
    used by the replay renderer); we deliberately do NOT extract it.
  * Trees (pos, age, owner) come from players[i][3] = {tree_id: [pos, age]}.
    observation['trees'] is NOT used: it maps tree_id -> [worker_id, absorbed]
    which is absorption bookkeeping, not tree state.

Outputs:
  bc_data/claude_family.npz   - claude-family side of every game involving a
                                claude-family strategy (name matches
                                /claude|calude|alphacarbon/i).
  bc_data/top_players.npz     - winner side of games whose winner is a
                                "head" strategy (see HEAD RULE below) and
                                whose winning score >= 8000.
  bc_data/README.md           - regenerated with the run's statistics.

HEAD RULE for top_players (computed from index.json alone, deterministic):
  a strategy name is "head" if, across the whole snapshot, it played >= 3
  games, won >= 60% of them, and its average own score is >= 8000.

Sample rules:
  * one sample per step t for the selected player, from that player's
    observation at t and that player's action dict at t;
  * steps whose entry status != 'ACTIVE' are skipped (in practice only the
    terminal entry, whose recorded action was never executed);
  * a unit absent from the action dict is labelled stay / none;
  * worker/base/trees slots are sorted by the numeric id suffix, so slot
    order is stable and matches between features and labels.

Run:  python3 extract_bc.py
"""

import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / "replays" / "index.json"
OUT_DIR = ROOT / "bc_data"

CLAUDE_RE = re.compile(r"claude|calude|alphacarbon", re.I)

MAX_WORKERS = 10
MAX_BASES = 2
MAX_TREES = 30

WORKER_ACT = {"UP": 1, "RIGHT": 2, "DOWN": 3, "LEFT": 4}  # 0 = stay (absent)
BASE_ACT = {"RECCOLLECTOR": 1, "RECPLANTER": 2}           # 0 = none (absent)
WORKER_TYPE = {"COLLECTOR": 0, "PLANTER": 1}

HEAD_MIN_GAMES = 3
HEAD_MIN_WINRATE = 0.60
HEAD_MIN_AVG_SCORE = 8000.0
TOP_MIN_WINNER_SCORE = 8000.0


def num_suffix(uid: str) -> int:
    try:
        return int(uid.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return 0


def slot_of_player(game, player: int) -> str:
    return game["order"][player]


def player_of_slot(game, slot: str) -> int:
    return game["order"].index(slot)


def load_game(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def extract_player_samples(game_meta, replay, player: int, counters):
    """Yield sample dicts for one player of one game."""
    slot = slot_of_player(game_meta, player)
    opp_slot = "b" if slot == "a" else "a"
    strategy = game_meta["strategies"][slot]["name"]
    opponent = game_meta["strategies"][opp_slot]["name"]
    score_self = float(game_meta["scores"][slot])
    score_opp = float(game_meta["scores"][opp_slot])
    won = game_meta.get("winner") == slot

    steps = replay["steps"]
    for t, entries in enumerate(steps):
        entry = entries[player]
        if entry.get("status") != "ACTIVE":
            counters["skipped_non_active"] += 1
            continue
        shared = entries[0]["observation"]          # shared players/trees/step
        own = entry["observation"]                  # this player's visible carbon
        players = shared["players"]
        actions = entry.get("action") or {}

        # mapping sanity: a player's action keys must be that player's units
        for uid in actions:
            if not uid.startswith(f"player-{player}-"):
                counters["action_key_mismatch"] += 1

        carbon = np.asarray(own["carbon"], dtype=np.float32)
        assert carbon.shape == (225,), carbon.shape

        cash = np.zeros(2, dtype=np.float32)
        bases = np.full((2, MAX_BASES), -1, dtype=np.int16)
        workers = np.zeros((2, MAX_WORKERS, 3), dtype=np.float32)
        workers_mask = np.zeros((2, MAX_WORKERS), dtype=bool)

        sorted_worker_ids = {}
        for i in range(2):
            p_cash, p_bases, p_workers, _p_trees = players[i]
            cash[i] = float(p_cash)
            for j, cid in enumerate(sorted(p_bases, key=num_suffix)[:MAX_BASES]):
                bases[i, j] = int(p_bases[cid])
            wids = sorted(p_workers, key=num_suffix)
            sorted_worker_ids[i] = wids
            for j, wid in enumerate(wids[:MAX_WORKERS]):
                pos, cargo, wtype = p_workers[wid]
                workers[i, j, 0] = float(pos)
                workers[i, j, 1] = float(cargo)
                workers[i, j, 2] = float(WORKER_TYPE[wtype])
                workers_mask[i, j] = True
            if len(wids) > MAX_WORKERS:
                counters["workers_overflow"] += 1

        # trees from players[i][3]: {tree_id: [pos, age]}, owner = i
        tree_rows = []
        for i in range(2):
            for tid, (pos, age) in players[i][3].items():
                tree_rows.append((i, num_suffix(tid), float(pos), float(age)))
        tree_rows.sort()
        if len(tree_rows) > MAX_TREES:
            counters["trees_overflow"] += 1
        trees = np.zeros((MAX_TREES, 3), dtype=np.float32)
        trees_mask = np.zeros(MAX_TREES, dtype=bool)
        for j, (owner, _n, pos, age) in enumerate(tree_rows[:MAX_TREES]):
            trees[j] = (pos, age, float(owner))
            trees_mask[j] = True

        # labels for the acting player only, aligned with sorted slots above
        worker_actions = np.zeros(MAX_WORKERS, dtype=np.int8)
        for j, wid in enumerate(sorted_worker_ids[player][:MAX_WORKERS]):
            act = actions.get(wid)
            if act is None:
                continue
            if act in WORKER_ACT:
                worker_actions[j] = WORKER_ACT[act]
            else:
                counters["unknown_worker_action"] += 1
        base_actions = np.zeros(MAX_BASES, dtype=np.int8)
        my_bases = players[player][1]
        for j, cid in enumerate(sorted(my_bases, key=num_suffix)[:MAX_BASES]):
            act = actions.get(cid)
            if act is None:
                continue
            if act in BASE_ACT:
                base_actions[j] = BASE_ACT[act]
            else:
                counters["unknown_base_action"] += 1

        yield {
            "step": int(shared.get("step", t)),
            "player": player,
            "carbon": carbon,
            "cash": cash,
            "bases": bases,
            "workers": workers,
            "workers_mask": workers_mask,
            "trees": trees,
            "trees_mask": trees_mask,
            "worker_actions": worker_actions,
            "base_actions": base_actions,
            "file": game_meta["file"],
            "strategy": strategy,
            "opponent": opponent,
            "score_self": score_self,
            "score_opp": score_opp,
            "won": won,
        }


def pack(samples):
    def stack(key, dtype=None):
        arr = np.stack([s[key] for s in samples])
        return arr.astype(dtype) if dtype else arr

    return {
        "step": stack("step", np.int16),
        "player": stack("player", np.int8),
        "carbon": stack("carbon", np.float32),
        "cash": stack("cash", np.float32),
        "bases": stack("bases", np.int16),
        "workers": stack("workers", np.float32),
        "workers_mask": stack("workers_mask", bool),
        "trees": stack("trees", np.float32),
        "trees_mask": stack("trees_mask", bool),
        "worker_actions": stack("worker_actions", np.int8),
        "base_actions": stack("base_actions", np.int8),
        "game_file": np.array([s["file"] for s in samples]),
        "strategy": np.array([s["strategy"] for s in samples]),
        "opponent": np.array([s["opponent"] for s in samples]),
        "score_self": stack("score_self", np.float32),
        "score_opp": stack("score_opp", np.float32),
        "won": stack("won", bool),
    }


def dataset_stats(samples):
    per_strategy = Counter(s["strategy"] for s in samples)
    per_game = Counter(s["file"] for s in samples)
    # action distribution over *valid* slots only (mask=1 workers / real bases)
    wdist = Counter()
    bdist = Counter()
    for s in samples:
        p = s["player"]
        valid = s["workers_mask"][p]
        for a, v in zip(s["worker_actions"], valid):
            if v:
                wdist[int(a)] += 1
        nb = int((s["bases"][p] >= 0).sum())
        for a in s["base_actions"][:nb]:
            bdist[int(a)] += 1
    return per_strategy, per_game, wdist, bdist


def main():
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(exist_ok=True)

    # ---- head-strategy table from the index alone -------------------------
    stats = defaultdict(lambda: {"games": 0, "wins": 0, "score": 0.0})
    for g in index:
        if g["scores"].get("a") is None or g["scores"].get("b") is None:
            continue  # one side timed out; its score is null in the index
        for slot in ("a", "b"):
            name = g["strategies"][slot]["name"]
            stats[name]["games"] += 1
            stats[name]["score"] += float(g["scores"][slot])
            if g.get("winner") == slot:
                stats[name]["wins"] += 1
    head = {}
    for name, s in stats.items():
        if s["games"] >= HEAD_MIN_GAMES:
            wr = s["wins"] / s["games"]
            avg = s["score"] / s["games"]
            if wr >= HEAD_MIN_WINRATE and avg >= HEAD_MIN_AVG_SCORE:
                head[name] = (s["games"], wr, avg)

    # ---- game selection ----------------------------------------------------
    claude_games = []   # (meta, player) for the claude-family side
    for g in index:
        for slot in ("a", "b"):
            if CLAUDE_RE.search(g["strategies"][slot]["name"]):
                claude_games.append((g, player_of_slot(g, slot)))

    top_games = []      # (meta, player) for the winner side
    for g in index:
        if g["scores"].get("a") is None or g["scores"].get("b") is None:
            continue
        w = g.get("winner")
        if w in ("a", "b"):
            wname = g["strategies"][w]["name"]
            if wname in head and float(g["scores"][w]) >= TOP_MIN_WINNER_SCORE:
                top_games.append((g, player_of_slot(g, w)))

    # ---- extraction ---------------------------------------------------------
    replay_cache = {}

    def get_replay(meta):
        if meta["file"] not in replay_cache:
            replay_cache[meta["file"]] = load_game(ROOT / "replays" / meta["file"])
        return replay_cache[meta["file"]]

    results = {}
    for label, selection in (("claude_family", claude_games), ("top_players", top_games)):
        counters = Counter()
        samples = []
        for meta, player in selection:
            samples.extend(extract_player_samples(meta, get_replay(meta), player, counters))
        results[label] = (samples, counters)
        np.savez_compressed(OUT_DIR / f"{label}.npz", **pack(samples))
        print(f"{label}: {len(selection)} games -> {len(samples)} samples; counters={dict(counters)}")

    # ---- README --------------------------------------------------------------
    def fmt_dist(dist, names):
        total = sum(dist.values()) or 1
        return ", ".join(
            f"{names[k]}={dist.get(k, 0)} ({100.0 * dist.get(k, 0) / total:.1f}%)"
            for k in sorted(names)
        )

    lines = []
    lines.append("# Carbon League 行为克隆数据集\n")
    lines.append("由 `extract_bc.py` 从 `replays/` 公开回放快照（400 局）抽取并自动生成，可重跑。\n")

    lines.append("## 数据集与筛选口径\n")
    c_samples, _ = results["claude_family"]
    t_samples, _ = results["top_players"]
    c_strat, c_game, c_w, c_b = dataset_stats(c_samples)
    t_strat, t_game, t_w, t_b = dataset_stats(t_samples)
    overlap = len(set(c_game) & set(t_game))
    lines.append(
        f"- **claude_family.npz**：策略名匹配 `/claude|calude|alphacarbon/i` 的对局共 "
        f"{len(claude_games)} 局，只收 claude 家族一方的样本：**{len(c_samples)}** 个决策点，"
        f"来自 {len(c_game)} 局。")
    lines.append(
        f"- **top_players.npz**：头部策略口径 = 在 400 局快照中出场 ≥{HEAD_MIN_GAMES} 局、"
        f"胜率 ≥{HEAD_MIN_WINRATE:.0%}、平均分 ≥{HEAD_MIN_AVG_SCORE:.0f}（按 index.json 计算，"
        f"共 {len(head)} 个策略名达标）；收其中胜方分 ≥{TOP_MIN_WINNER_SCORE:.0f} 的对局的胜方样本："
        f"**{len(t_samples)}** 个决策点，来自 {len(t_game)} 局。")
    lines.append(f"- 两个数据集重叠对局 {overlap} 局（同一局可同时入选两边，样本一方不同或相同视口径而定）。\n")

    lines.append("头部策略名单（出场/胜率/平均分）：\n")
    for name, (gm, wr, avg) in sorted(head.items(), key=lambda kv: -kv[1][2]):
        lines.append(f"- {name}: {gm} 局 / 胜率 {wr:.0%} / 平均分 {avg:.0f}")
    lines.append("")

    for title, samples, strat, game, wd, bd in (
        ("claude_family", c_samples, c_strat, c_game, c_w, c_b),
        ("top_players", t_samples, t_strat, t_game, t_w, t_b),
    ):
        lines.append(f"## {title} 样本统计\n")
        lines.append(f"- 总决策点：{len(samples)}；对局数：{len(game)}")
        lines.append("- 每策略样本数：" + "；".join(f"{k}={v}" for k, v in strat.most_common()))
        lines.append("- 工人动作分布（仅有效工人槽）："
                     + fmt_dist(wd, {0: "stay", 1: "UP", 2: "RIGHT", 3: "DOWN", 4: "LEFT"}))
        lines.append("- 基地动作分布（仅有效基地槽）："
                     + fmt_dist(bd, {0: "none", 1: "RECCOLLECTOR", 2: "RECPLANTER"}))
        lines.append("")

    counters_all = results["claude_family"][1] + results["top_players"][1]
    lines.append("## 校验计数（两数据集合计）\n")
    for k in ("skipped_non_active", "action_key_mismatch", "unknown_worker_action",
              "unknown_base_action", "workers_overflow", "trees_overflow"):
        lines.append(f"- {k}: {counters_all.get(k, 0)}")
    lines.append("")

    lines.append(FIELDS_DOC)
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT_DIR / "README.md")


FIELDS_DOC = """## 字段定义（两份 npz 同构）

每个样本 = 某一局某一步、被收录一方（`player`）在该步的观测 + 该步实际动作。
样本按对局（index 顺序）→ 步 t 升序排列；同一步只收被选中的一方。

| 字段 | shape / dtype | 含义 |
|---|---|---|
| `step` | (N,) int16 | 步号（取共享观测的 step，与回放 steps 下标一致） |
| `player` | (N,) int8 | 样本方在回放内的玩家下标 0/1。注意这不是策略槽 a/b：`order[player]` 才是槽位，抽取时已用 index.json 的 order 映射核对 |
| `carbon` | (N,225) float32 | 样本方该步的**可见碳**，行主序 15x15（index = row*15+col）。不是全图碳，见「已知坑」 |
| `cash` | (N,2) float32 | 双方资金，[player0, player1]（共享字段，双方可见） |
| `bases` | (N,2,2) int16 | 双方各自 2 个招募中心的位置 index，按中心 id 数字后缀排序；缺失填 -1 |
| `workers` | (N,2,10,3) float32 | 双方各自最多 10 个工人，按工人 id 数字后缀排序；每行 (pos, cargo, type)，type: 0=COLLECTOR, 1=PLANTER |
| `workers_mask` | (N,2,10) bool | workers 槽是否有效 |
| `trees` | (N,30,3) float32 | 全部树（双方合并）最多 30 棵，按 (owner, 树 id 数字后缀) 排序；每行 (pos, age, owner)，owner 为玩家下标 0/1 |
| `trees_mask` | (N,30) bool | trees 槽是否有效 |
| `worker_actions` | (N,10) int8 | 样本方工人动作标签，与 `workers[player]` 槽位对齐：0=stay, 1=UP, 2=RIGHT, 3=DOWN, 4=LEFT。动作 dict 里没出现的单位 = stay |
| `base_actions` | (N,2) int8 | 样本方基地动作标签，与 `bases[player]` 槽位对齐：0=none, 1=RECCOLLECTOR, 2=RECPLANTER |
| `game_file` | (N,) str | 来源回放文件（相对 replays/） |
| `strategy` / `opponent` | (N,) str | 样本方 / 对手策略名 |
| `score_self` / `score_opp` | (N,) float32 | 该局终局比分（样本方 / 对手），同一局每步相同 |
| `won` | (N,) bool | 样本方是否赢下该局 |

`players` 原始结构说明：observation 的 `players[i] = [cash, {center_id: pos},
{worker_id: [pos, cargo, type_str]}, {tree_id: [pos, age]}]`；树就存放在各自
owner 的 `players[i][3]` 里，本数据集的 `trees` 由此合并而来。

## 已知坑

- **可见碳 ≠ 全图碳**：`carbon` 是该玩家当步能看到的碳（采集员只见本格、种植员
  见四邻，其余为 0）。回放观测里另有 `full_carbon`（全图真值，渲染用），那是
  决策时拿不到的隐藏信息，**抽取时已刻意排除，不要拿它做特征**。
- **只有 player 0 的观测条目带共享字段**：steps[t][1] 的 observation 只有
  carbon/player/remainingOverageTime，`players`/`trees`/`step` 要从
  steps[t][0] 取（这两个字段在环境定义里是 shared，双方一致）。
- **observation['trees'] 不是树状态**：它映射 tree_id -> [最后接触的 worker_id,
  本步吸收量]，是吸收记账；树的位置/年龄/归属只能用 `players[i][3]`。
- **order 映射**：index.json 的 `order[player]` 给出该玩家对应的策略槽；
  `winner` 和 `scores` 都按槽（a/b）记，不是按玩家下标。搞反会把败方当胜方学。
- **首步记录口径**：idx=0 的观测是初始空局、动作恒为 {}；idx=1 的观测里
  双方已各有 2 名工人，但 idx=1 记录的招募动作并未在 idx=2 产生额外工人——
  疑似环境首步自动招募或站点记录口径如此（全 400 局一致、双方对称）。
  本数据集一律按同条目 (观测, 动作) 配对，与回放消费方的通行口径一致。
- **终局步样本被剔除**：status != 'ACTIVE' 的步（每局至多 1 步，即终局条目）
  记录的动作并未真正执行，已跳过（计数见 extract_bc.py 输出）。
- **短局存在**：个别对局因一方提前出局只有几十步，样本量按实际步数计。
- **超时局分数为 null**：快照里 5 局有一方 TIMEOUT、index 中该方 scores 为
  None（另有 8 局 winner='draw'）；头部统计与 top_players 入选已跳过分数为
  None 的对局，claude_family 不受影响（这 5 局均不含 claude 家族）。
- **工人工资帽**：工人槽上限 10 = 环境 workerLimit；树槽上限 30，本快照实际
  单局总树数远低于 30（溢出计数见脚本输出，应为 0）。
- **槽位排序是人为约定**：按单位 id 的数字后缀排序，只保证同一样本内特征与
  标签对齐；跨步同一槽不保证是同一单位（要用 pos 自行跟踪）。
"""

if __name__ == "__main__":
    main()

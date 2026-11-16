#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3 两阶段抽样器（B/C 确认件 13d3f3b4；快路径裁定 9651f30b §二/§三）。

口径：
  - 第一阶段（局）：π_g ＝ 1，300 局全入。
  - 第二阶段（局内）：B ＝ 每局上限 300 根；C ＝ 总上限 90,000 根。
    该局全部可辨根 π＝1 入样；余名额（B − 该局可辨根数）从该局
    非可辨根均匀无放回抽样，π_{r|g} ＝ 名额／该局非可辨根数
    （名额 ≥ 非可辨根数时全入、π＝1）。局根数 ≤ B 的局全入。
  - RNG：逐局独立流 seed_g ＝ BASE_SEED×10000 ＋ game（纯整数种子，
    与运行范围无关——先导局 0–9 的抽样与未来全量运行逐字一致）。
  - 分层（报告/敏感性拆分用）：按局根数 low ≤1128／mid 1129–1278／
    high ≥1279（冻结记录切点）。

范围纪律（裁定 9651f30b §三）：主测暂缓期间**不预生成全量补充件**。
本器按 --games 指定范围产出补充件（本次先导＝局 0–9）；--validate-all
只做聚合计数校验（全 300 局的可辨总数、拟入样总数上限），不落
全量逐根补充件。

产出（范围＝--games）：
  - sampling_supplement.json：meta（B/C/π_g/RNG 方案/冻结根单 md5/
    分层切点）＋逐局表＋逐根表（root_id、game、π_{r|g}、是否可辨、
    所属层）。
  - sampled_roots.jsonl：入样根全记录（冻结记录去 enum_provenance，
    加 in_sample/pi_r_given_g/stratum），供运行器消费。
校验（硬停）：冻结根单 md5 932226cd… 不合停；逐局根数与冻结册
不合停；总根 349,524 不合停；入样可辨根数与范围应有数不合停；
每局入样 >B 停；拟入样总数 >C 停（--validate-all 时对全集验）。
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN_LIST = os.path.expanduser(
    "~/workspace/search-group/s3_freeze/merged_out/S3_ROOT_LIST_FROZEN.json")
FROZEN_LIST_MD5 = "932226cdbe159a5e2ed67216e8c09b0b"
GAME_REGISTER = os.path.expanduser(
    "~/workspace/search-group/s3_freeze/merged_out/S3_GAME_REGISTER.json")
EXPECTED_TOTAL_ROOTS = 349524
EXPECTED_DECIDABLE_TOTAL = 778

B_PER_GAME = 300
C_TOTAL = 90000
BASE_SEED = 20261116
STRATA_CUTS = (1128, 1278)


def stratum_of(root_count: int) -> str:
    if root_count <= STRATA_CUTS[0]:
        return "low"
    if root_count <= STRATA_CUTS[1]:
        return "mid"
    return "high"


def md5_of_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 23), b""):
            h.update(chunk)
    return h.hexdigest()


def stream_roots(path: str):
    """流式读冻结根单（json.dump indent=1 形态）：逐根 yield 记录 dict。
    根对象以恰 2 空格缩进的 { 起、} 或 }, 止；嵌套对象缩进更深，
    不会误触边界。整载 404 MB 禁止。"""
    in_roots = False
    buf = None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not in_roots:
                if line.strip() == '"roots": [':
                    in_roots = True
                continue
            s = line.rstrip("\n")
            if buf is None:
                if s == "  {":
                    buf = [line]
                elif s.strip() in ("]",):
                    break
            else:
                buf.append(line)
                if s in ("  }", "  },"):
                    text = "".join(buf).rstrip()
                    if text.endswith(","):
                        text = text[:-1]
                    yield json.loads(text)
                    buf = None
    if buf is not None:
        raise RuntimeError("冻结根单流式解析未收尾（格式漂移）—— 停")


def sample_game(game: int, decidable: list, nondecidable: list):
    """一局抽样：返回 (入样记录列表 [(rec, pi)], 统计 dict)。"""
    d = len(decidable)
    n = len(nondecidable)
    quota = min(n, max(0, B_PER_GAME - d))
    if quota >= n:
        chosen = list(nondecidable)
        pi_nondec = 1.0
    else:
        rng = random.Random(BASE_SEED * 10000 + game)
        ids_sorted = sorted(r["root_id"] for r in nondecidable)
        chosen_ids = set(rng.sample(ids_sorted, quota))
        chosen = [r for r in nondecidable if r["root_id"] in chosen_ids]
        pi_nondec = quota / n if n else 1.0
    out = [(r, 1.0) for r in decidable] + [(r, pi_nondec) for r in chosen]
    stats = {"decidable": d, "nondecidable": n, "quota_nondec": quota,
             "sampled_nondec": len(chosen), "sampled_total": len(out),
             "pi_nondec": pi_nondec}
    return out, stats


def main():
    ap = argparse.ArgumentParser(description="S3 抽样器（B/C 口径）")
    ap.add_argument("--games", default="0-9",
                    help="产出补充件的局范围，如 0-9（先导）")
    ap.add_argument("--frozen", default=FROZEN_LIST)
    ap.add_argument("--register", default=GAME_REGISTER)
    ap.add_argument("--out-dir", default=_HERE)
    ap.add_argument("--validate-all", action="store_true",
                    help="对全 300 局只做聚合校验、不落全量补充件")
    args = ap.parse_args()
    g0, g1 = (int(x) for x in args.games.split("-"))
    scope = set(range(g0, g1 + 1))

    got_md5 = md5_of_file(args.frozen)
    if got_md5 != FROZEN_LIST_MD5:
        raise SystemExit(f"冻结根单 md5 不合：got {got_md5}，"
                         f"want {FROZEN_LIST_MD5} —— 停")
    print(f"冻结根单 md5 验合：{got_md5}", flush=True)

    with open(args.register, encoding="utf-8") as fh:
        reg_doc = json.load(fh)
    register = {e["game"]: e for e in reg_doc["register"]}
    register_md5 = md5_of_file(args.register)

    per_game_stats = {}
    supplement_roots = []
    sampled_records = []
    total_roots_seen = 0
    total_decidable_seen = 0
    would_sample_total_all = 0

    cur_game = None
    buf_dec, buf_non = [], []

    def finalize(game):
        nonlocal would_sample_total_all
        if game is None:
            return
        reg = register[game]
        rc = len(buf_dec) + len(buf_non)
        if rc != reg["root_count"]:
            raise SystemExit(
                f"局 {game} 根数与冻结册不合：流内 {rc} vs 册 "
                f"{reg['root_count']} —— 停")
        sampled, st = sample_game(game, buf_dec, buf_non)
        st.update({"game": game, "seed": reg["seed"], "root_count": rc,
                   "stratum": stratum_of(rc), "pi_g": 1.0})
        per_game_stats[game] = st
        would_sample_total_all += st["sampled_total"]
        if game in scope:
            for rec, pi in sampled:
                supplement_roots.append({
                    "root_id": rec["root_id"], "game": game,
                    "pi_r_given_g": pi,
                    "settlement_decidable": bool(
                        rec["settlement_decidable"]),
                    "stratum": st["stratum"]})
                out_rec = {k: v for k, v in rec.items()
                           if k != "enum_provenance"}
                out_rec["in_sample"] = True
                out_rec["pi_r_given_g"] = pi
                out_rec["stratum"] = st["stratum"]
                sampled_records.append(out_rec)

    for rec in stream_roots(args.frozen):
        total_roots_seen += 1
        if rec["settlement_decidable"]:
            total_decidable_seen += 1
        g = rec["game"]
        if cur_game is None:
            cur_game = g
        elif g != cur_game:
            finalize(cur_game)
            buf_dec, buf_non = [], []
            cur_game = g
            if g in per_game_stats:
                raise SystemExit(f"局 {g} 根记录非连续出现 —— 停")
        (buf_dec if rec["settlement_decidable"] else buf_non).append(rec)
    finalize(cur_game)

    if total_roots_seen != EXPECTED_TOTAL_ROOTS:
        raise SystemExit(f"总根数不合：{total_roots_seen} vs "
                         f"{EXPECTED_TOTAL_ROOTS} —— 停")
    if args.validate_all or True:
        # 全集聚合校验（不落全量补充件）：可辨总数与拟入样上限。
        if total_decidable_seen != EXPECTED_DECIDABLE_TOTAL:
            raise SystemExit(f"可辨根总数不合：{total_decidable_seen} vs "
                             f"{EXPECTED_DECIDABLE_TOTAL} —— 停")
        if would_sample_total_all > C_TOTAL:
            raise SystemExit(f"拟入样总数 {would_sample_total_all} 超 "
                             f"C={C_TOTAL} —— 停")
        for g, st in per_game_stats.items():
            if st["sampled_total"] > B_PER_GAME:
                raise SystemExit(f"局 {g} 入样 {st['sampled_total']} 超 "
                                 f"B={B_PER_GAME} —— 停")

    scope_decidable = sum(1 for r in supplement_roots
                          if r["settlement_decidable"])
    scope_games = [per_game_stats[g] for g in sorted(scope)]
    with open(os.path.abspath(__file__), "rb") as fh:
        sampler_md5 = hashlib.md5(fh.read()).hexdigest()
    supplement = {
        "meta": {
            "kind": "S3 sampling supplement (B/C, ruling 13d3f3b4)",
            "scope_games": [g0, g1],
            "scope_note": "先导范围（裁定 9651f30b §二）；全量补充件"
                          "暂缓不预生成（§三）。逐局独立 RNG 流使本"
                          "范围抽样与未来全量运行逐字一致。",
            "pi_g": 1.0, "B_per_game": B_PER_GAME, "C_total": C_TOTAL,
            "rng": {"scheme": "per-game random.Random(seed), sample "
                              "without replacement over root_id-sorted "
                              "non-decidable ids",
                    "base_seed": BASE_SEED,
                    "per_game_seed": "BASE_SEED*10000 + game"},
            "frozen_list": {"path": args.frozen, "md5": got_md5,
                            "total_roots": total_roots_seen},
            "game_register_md5": register_md5,
            "strata_cuts": {"low_max": STRATA_CUTS[0],
                            "mid_max": STRATA_CUTS[1],
                            "basis": "per-game root_count"},
            "sampler_md5": sampler_md5,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "validation_all_games": {
                "total_roots": total_roots_seen,
                "total_decidable": total_decidable_seen,
                "would_sample_total": would_sample_total_all,
                "within_C": would_sample_total_all <= C_TOTAL,
            },
        },
        "games": scope_games,
        "roots": supplement_roots,
        "totals": {"scope_games": len(scope),
                   "scope_sampled_roots": len(supplement_roots),
                   "scope_decidable_sampled": scope_decidable},
    }
    sup_path = os.path.join(args.out_dir, "sampling_supplement.json")
    with open(sup_path, "w", encoding="utf-8") as fh:
        json.dump(supplement, fh, ensure_ascii=False, indent=1)
    jsonl_path = os.path.join(args.out_dir, "sampled_roots.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for rec in sorted(sampled_records,
                          key=lambda r: (r["game"], r["step"], r["seat"],
                                         r["root_id"])):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def file_md5(p):
        return md5_of_file(p), os.path.getsize(p)

    sm, sb = file_md5(sup_path)
    jm, jb = file_md5(jsonl_path)
    print(f"范围局 {g0}–{g1}：入样根 {len(supplement_roots)}"
          f"（可辨 {scope_decidable} 全入）")
    print(f"全集校验：总根 {total_roots_seen}、可辨 {total_decidable_seen}、"
          f"拟入样总数 {would_sample_total_all} ≤ C {C_TOTAL} —— 合")
    print(f"产出：{sup_path}（md5 {sm}，{sb} B）")
    print(f"产出：{jsonl_path}（md5 {jm}，{jb} B）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

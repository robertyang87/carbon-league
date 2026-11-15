# PROVENANCE — realized U 构造来源记录（PPO-U 价值侧诊断）

## 续弈定义（预注册过签字面口径，原文转述）

真值＝源局**按已记录的实际对局打到底的终局**：源回放里那局真实
打完的终局结果，不重演、不换策略续打。每个根的 realized U 就是
其源局终局 margin 按席位定向后代入平台 U 公式的值——与根所在的
步无关，同一（局，席）的全部根共用一个终局 U。

## 来源字段链

1. 语料索引的终局比分 `scores = [score_玩家0, score_玩家1]`：
   ds_m3s2 构建产品的 meta games 表在册字段，经 Stage 1 的
   `build_roots.py` 原样 carried 进 `rootset32.json`
   （`games[].scores`，备制时 md5 `c0871c50…` 的 L20 副本）。
2. 定向：`margin_席 = scores[席] − scores[1−席]`。这正是
   `cnet2_build.build_game` 收到 `margins` 参数的构造方式
   （build_roots.py 传 `[s0−s1, s1−s0]`）；build 在有 raw 精确标签
   时还会断言声明 margin 与该 scores margin 一致（cnet2_build.py
   :340-356，L20 副本 46c8a538）。
3. 公式：`U = 0.5·W + 0.5·sigmoid(margin/1000)`，W = 1/0.5/0 对应
   margin > / = / < 0。与构建产品自家函数
   `cnet2_build._u_of_margin`（:168-174）逐字同——构建产品的
   `u_raw` 列对 normal 局就是该函数作用于 `margin_raw`；
   `realized_u.py` 在运行时 import 该函数逐（局，席）断言等值。

## 自检（realized_u.py 内硬断言，任一失败即停）

1. 同（局，席）全部根 U 恒同值；
2. 逐局 U(席0) + U(席1) = 1（float64，容差 1e-12），且
   sign(U0 − 0.5) 与 sign(scores[0] − scores[1]) 一致（平局则 U=0.5）；
3. 每（局，席）恰 99 根、步 ∈ [200, 298] 且不超过源回放的终局步；
4. 与 `cnet2_build._u_of_margin` 逐（局，席）等值。

残局距离（检验 B 用）：`dist_to_end = 源回放 steps 总数 − 1 − 根的步`，
steps 总数逐局从源回放文件（`/root/ppo_attrib/roots/*.json.gz`）
实读，逐局表随 `out/realized_u_provenance.json` 落盘。

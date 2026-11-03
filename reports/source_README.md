# Carbon League 本地工作区备忘

## 对测留痕规矩（2026-11-03 用户点名立规）

**今后所有本地对测局默认录 traces，不录的局不算数。**

- 理由：只存终分比分没法复盘——比分只说谁输了，不说输在哪一步、哪棵树、
  哪次碰撞。用户 2026-11-03 明确批评「不存对弈棋局怎么复盘」，此规矩即源于此。
- 做法：用 `record_trace.py` 跑对测（或给 eval 脚本接上同样的留痕逻辑），
  每局产出一个逐步轨迹文件：
  `traces/<版本>_vs_<对手>_seed<seed>_side<0|1>.json.gz`
  （含初始全图、每步双方动作、每步结束时双方全部单位/树/双方资金）。
- 索引：每局必须在 `traces/INDEX.md` 登记一行（版本、对手、seed、先后手、终分）。
- 复现性：本地策略均为确定性，`FastCarbon.reset(seed)` 同 seed 同图；
  补录历史局时终分必须与原 results jsonl 对得上（浮点末位差内），对不上要停下报告。
- 看回放：打开 `replay_viewer.html`，局目列表里切换（文件经 fetch 读
  `traces/` 同级目录；若用 file:// 直接打开被浏览器拦截，用页面里的
  文件选择框手动选 trace 文件）。

## 历史补录（2026-11-03）

- `diag_v1`（policy_v1 vs juzi）与 `eval_v01`（policy_v0_1 vs juzi）共 16 局
  （seeds 1000–1003 × 双方）此前由 `eval_v0_1.py` 只存了比分
  （`eval_v0_1_results.jsonl`），已用 `record_trace.py --games16` 按同 seed
  确定性重跑补录，终分与原记录逐局核对，结果见 `traces/INDEX.md`。

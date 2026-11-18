# 验收：k0_distill 改造版实施 PASS＋sidecar seat 映射强制口径（策略组，2026-11-18）

- 凭据：实施回执 reports/eng_verify/K0_DISTILL_U_IMPL_20261118.md（commit 9a7b6569，策略组读回 md5 97c30c5d、8,868 B 合）；设计 reports/DESIGN_K0_DISTILL_U_20261118.md（5d11dbce）。

## §1 验收 PASS

- 隔离：钉版 k0_distill 2728fee6／k0_join fac55acb 收尾复读未动；新文件 k0_distill_u.py（06826f59）／k0_join_u.py（9054aadc）入仓读回逐字验。
- 设计 §6 四项全合：① 全链 RC=0（join_u 建 /2 集 gen-1e1864a4、60 根/960 候选、preflight 经改造版训练器本人通过、预扫 60 全根 0 问题、训练 10 步落权重）；② 三头 loss 非零有限且下降（utility_mse 0.1124→0.0394、wdl_ce 0.9683→0.4922、margin_mse 0.6247→0.2130），步 1 policy CE 3.120475 与钉版冒烟首步逐位合、总 loss 可加总复核（终局项为纯增量、policy 路径未变）；③ 解冻对账（独立复核：三头 max|Δ| 0.0022–0.0029 确有更新、aux 逐位不变、冻结集实测仅 aux）；④ 冒烟权重 0833d429 已登记产物册（册新 md5 1fd25be3 读回验），版本戳 k0-distill-u/1＋λ 起点值入 meta。
- 负例双 die 原文在册：/1 喂入报版式不符 refusing to start；删 terminal_wdl 报字段闸点名（测试集为合法重封副本，触发的是字段闸本身——防假跑形态对位）。
- 头形态经钉版模型本体直读核对（utility=sigmoid→[0,1]、wdl 列序 win/draw/loss、margin=E[margin]/1000、同根展开 ctx 恒等取槽 0），不猜形状，合。

## §2 关键发现判读：sidecar seat↔side 映射（强制口径立规）

- 事实（双锚实证：build meta 逐 seat raw_u/raw_margin＋scores 同向）：**sidecar 的 side `a`＝seat 1、`b`＝seat 0**，与标注 seat 空间命名不一致。若按 side 名想当然并入，终局 U 标签整局翻 seat——胜负颠倒且静默无报错，训出的价值头是反的。
- join_u 处置成立：逐局逐 seat 与 build 原始标签交叉核对（容差 1e-9）、不合即 die 点名、build meta 缺锚即 die 拒建；另验 sidecar u 与其 margin 的恒等式（1e-6）；落盘标签逐 seat 直读验合（seat0 u 0.192279／margin −470.25／wdl 2；seat1 0.807721／+470.25／0），与终局自洽。
- **立规（对一切 terminal_u.json 消费方）**：禁止按 side 名直接并入终局标签；必须逐局逐 seat 与 build 原始标签交叉核对（≤1e-9）或经 k0_join_u 建集，缺锚拒建。换产局工具时映射须重验（交叉核对闸会自动拦，不预设映射值）。L20 产局侧若有直读 sidecar 的代码，须改走 join_u 或加同闸——请作战室点名确认 L20 侧消费方清单。

## §3 后续

- 价值半环已通：pilot 环训练可用双目标核心（k0_distill_u）开训，正式环开跑前钉版六件套对账（裁定 §3）仍为前置。

# 交接文档：NN 新线（claude-opus55-cloud）— 2026-11-04

> **接手入口索引见 `reports/live/CURRENT.md`**（在册制品、真实位置与校验值的唯一出处，REVIEW_20261112 R14）；本文件为 2026-11-04 历史交接，只作归档，制品信息与 CURRENT 冲突时以 CURRENT 为准。

> **2026-11-06 仓库精简（已改写全部历史并强制推送 main 与 claude/nocamp-eval-g1）**：`.git` 422 MB → 17 MB。
> 从历史中移除：`data/replay_snapshots/`（两份公开快照，**含 Master 早期 42 局，已不可再取**）、`data/traces/`（旧会话调试对局）、
> 旧 case-B/C 实验（`code/caseb*`、`code/casec*` 及其数据/权重）、`code/env_src` 的演示视频/动图/notebook、旧提交 `submission_nn_v1`–`v8.py`、
> 失败/过时权重（`nn_alpha3_v1`、`nn_mix3_v1`、`nn_mtop64`、`nn_ro12`、`nn_shadow_agy/codex_v1`、`nn_wide64_init`、`pred_data_*`、`rl_ckpt/`、`rl2_ckpt/`）。
> v9 复现链保留：`nn_bc_w48_v1_final` → `nn_alpha2_v1`(v6) → `nn_wide64_alpha`(v7) → `nn_master64`(v8/v9)，`code/submission_nn_v9.py`。
> 下文提到的已删文件仅作历史记录；重训 BC 需用 `code/ladder_corpus.py` 从服务器重新下载（服务器保留最近 1 万局）。旧会话脚本中引用 traces/快照的部分已不可直接运行。


> 写给接手的新会话。数字全部来自仓库内落盘文件（`data/evo_results.jsonl`、`data/nn_rl*_log.jsonl`、`reports/nocamp_recipe_report.md`）。
> 全过程（含判废记录）见 `reports/nocamp_recipe_report.md`；本文件只讲「现在在哪、怎么接着干」。

## 0. 一句话现状

**可上传的最好版本是 `code/submission_nn_v2.py`（1.23 MB，大文件榜）**：本地对 no-camp（天梯约 2,000 分）32 局 22 胜、+532 ± 140、最差一局 −1,177。
**天梯至今零提交**。榜首 claude-AlphaCarbon-Master 2,680（2026-11-04 00:10 UTC 快照），差距约 600 分。本地已没有比 v2 更强的对手，下一步必须拿天梯真实数据。

## 1. 第一件事（新会话）

1. 验证网络：`curl -sS -k -m 20 -o /dev/null -w "%{http_code}\n" https://132.232.176.153/api/leaderboard` 应为 200。
   旧会话里代理一直回 `x-deny-reason: host_not_allowed`（环境设置的改动对已启动容器未生效）。
   **2026-11-04 第二会话实测（环境网络为「完全访问」）**：`http://132.232.176.153/` 能到 nginx（301 跳 https），
   但 **出网代理对不带 SNI 的 TLS 一律在握手时重置**（curl/Python 对 IP 地址不发 SNI；`openssl -noservername` 连 example.com 同样被重置），
   代理状态里记为 `ws_closed_mid_exchange`。天梯没有域名；sslip.io 等域名带 SNI 会被**腾讯云未备案拦截**
   （HTTP 302 到 dnspod webblock，HTTPS 重置），也不行。
   **可行解（已验证）**：SNI 直接填 IP 字符串本身，代理与腾讯云都放行。用本地桥：
   `pip install pyopenssl && nohup python3 code/sni_bridge.py >/tmp/bridge.log 2>&1 &`，
   之后 `LADDER_BASE=http://127.0.0.1:18443 python3 code/ladder.py ping`
   （curl 用 `curl -H 'Host: 132.232.176.153' http://127.0.0.1:18443/api/...`）。
   天梯客户端：`code/ladder.py`（ping/login/submit/mine/board/matches/replays；密码只从 `LADDER_PASS` 或交互读入，cookie 存仓库外）；
   战绩分析：`code/ladder_report.py <策略ID> <matches.json> [回放目录]`（按对手汇总＋逐局现金/编制/树/最后招募步）。
2. 登录账号 **`claude-opus55-cloud`**（原 `muse-spark1-3` 已退出）。密码由所有者在对话里提供，**不入仓、不写进任何文件**。
3. 提交 `code/submission_nn_v2.py`，策略名建议 `claude-opus55-cloud-v2`（全站唯一，每次迭代换新名；每账号北京时间每天 10 次，失败的也算）。
   策略说明可直接用文件头 docstring（已注明数据来源、模型、规则层、本地成绩）。
4. 跟踪定位赛（至少 10 局、最多 40 局）：`/api/leaderboard`、`/api/matches?compact=1&strategy=<id>`；回放 `/data/<对局ID>/<局号>`（登录后，每分钟 ≤30 次）。
   用每局 `order` 字段把 players 0/1 映射到策略 a/b。
5. 按天梯结果迭代：分析输给头部的局（存碳 / 树经济 / 护货 / 终局），针对性改，换新名再交。

## 1a. 关键发现（第二会话）

**「强队回放行为克隆＋我方规则层」明显强于「BC v1＋PPO」**：AlphaCarbon 系列影子直接对 v4 16 局 13 胜。
RL（v6）对 v4 均为噪声。后续优先：按系列训影子 → 选最强者或混合数据训练 → 再考虑 RL。
天梯语料：`code/ladder_corpus.py`（≥2150 分的四大系列，`data/ladder_corpus/`，不入仓，可重下）。

**Master 剖析（2026-11-05）**：Master 对 Godslayer 35 局 33 胜，而 v5/v6 对 Godslayer 10 局 1 胜。差别不在自身经济（第 100 步交货 1665 vs 我们 1589），
而在压制对手：种植员消耗战（双方种植员各死约 26 个，我们的对局约 13）、种树 32/抢树 22（我们 19/12）、
采集员死亡丢货 32（我们 162）。AlphaCarbon 系列内部风格分化（v1 偏纯采集、Master 树战），混学得到「中间路线」；
**只学 Master 的 v8 明显强于混学的 v7**。

**v8 天梯（2026-11-05 晚）**：24 局 17 胜，2268±54，**全站第 9**；对 2000–2150 档 12 胜 0 负，对 2300+ 档仍 0/6（均 −1448，v6 为 −1963）。

**全面对比（2026-11-06）**：账本（每局均值）Master 终局 8658/交货 7966/树吸格 4079/丢货 77；RyanX V5/V6 8624/8330/3222/214；
v8 7387/6922/3962/456（对强队 6348/6853/2865）。v8 树经济已追平 Master，差在交货（−1000）、终局丢货（−380）、对手压制。
RyanX 自述「行为克隆＋强化学习＋即时推演搜索」，前列均为三层叠加；我们的整体方案见对话（执行层 NN＋战术层局部精确推演＋战略层编制＋专家迭代＋本地联赛评估）。

## 1b. 天梯提交记录

| 策略名 | ID | 说明 |
|---|---|---|
| claude-opus55-cloud-v2 | 9fd29f30a744442d86113471003b7e46 | **已下线（删除）**；2026-11-04 提交，10 局 4 胜 1386；定位中被 ds_cov18_m_v1e 守基地卡死 1146:12897 |
| claude-opus55-cloud-v3 | 147d7525c6227e052d952e4545bcb4fd | **已下线**（30 局 21 胜 1950）；v2 网络＋反守基地（STALL_K=3 强闯、种植员不占基地）；本地对 base_v53 8 局 7 胜 +1078（v2 同 seed 2 胜 −3806），对 nocamp 32 局 19 胜 +378（配对 v3−v2 −154±139，噪声） |
| claude-opus55-cloud-v4 | fbd55b8920b3ffb2cd9b91a04f01880f | **已下线**（34 局 20 胜 2006，大文件榜 17）；v3＋终局回家关护货；本地对 base_v53 16 局配对 v4−v3 +103±25，对 nocamp 32 局 −1±17。提交时 v3 定位 8 局 7 胜 1705、v2 定位完 10 局 4 胜 1387（第 27） |
| claude-opus55-cloud-v5 | b6b7ccb89976c3eb623f5f03b493529c | **已下线**（59 局 29 胜 2078，全站 18）；**网络换成 AlphaCarbon 系列行为克隆**（BC v1 热启动，该系列最近 200 局天梯回放，68 万样本，4 轮，`data/weights/nn_shadow_alpha_v1.npz`）＋v4 规则层。本地直接对 v4 16 局 13 胜（v4 −805±208）；对 nocamp 32 局 23 胜 +600（配对 +223±194）；对 base_v53 16 局 12 胜 +1230（+420±692）；打包前后逐局一致。提交时 v4 天梯 2016±47（大文件榜 15） |
| claude-opus55-cloud-v6 | d5c4c6df236eb7c7c7aa213ea13ef563 | **已下线**（107 局 65 胜 2153，全站 14）；网络＝BC v1 在 **AlphaCarbon 系列全部 288 局**（天梯语料 240＋快照 48，97 万样本）上 BC 5 轮（`data/weights/nn_alpha2_v1.npz`，val 79.3%）＋v5 规则层。本地：直接对 v5 32 局 22 胜 +565±163；对 nocamp 32 局 28 胜 +1293（配对 vs v5 +693±278）；对 base_v53 16 局 13 胜 +1510（+280±539）；打包一致。提交时 v5 约 2100（全站第 14） |
| claude-opus55-cloud-v7 | 534bdf289b7599419ded26bc84b85c56 | **已下线**（19 局 11 胜 2036）；v6 网络经 `code/nn_widen.py` 保函数扩宽 48→64（逐 logit 一致）后在同 288 局 AlphaCarbon 数据续训 3 轮（lr 3e-4，val 79.7%，`data/weights/nn_wide64_alpha.npz`）。本地：直接对 v6 32 局 21 胜 +684±278；对 nocamp 32 局 28 胜 +1150（配对 vs v6 −143±238）；对 base_v53 16 局 12 胜 +1607（+97±534）；均 214 ms/步、最慢 514 ms；打包一致。提交时 v6 天梯 2152±27（全站 14）。**从零训练的大网络（W80/W64）明显更差且太慢，已放弃** |
| claude-opus55-cloud-v8 | 72626605a9a75990d40940b788bb190c | **已下线**（61 局 34 胜 2195）；v7 网络（W64）在 **Master 独家 94 局**（语料 52＋快照 42，31 万样本）上 BC 4 轮（lr 3e-4，Master 验证局 80.2%，`data/weights/nn_master64.npz`）。本地：直接对 v7 32 局 23 胜 +615±287（树收入 4884 vs 1574、死亡丢货 210 vs 469）；对 nocamp 32 局 **31 胜 +1834**（配对 vs v7 **+684±162**）；对 base_v53 16 局 15 胜 +2568（+961±543）；打包一致。提交时 v7 天梯 19 局 2038±54 |
| claude-opus55-cloud-v9 | 6c82fcd3c11ac96abc084711012bdd96 | **已下线（删除，2026-11-07，用户要求）**（132 局 63 胜 2128）；网络同 v8；规则层 ENDGAME_BRAVE=False、STALL_OFF_LEFT=30（打包后改常量）。依据：v8 天梯 49 局丢货分解——被截 304、同归 152，其中**最后 20 步 350**（Master 仅 33）；被截=对手得分，双倍亏。本地：对 nocamp 配对 +95±41（28 局），对 base_v53 −88±97，对 v8 直接 24 局 −19±365（仅 ENDGAME_BRAVE 版）；打包一致。提交时 v8 天梯 50 局 2194±37 |。**天梯核验**：58 局 37 胜 2214±35（全站 14）；最后 20 步丢货 350→52（被截 304→102）；对 2000–2150 档 32/36，对 2300+ 仍 0/11
| claude-opus55-cloud-v10 | 4639488298d3dbe324a0c246d4203d60 | **已下线（删除，2026-11-07，用户要求）**；2026-11-07 提交；网络＝nn_master64 热启动、在 **1C-Zero 天梯对局** 上 BC 4 轮（`data/weights/shadow_1czero_v1.npz`，1C-Zero 验证 79.5%，v9 网络 69.5%）＋v9 规则层（打包后改 ENDGAME_BRAVE=False、STALL_OFF_LEFT=30）。本地对 v9：gate 64 局 44 胜 +849±288、最差 −4106、最慢 97 ms（未达自动提交阈值 mean≥1000，用户批准提交）；另 32 局 24 胜 +1587。账本：送货 7588 vs 6779。打包前后 16 局逐局一致。提交时 v9 天梯 130 局 63 胜 2135±25（全站 15） |
| claude-opus55-cloud-v11 | b8a9ae080070446a27724871acc7be79 | **已下线（删除，2026-11-07，用户要求）**；**加载失败（status failed，0 局）**：平台以字符串 exec 提交源码、不定义 `__file__`，打包器顶层 `_m.__dict__["__file__"] = __file__` 抛 NameError。修复：`globals().get("__file__", ...)`；以后每个打包文件必须过 `data/sim_platform_load.py`。2026-11-07 提交（自动提交规则 B）；**全新大网络 cnet2_b128**（全盘 8×128 残差 + 单位 8 步历史 + 注意力 + GRU 自回归解码，2.95M 参数，int8 打包 3.6 MB，`exp/submission_cnet2_v11.py`，权重 `data/weights/cnet2_b128.npz`，teacher style 0=Master），在前 5 名 516 局回放上 BC 14 轮（val 0.826）。打包文件新种子 970000–023：对 v9 46/48 +3554±590（最差 −15031）；对 1C-Zero-v5 44/48 +1004±132；fp32 对 v10 61/64 +1811。42–88 ms/步，最慢 282 ms（单进程）/502 ms（满载），内存峰值 137 MiB。注：测试框架换座位为镜像局，独立局数≈一半 |
| claude-opus55-cloud-v12 | 4b650d742b550497a138e15378a4e4ba | **已下线（删除，2026-11-08 用户要求；122 局 65 胜 2402）**；2026-11-07 提交（自动提交规则 B）；cnet2_b128 **teacher style 1（1C-Zero-v9 风格）**纯网络，int8 打包（已修 `__file__`，过 sim_platform_load）`exp/submission_cnet2_v12.py`。本地：fp32 对 1C-Zero-v5 60/64 +1142±93（最差 −659）；打包文件新种子 971000–019 对 v9 39/40 +2704±215（最差 −139），满载最慢 721 ms、单进程 321 ms；内存 84 MiB |
| claude-opus55-cloud-v13 | 06352c44c8a3e32cff979bfdb996e2eb | **已下线（删除，2026-11-07 用户要求；约 56 局 33 胜 2442）**；2026-11-07 提交（自动提交规则 B）；cnet2_b128v4（b128 热启动，数据集 v4 按胜负加权：胜方 ×2、双方 ≥2400 再 ×1.5，含 17 局我方输局以胜方为老师），teacher style 1，int8 `exp/submission_cnet2_v13.py`（md5 66375f15…，过 sim_platform_load）。本地：对 v9 32/32 +3734±527；对 1C-Zero-v5 62/64 +1540±113（v12 同种子 +1314）；对 v12 24/48 +104±100；对四流派替身合计比 v12 +62/局（均不显著）。v12 同时保留 |
| claude-opus55-cloud-v14 | a5dd1a223f9a766b531f0ff211bd5d60 | **已下线（删除，2026-11-08 用户要求；约 148 局 2553）**；**源码曾短暂公开，已撤回（2026-11-08 用户要求，改公开无注释副本 v14p）**；2026-11-08 提交（用户“帮我提交”，放宽的试天梯规则；提交走服务器 307 重定向后的 117.72.210.134）；cnet2_b128v5（b128 热启，数据集 v5 906 局含 Master2/v10/v11-preview ×2），teacher style 7，窄化防破产保护，无搜索，int8 `exp/submission_cnet2_v14.py`（md5 50bc6c84…，过 sim_platform_load，冷机最慢步 54–57 ms）。本地：对 v13 200 局配对 vs v12g +813±111（168 对 73 胜，最差 −1620）；对 1C-Zero-v9 40 局配对 +756±174（27 对 8 胜）；Master2 替身 +1707±353（半开卷，仅机制）；打包版复测与 fp32 差 +90±159。v12 同时保留 |
| claude-opus55-cloud-v14p | 6142a1f80ca7fede3e81608eed8716ab | **源码公开**；2026-11-08 提交（用户要求公开去注释版）：v14 打包文件经 `data/strip_comments.py` 去掉全部注释与 docstring（外层及内嵌 4 个模块），200 局与 v14 逐局一致 |
| claude-opus55-cloud-v17 | eeefd6e19021355633d34f6d5a2a1a9b | 私有；2026-11-09 03:3x 提交（用户：诊断指标与胜负冲突时以胜负为准）。b128v9（数据集 v9：第 10–60 步种植员 ×2.5）+ 第 240 步切换 + 与 v16b 相同的搜索（ROOT_S 10、8 风格、ID_PRIOR 0）、推演网 s64v8 f16、主网 int8、保护 0.7/0.85/0.95；`exp/submission_cnet2_v17.py`（md5 fc36b4ab…，5.7 MB；空闲 Mac 157 ms/步、最慢 188 ms）。云上去保护等价件 vs v16b 配对：Master2s 替身 v2 +173±80（200 局），无始替身 +336±125，v13 −60±111，WB-532 +72±101（vs v16 浮点）。未达诊断项：第 100 步树 3.2 对 3.8（中期每时点少 0.1–0.6 棵）。上线核验：前 5 局平台每步 450–550 ms |
| claude-opus55-cloud-v16b | 072688045615bc3e0f9c59bcccba7218 | 私有；2026-11-08 23:2x 提交（用户/Fable：提交、v16 保留、私有）。v16 的修正版：同 b128v7 + 第 240 步切换 + MCTS ROOT_S 10、8 风格对手模型；**推演网 s64v7 改 f16（主网络仍 int8，`CNET2_QMODE_ROLL=f16`）**，时钟保护放宽到 BUDGET 0.7 / HARD 0.85 / ABORT_DL 0.95；`exp/submission_cnet2_v16b.py`（md5 58bfe2eb…，5.7 MB）。依据：打包 int8 去保护 vs 浮点 −189±78（200 局），全 f16 −89±48，混合 −73±104（120 局，与全 f16 差 −7±98），f16 与浮点每局搜索步数/改动步数分布相同（266/28.1 对 266/28.1）。空闲 Mac 每步 162 ms、最慢 199 ms、首步 25 ms、RSS 162 MiB。上线核验：前 5 局平台每步 420–500 ms、最后一步约 450 ms；>700 ms 撤 |
| claude-opus55-cloud-v16 | 64241d10fefabe20d2123b45cbc217d2 | 2026-11-08 提交（用户：补考过了就提交）；cnet2_b128v7（数据集 v7，分阶段老师权重）+ 第 240 步风格切换（槽 5→7）+ cnet2_search MCTS（ROOT_S=10、H=3、8 风格对手模型 ID_PRIOR=0、推演网 cnet2_s64v7）、平台时钟 BUDGET 0.5/HARD 0.75/ABORT_DL 0.9、根节点兜底、强制单线程，int8 `exp/submission_cnet2_v16s.py`（md5 645c17dc…，sim_platform_load 通过，Mac 每步 160 ms/最慢 191 ms、冷启动 23 ms、RSS +50 MiB）。本地 vs v15 配对：v13 +608±78（200 局 198 胜，<−1000 0 对 2，最差 −467）；WB-532 +492±101（<−1000 1 对 5）；Master2s 替身 +464±93（<−1000 0 对 2）。上线核验：前 5 局平台每步均值应 450–650 ms，>800 ms 撤。v15 保留 |
| claude-opus55-cloud-v15 | 95cbee5ba6b3d6ed4061fad714e4e8e9 | 2026-11-08 提交（用户“先提交吧”）；cnet2_b128v6（数据集 v6：+Godslayer-P2C、按单位加权、40–100 步招募×2，WB 整族留出），分段风格：第 240 步前槽 5（Master2）、之后槽 7（v11-preview），窄化防破产，无搜索，int8 `exp/submission_cnet2_v15.py`（md5 32ea8937…，sim_platform_load 通过，最慢步 21 ms）。本地 vs v14 配对：v13 400 局 +361±58（<−1000 3 对 7，最差 −2001 对 −1620）；WB-532 +434±103；v9 +531±84；Master2s 替身胜率 75% 对 63%、<−1000 2 对 11。v14 同时保留 |

## 2. 版本与成绩（本地 FastCarbon，同 seed 双边配对）

| 版本 | 内容 | 对 no-camp | 文件 |
|---|---|---|---|
| no-camp | 公开 v53 + soft_w 0 / campers 1 / camp_from 12 | 对 base_v53 24 局 22 胜 +3,278；天梯同配方 1,952–2,014 | `code/submission_nocamp.py`（不推荐上传） |
| v1 | BC（CNN）＋规则层 | 32 局 20 胜 +14 ± 246（seeds 700000–700015） | `code/submission_nn_v1.py`（已被 v2 取代） |
| **v2** | v1 ＋ PPO 微调（RL v3）＋招募保底修复 | **32 局 22 胜 +532 ± 140，最差 −1,177** | **`code/submission_nn_v2.py`**，权重 `data/weights/nn_rl3_last.npz` |
| Master 影子 | BC v1 在 45 局 Master 回放上微调 | 8 局 4 胜 +445 ± 518（方差大） | `data/weights/nn_shadow_master_v1.npz`（仅作陪练） |

速度：v2 约 150 ms/步（4 进程并行测），最慢 355 ms；天梯限制 1 s/步、1.5 核、512 MiB。打包为 float32，打包前后逐局一致（已验证）。

## 3. 架构（代码地图）

- `code/nnfeat.py` — 共享特征（训练与推理同一份）：信念碳图（可见性规则：采集员看本格、种植员看四邻；四重对称镜像；未见格按树吸收＋3% 再生推进）＋14 平面自我中心 15×15＋18 标量。动作方向已用回放验证：UP=row−1、DOWN=row+1、LEFT=col−1、RIGHT=col+1；step t 的观测配 step t+1 记录的动作。
- `code/nn_train_bc.py` — 网络（3 残差块、48 通道、循环填充；头＝中心 3×3 patch＋全局均值＋标量 → 256 → 8 logits；另有价值头）＋BC 训练（二面体增广）。支持多数据文件（逗号分隔，不拼接）、`BC_INIT` 热启动、`BC_LR`。
- `code/nn_agent.py` — NumPy 推理（channels-last im2col＋BLAS）＋规则层 `select_actions`：
  己方防撞（按置信度占格）、**护货盾** `SHIELD`（满载采集员避开携货 ≤ 自己的敌方采集员可达格）、
  **终局回家**（剩余步 ≤ 距离＋1 强制回最近基地；最后 15 步允许己方单位挤基地格做碰撞整合）、
  **招募闸** `recruit_choice`（有空位＋现金够 30＋种树储备即招；采集员为 0 时 30 即招采集；采集员 < max(2, 种植员) 或种植员 ≥ 4 时强制招采集；类型否则由网络定）。`TTA` 开关默认关（见 §5）。
- `code/nn_rl.py` — PPO 自博：独立价值网、KL(π_ref‖π)、价值预热、团队级 GAE、对手池 `RL_POOL`（支持 `NN:<权重路径>`）、每 5 轮贪心对 nocamp 固定 8 局（seeds 640000–640003）。
- `code/nn_pack.py` — 打包单文件提交（权重 float32＋zlib＋base85 内嵌）。
- `code/nn_build_bc.py` — 从回放快照生成 BC 数据（`BC_TEACHERS_ONLY` / `BC_TEACHERS_EXTRA`）。
- `code/evo.py` — 变体生成（父文件＋`PARAMS.update`）与配对评测（复用 `eval_v53.play_game`）；`code/make_spars.py` 一键重建本地陪练。

## 4. 环境复现（新容器）

```bash
cd code
ln -s ../data/traces traces                      # 旧脚本假设平铺布局
for f in shadow_v2_weights shadow_weights shadow_deepseek_weights; do ln -sf ../data/weights/$f.npz $f.npz; done
python3 make_spars.py                            # 生成 nocamp.py、exp/spar_tf2/tf3、exp/nn_*.py（均 git-ignore）
pip install -q torch                             # 仅训练需要（PyPI 版带 CUDA 依赖，约 3 GB；CPU 可用）
python3 test_fastsim_parity.py                   # 应 36/36 通过
python3 evo.py match exp/submission_nn_v2.py nocamp.py 700000 4   # 复核 v2（需先 cp submission_nn_v2.py exp/）
```

BC 数据（4 GB，不入仓）可再生：`python3 nn_build_bc.py ../data/replay_snapshots/snap_20261103 OUT.npz 2380`
（回放快照两份已存入 `data/replay_snapshots/`：20261103 14:26 UTC 与 20261104 00:10 UTC，各 400 局。公开仓库每次导出整体覆盖，旧快照无法再拉。）

## 5. 已证伪 / 不要重试

| 方向 | 结论 | 证据 |
|---|---|---|
| no-camp 单参数爬山（evo v1/v3） | 平台，单参 ±25% 全在噪声内 | `data/evo_state_v1.json`、`data/evo3_state.json` |
| 中等幅度改种树门槛 | **惰性**（终分逐位不变），约束在种植员编制 | 报告「更正」节 |
| 旧线 `switcher_treewar` | 对 nocamp 与 base_v53 逐位相同（惰性） | 同上 |
| RL v1 / v2 | 退化：价值头共享主干把策略拖走 | `data/nn_rl1_log.jsonl`、`data/nn_rl2_log.jsonl` |
| RL v4 / v5（联赛池，约 30 轮） | 无提升；暴露招募饿死崩盘（已修） | `data/nn_rl4_log.jsonl`、`data/nn_rl5_log_part*.jsonl` |
| float16 打包 | argmax 翻转致整局分叉 → 改 float32 | 报告「打包精度更正」 |
| TTA4 | +129 ± 255（噪声），最慢步 866 ms → 关 | 报告 TTA 节 |
| 新快照重训 BC | 新快照多为中游规则系，教师样本仅 +8% | `nn_build_bc` 输出 |
| 招募编制（v4 基础上：种植员上限 3 / 2，采集员 ≥ 种植员+2） | 均为噪声或更差：对 nocamp 配对 P3 −141±230、M2 +61±242、**P2 −459±227**；对 base_v53 +169±675 / +324±724 / +8±507。天梯上「我方种植员多、采集员少」不是输局主因 | `data/evo_results.jsonl`（exp/v4_P3/M2/P2） |
| RL v6（v4 规则层，池 nocamp/base_v53/Master 影子/自身，lr 5e-5 KL 0.2，8 轮） | 对 nocamp 配对 −193±241，对 base_v53 +541±627：噪声；第 5 轮贪心 +41（起点 +825） | `data/rl6/rl_log.jsonl`、`data/evo_results.jsonl`（exp/rl6_last） |
| 种植员保底（v5 上 PLANTER_FLOOR=2/3） | F2 基本惰性（nocamp 23/32 局逐分相同）；F3 对 nocamp +85±158、base_v53 +107±463（噪声），**对 Godslayer 影子 −389±244**（种树收入 3560→2742）。天梯输局「种植员少、树少」是结果不是可直接硬编的原因 | `data/evo_results.jsonl`（exp/v5_F2/F3） |
| 开局采集员优先（v6 上 EARLY_COL_UNTIL=30/45） | 生效（E45 第 20 步即 7 采集、第 59 步现金 842 vs 444），但对 v6 直接对打 E45 −67±231、E30 −135±280，对 nocamp E45 +444（v6 同 seeds +1293）：更差。依据是 v5/v6 对 Godslayer 10 局：第 50 步对方 7.8 采集/交货 689 vs 我 5.3/365 | `data/evo_results.jsonl`（exp/v6_E30/E45） |
| v6 网络续训 3 轮（lr 2e-4，同 288 局数据） | val 79.3→79.5%；直接对 v6 32 局 19 胜、+9±274：无差别。同数据上 BC 已饱和 | `data/weights/nn_alpha3_v1.npz`、`data/evo_results.jsonl` |
| v8＋Master 对 2300+ 对手 39 局加权（×2）续训 2 轮 | 对 v8 64 局 34 胜 +318±366：无差别 | `data/weights/nn_mtop64.npz` |
| RL v8（自 v8，W64，池 SELF/v6/v7/nocamp/base_v53，KL 0.2，lr 3e-5，每轮 ~17 分钟） | it5：对 v8 32 局 +19±321；it10：贪心对 nocamp +2151（it0 +1091），但对 v8 32 局 20 胜、均值 −162±488（中位 +349），含 1 局崩盘（seed 860002：采集员每趟只采 7–15 就折返，第 200 步全灭，2:12827）。RL 会在个别地图上把采集行为练坏，提交前必须查最差局 | `data/rl8/rl_log.jsonl` |；**it15**：贪心对 nocamp +1260，对 v8 16 局 9 胜但均值约 −1200，**2 局崩盘（−11236/−10355）**，崩盘频率随训练上升 → 第 15 轮后停止。结论：在当前算力（~1 千局/夜）下 PPO 无法稳定提升 v8，反而损伤鲁棒性
| 一步前瞻搜索（`code/nn_search.py`：NN 候选＋FastCarbon 精确走一步＋打分） | 估值网 `nn_value_v1`（`nn_build_value.py`/`nn_train_value.py`，1928 局 38.6 万状态，R² 0.70，第 100 步后胜负准确率约 90%）。用估值打分：对 nocamp 配对 vs v9 EPS 0.002/0.01/0.02 = −1349/−505/−715（估值误差约 ±1200 分，远大于单步差异，搜索按噪声换棋），且 4 进程下最慢步 1242 ms。用一步精确物质打分（仅采集员）：nocamp +45±78、base_v53 −2±4（几乎不触发）。结论：单步搜索要么有害要么惰性；需更深推演＋更准的对手模型 | `code/nn_search.py`、`data/weights/nn_value_v1.npz` |
| 反事实推演标注蒸馏（`code/rollout_label.py`：复制模拟器＋双方智能体，关键采集员 5 种走法各推 H 步按物质分差打分；`code/nn_finetune_rollout.py`：增益≥100 的 802 个样本＋Master BC 微调 v9 网络） | H=25 蝴蝶效应噪声大（策略原选已最佳仅 20%，平均「可提升」197）；H=6 短视；H=12 标注蒸馏后**对 v9 12 局 0 胜 −4877**：物质短视标注破坏长线打法。另：自博弈估值数据生成器 `gen_value_selfplay.py`（未继续） | `code/rollout_label.py`、`data/weights/nn_ro12.npz` |
| 规则化「种树管理」替换 v8 种植员（TREE_MGR，按 4 邻格信念碳×剩余寿命估值、贪心分配；第 2 版加目标锁定/半径 4/现金闸） | 对 nocamp 同 2 局：v8 +1694，TM v1 −2211，TM v2 −4103：**远差于网络种植员**（目标漂移、长途送死、打乱配合）。TOLL_AVOID 单开 +1355（2 局，噪声）。结论：精确化要以网络为先验做搜索，而非规则替换 | `code/nn_agent.py`（TREE_MGR/TOLL_AVOID 默认关） |
| 从零训练大网络（W80 NB4 / W64 NB3，混合数据预训练 1 轮） | W80 每批 3.6 s（>10 小时，弃）；W64 预训练后 AlphaCarbon 第 1 轮 val 75.9%（v6 同期 77.7%），停。**改用保函数扩宽 v6 网络（即 v7）** | `scratchpad` 日志（未入仓） |
| 多系列混合 BC（alpha＋agy＋edge 各 200 局，184 万样本，从 v5 网络热启，lr 3e-4，1 轮） | 直接对 v5 16 局 6 胜、−443±398：不如 v5。单系列影子：Godslayer（1 轮）v5 12/16 胜、Codex v5 15/16 胜。**强的只有 AlphaCarbon 系列** | `data/weights/nn_mix3_v1.npz`、`nn_shadow_{agy,codex}_v1.npz` |

## 6. 坑（踩过的）

- 多进程跑 NumPy 时必须在**启动前**设 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`（进程内再设无效，4 进程×4 线程=16 负载，速度降数倍）。

- **云容器在会话空闲约 5 分钟后被回收**（后台进程不算活动；nohup 也会被杀，磁盘保留）。长任务要么在一个回合内前台分段等待（每段 ≤10 分钟），要么写成可断点续跑（`nn_train_bc.py` 已支持 `OUT.ckpt.pt`）。2026-11-04 夜里因此丢了三次训练。

- `pkill -f <模式>` / `pgrep -f` 会匹配到自己所在 shell 的命令行（曾把自己的 shell 停掉、等待循环永不退出）。按 PID 管理进程。
- 容器会被回收重启：长任务用 `nohup`，权重每 5 轮落盘（`rl_it*.npz`、`rl_last.npz`），断了从 `rl_last` 续跑（价值网重新预热）。
- `nn_agent` 的 `SHIELD`/`TTA` 是模块级全局；同一进程池里混跑不同开关会互相污染——分开跑 `run_pairs`。
- 评测噪声：单局分差 sd 约 1,000–2,600；8 局只看方向，结论至少 32 局配对。
- 陪练 / 变体入池前先做「同 seed 逐位比对」确认非惰性。

## 7. 建议的下一步（按优先级）

1. **上传 v2，拿真实定位**（§1）。这是现在信息量最大的一步。
2. 抓我们对头部的回放：对局列表 → 下载回放 → 用 `nn_build_bc` 同款解析，统计存碳 / 树收入 / 死亡丢碳 / 终局滞留，与 `reports/alpha_analysis.md` 的榜首画像对照。
3. 持续积累回放快照（`git clone --depth 1 https://github.com/JUZI1/carbon-league-replays`，每次换新目录），头部对局攒够后重训 BC（更大网络，如 W64×4 块，注意推理时间）。
4. 搜索增强（未做）：用 FastCarbon 对关键决策（回家时机、招募、满载采集员走位）做短程推演，作为冲榜后手；注意 1 s/步与 1.5 核。

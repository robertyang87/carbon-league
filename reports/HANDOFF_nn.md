# 交接文档：NN 新线（claude-opus55-cloud）— 2026-11-04

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

## 1b. 天梯提交记录

| 策略名 | ID | 说明 |
|---|---|---|
| claude-opus55-cloud-v2 | 9fd29f30a744442d86113471003b7e46 | **已下线（删除）**；2026-11-04 提交，10 局 4 胜 1386；定位中被 ds_cov18_m_v1e 守基地卡死 1146:12897 |
| claude-opus55-cloud-v3 | 147d7525c6227e052d952e4545bcb4fd | **已下线**（30 局 21 胜 1950）；v2 网络＋反守基地（STALL_K=3 强闯、种植员不占基地）；本地对 base_v53 8 局 7 胜 +1078（v2 同 seed 2 胜 −3806），对 nocamp 32 局 19 胜 +378（配对 v3−v2 −154±139，噪声） |
| claude-opus55-cloud-v4 | fbd55b8920b3ffb2cd9b91a04f01880f | **已下线**（34 局 20 胜 2006，大文件榜 17）；v3＋终局回家关护货；本地对 base_v53 16 局配对 v4−v3 +103±25，对 nocamp 32 局 −1±17。提交时 v3 定位 8 局 7 胜 1705、v2 定位完 10 局 4 胜 1387（第 27） |
| claude-opus55-cloud-v5 | b6b7ccb89976c3eb623f5f03b493529c | **网络换成 AlphaCarbon 系列行为克隆**（BC v1 热启动，该系列最近 200 局天梯回放，68 万样本，4 轮，`data/weights/nn_shadow_alpha_v1.npz`）＋v4 规则层。本地直接对 v4 16 局 13 胜（v4 −805±208）；对 nocamp 32 局 23 胜 +600（配对 +223±194）；对 base_v53 16 局 12 胜 +1230（+420±692）；打包前后逐局一致。提交时 v4 天梯 2016±47（大文件榜 15） |
| claude-opus55-cloud-v6 | d5c4c6df236eb7c7c7aa213ea13ef563 | 网络＝BC v1 在 **AlphaCarbon 系列全部 288 局**（天梯语料 240＋快照 48，97 万样本）上 BC 5 轮（`data/weights/nn_alpha2_v1.npz`，val 79.3%）＋v5 规则层。本地：直接对 v5 32 局 22 胜 +565±163；对 nocamp 32 局 28 胜 +1293（配对 vs v5 +693±278）；对 base_v53 16 局 13 胜 +1510（+280±539）；打包一致。提交时 v5 约 2100（全站第 14） |

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
| 多系列混合 BC（alpha＋agy＋edge 各 200 局，184 万样本，从 v5 网络热启，lr 3e-4，1 轮） | 直接对 v5 16 局 6 胜、−443±398：不如 v5。单系列影子：Godslayer（1 轮）v5 12/16 胜、Codex v5 15/16 胜。**强的只有 AlphaCarbon 系列** | `data/weights/nn_mix3_v1.npz`、`nn_shadow_{agy,codex}_v1.npz` |

## 6. 坑（踩过的）

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

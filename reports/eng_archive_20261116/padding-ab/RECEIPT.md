# Padding 同局对拍回执（工程组，2026-11-14）

## 结论
定谳闭环成立：同局、同代码（修复前 gate），仅切换 padding 根包含与否，共有真实根的
per-root 数值逐位一致（2/2），聚合差异全部由 A 独有 padding 行自身的值解释
（A 限共有根聚合 ≡ B 聚合，逐项相等）。补救口径（按 step < first_terminal 过滤后
重算 du_mean/σ̂）有效。

附带形态修正：padding 根是否为零行取决于终局类型——
- extinction 终局：root 状态在 sim 内已灭绝，continue_game 首步短路，
  teacher_calls=0、du≡0（代码＋状态诊断确立，见下 6a05e1a6 局）；
- status 终局（平台 TIMEOUT/判负）：sim 状态仍存活，supra-FT 根是真实续局，
  teacher_calls>0、du≠0（本局实测两根：-0.00541 / +0.02127），
  对聚合的污染方向任意，不是向零稀释。

## 标的局
- 真实天梯局 `899d25b15b488b25ae538a5afc5fbc21-1`（harvest 原文件 md5 3dc53b5d…，
  189 帧，真实终局 frame 188，cause=status：seat1 TIMEOUT 判负）。
- 本地 harvest 文件一律截断于终局帧（抽样 240 局 len=FT+1，无一带 padding），
  故 padded 版 = 真实帧 0..188 + 终局帧副本 ×111（→300 帧），与所有者 K0 反例的
  padding 构造同构。真实根的计算从不读自身 step 后的帧、续局从不读回放，
  padding 尾内容不可能经数据通道影响真实根；所验通道为根集构成
  （同进程缓存/RNG/共享状态）。
- FT 由 code/k0_aux_labels.py 的 first_terminal 口径求得：188（status）。

## 代码钉（批一 39d71cce 之父 d0370c6a 快照，本地副本在 src/）
- stage1_teacher_gate.py（修复前）aae25fda…
- stage0_teacher.py c6ade5ce… / cnet2_feat.py 2a45b08b… / cnet2_np.py d8c15ad4…
- cnet2_search.py 94f7fb64… / eval_v53.py 4426e12d… / fastsim.py aefe6c83…
- env_src carbon.json cd55531b…
- FT 口径：k0_aux_labels.py（main）f3b40b3b…（+ aux_labels 6cf947aa…）
- 权重：正典 heads v2 `heads-v2-diff/dgx.npz` 06a03d9c…（187 键）
- teacher：真实 stage0 v3，belief=sample，teacher_style=2；
  K/M/H=(2,1,1)（降格，理由见局限）。

## 根集
旧采样器（LAYERS_DU + opening_share）在 6 个 scene 种子（995300–995305）下共抽
17 根（真实 11、padding 6）。成本所限完成子集：
- 共有真实根（A/B 对拍）：(165,1)、(187,1)
- A 独有 padding 根：(216,0)、(219,0)（(219,0) 在 A2/A3 两进程各跑一次）

## 逐根比对（du / u_guided / u_greedy / teacher_calls 全字段）
| 根 | A du | B du | calls A/B | 判定 |
|---|---|---|---|---|
| (165,1) | -0.007752127644175905 | -0.007752127644175905 | 134/134 | 逐位一致 |
| (187,1) | -0.023520302844535948 | -0.023520302844535948 | 112/112 | 逐位一致 |
| (219,0) A2 vs A3 | +0.021270650262398938 | +0.021270650262398938 | 80/80 | 跨进程逐位一致 |

## A 独有 padding 行形态（构造 padding 下的实测值）
| 根 | du | u_guided | u_greedy | teacher_calls |
|---|---|---|---|---|
| (216,0) | -0.005414671206711907 | 0.8410457274982441 | 0.8464603987049560 | 83 |
| (219,0) | +0.021270650262398938 | 0.8746309856451255 | 0.8533603353827266 | 80 |

## 聚合对照（单局；game_mean = root mean，σ 为根级 σ_root，σ_game 在 n_games=1 不可估）
| 集合 | n_rows | 零行 | mean du | σ_root |
|---|---|---|---|---|
| A 全根 | 4 | 0 | -0.0038541128582562055 | 0.018579940904782825 |
| A 限共有真实根 | 2 | 0 | -0.015636215244355926 | 0.011149783611112135 |
| B（仅真实根） | 2 | 0 | -0.015636215244355926 | 0.011149783611112135 |

## extinction 形态旁证
局 `6a05e1a60a949c4ee303ff2e2e964e0e-1`（127 帧，FT 126）：extract_scene 诊断
step 126 处 sim 状态 extinct1=True（step 125 尚 False）——此类局的 supra-FT 根
按 continue_game 首步短路必为 teacher_calls=0、du≡0（未实际产行：其文件截断、
无 padded 版；结论由代码路径＋状态诊断确立）。

## 局限
1. K/M/H 降格为 (2,1,1)：本 VM 实测 evaluate_root 单价 (2,1,1)≈4.9s、
   (4,2,1)≈18s（cnet2_np.conv 主导），正典 (16,8,2) 单根需数小时不可行。
   逐根纯度机制（逐步种子 seed_base=cont+step_no、CachePolicy 全签名备忘）
   与 K/M/H 无关；行值仍为真实 teacher 输出。正典配置同局对拍可在 Studio
   补跑（单 call 7.53s ⇒ 单根约 15–17 分钟）。
2. 根集子集：(40,120) 层与 (120,200) 层其余 9 根真实根未跑（每根 110–160 calls）；
   已跑两根一在 FT 前一帧、一在 FT 前 23 帧，均为污染边界最贴身位置。
3. 单局实验：聚合层（多局 σ̂）结论仍归策略组的 join 重算，本实验只闭逐根命题
   与单局聚合恒等。
4. padding 尾为构造（终局帧副本）；原生 padded pilot 文件在 Studio/DGX，
   其 supra-FT 行的具体数值随平台 post-game 帧形态而变，但 status 终局下
   「非零行」的定性结论对任何 padding 形态成立（sim 中无人被淘汰）。
5. A2 进程曾被 runtime 重启杀掉（(219,0) 已落盘），以 A3 同批重跑补齐，
   两进程同根逐位一致已计入证据。

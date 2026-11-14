# KataGo 探针②构造回执 —— 首行即卡点

**PREREG 钉死的价值头来源「U.*」与指定工件实测冲突：cnet2_m3s2r1L.npz（md5 b6db08c9…）中的 `U.*` 不是价值头，而是策略解码器的动作嵌入投影；按字面接线会造出假玩家、且构造关会被空过（见 §2）。构造未动手、构造关未跑，待策略组裁定价值来源口径（建议见 §6）后即可开工——设计与地面工作已全部备齐（§3–§5），且 DGX 机器窗条件当前亦未满足（§5），无时间损失。**

## §1 工件定位与验合

- npz 副本：DGX `/home/arm/junyang/s26stage/data/weights/cnet2_m3s2r1L.npz`，md5 = **b6db08c9b076257c54817377bb93130d**，与 PREREG 钉死值逐字合（DGX 树内仅此一份命中）。
- 底本（main 读回验 md5）：cnet2_search.py `94f7fb6438bb7577b141e388f068e122`、packs/cnet2_search_pack.py `48392e98c03f25a3a6f616561c8a072e`、cnet2_np.py `d8c15ad4fdd8ee079363cf215d52daf3`、cnet2_feat.py `2a45b08b6f97508541e62867e408748e`、fastsim.py `aefe6c83826efae4de83bef8c643c80b`。本地副本在 `~/workspace/katago-probe/impl/base/`。
- harness 两件（bigeval 354bcd1b／eval_v53 45792dfb）为在册钉死值，本趟未动。

## §2 卡点证据：U.* 是策略组件，本工件唯一的价值读出是 tanh `value.*` 头

npz 全键盘点（181 键）：`U.weight (32,128)`、`U.bias (32,)`、`V.weight (32,15)`、`V.bias (32,)`、`value.weight (1,128)`、`value.bias (1,)`；**无 `utility.*`、无 `wdl.*`、无 `margin.*` 键**。meta：`{"cfg": {"C":128,"NB":8,"D":128,"E":32,...}, "epoch": 19, "params": 3480130}`。

与 cnet2_model.py（main 读回件）前向逐项对位：

- `self.U = nn.Linear(D, E)`（:171，E=32）、`self.V = nn.Linear(F.MF, E)`；解码 `lg = base + (self.V(mf) * self.U(h)[:, None,:]).sum(-1)`（:238）——U 是把 GRU 隐状态投到 32 维动作特征空间的**策略解码组件**，cnet2_np.py:200 同构消费（`self.lin(mf, "V") @ self.lin(h, "U")`）。
- `self.value = nn.Linear(D, 1)`（:164）——tanh 值头，`v = tanh(self.value(ctx))`（:219）。
- `self.utility = nn.Linear(D, 1)`（:162，注释「Stage 0 value heads (2026-11-13)」）是后加的 E[U] 头；本工件（epoch 19、无 Stage-0 头键）**根本不含 utility 头**，cnet2_np 的 `want_u` 路径对它只会 KeyError（其 docstring 明示不许静默回退值头）。

按 PREREG 字面接 U.* 的后果（为何必须先裁）：① U 没有标量价值读出可言，它是 32 维嵌入；② 「价值头输出置零」若施于 U 张量，等于把策略解码打残——落子必然大变，构造关将**空过**（防伪关恰要防这种假过关）；③ 所谓「价值叶玩家」实际是策略受损玩家，强度场读数无意义。此即 C 预检 (d) 类「读出口径不存在」的同族问题，但限于本工件的命名层，不是 C 的全面作废情形——因为本工件内确有受训价值读出（§3）。

## §3 本工件的 tanh 值头是受训的（DGX 只读实测，非构造关、仅作卡点定界）

方法：DGX 上以钉死件 cnet2_np（scratch `/home/arm/junyang/katago_probe_scratch/`）加载该 npz，对归档 roots.npz（`/home/arm/junyang/l20_ppo_u_archive_20261114/ppo_attrib/roots.npz`）前 4 局 × 双席 × 步 [200,299) 共 **792 根**做 encode（teacher 取根集登记值 2），读 `v̂ = tanh(ctx @ value.weightT + value.bias)`，û=(v̂+1)/2，对源局终局 U（rootset32 scores 按席 margin → 平台公式）。

结果：û mean 0.573、std 0.180、range [0.188, 0.814]；**Pearson(û, 终局 U) = 0.849**；分组均值与终局 U 同向（失配最重一组 0315dedc seat1：û 0.562 对 U 0.129，其余 7 组贴合）。与 C 预检里 PPO actor 导出件「value.weight 与 init 逐字同」形成对照——**本工件的值头是真受训的**，工件内存在可忠实接线的价值来源，只是名字是 `value.*` 不是 `U.*`。

## §4 leaf 消费方式与映射设计（已备齐，待裁后执行）

现行 leaf（cnet2_search.py:663）消费方式（两模式皆纯比较量、现金量纲）：

- 终局分支返回 F01 原版终局 margin（存活方现金 − 灭绝方罚分），是**真值计分、非启发式**；_search 的 C01 注释亦明分「exact terminal」与「heuristic leaves」。
- 非终局分支＝现金差＋CARGO_W·cargo＋WORKER_V·工人＋TREE_V·树（手写启发式）。
- sweep（_search:1649）：各分支 leaf 取 argmax，且须超默认分支 **MARGIN=20.0**（现金单位）才改落子。
- mcts（_mcts:1444）：与默认分支的**配对 leaf 差**取均值（LEAF_Q 时取分位差），> MARGIN 才改。

设计（单变量＝仅非终局叶评估）：终局分支保持原样（真值计分不属待换的启发式）；非终局叶值改由网络 û 经平台 U 公式（cnet2_build._u_of_margin：U = 0.5·WDL + 0.5·sigmoid(M/1000)）的逆映射给 margin 等价值 **M_eq = 2000·atanh(2û − 1)**（与舰队 v=2u−1 口径自洽；对 tanh 头有 atanh(tanh(z))=z，M_eq 恰＝2000×头预激活值，数值干净）。量纲核验：û 实测域 [0.19, 0.81] → M_eq ∈ [−1435, +1435]，与 leaf 现行现金量纲（终局现金差同量级）相容，MARGIN=20 的相对意义不变。策略先验、分支生成、rollout、守卫等其余机制一字不动；价值前向复用 pol 自身 encode 的 ctx（与玩家同帧同 teacher 约定），不另起表示。

消融口径（随 §6 裁定一并确认）：「价值头输出置零」＝ v̂:= 0（û≡0.5、M_eq≡0）——叶值全常数后 sweep 退化为恒选默认分支，改变数＝活值玩家与默认落子相异的根数，定义干净。

search 预算参数（现行 DEFAULTS，cnet2_search.py:35–38，本趟冻结）：SEARCH=True、MODE="sweep"、H=5、BUDGET=0.38、HARD=0.6、MARGIN=20.0、MAX_BR=16、MIN_DEPTH=3、KEY_R=2、MAX_KEY=3、MINC=10.0、CARGO_W=0.9、WORKER_V=30.0、TREE_V=2.0、ENDGAME_STOP=1、TEACHER=1、OPP_TEACHER=1、ROLL="s64"、LEAF_Q=None、GUARD=False、C_PUCT=1.5（mcts 用）、ROOT_S=12（mcts 用）。强度场派单时若部署包 CONF 与此有出入，以包 CONF 为准并在回执注明（属下一趟）。

## §5 根集与机器窗现状

- 根集：rootset32 = 32 局（rootset32.json 裸 list，{key, md5, names, sides_in_ds, gi, scores}）× 双席 × 步 [200,299) = 6,336 根；roots.npz 与 rootset32.json 已在 DGX 归档 `/home/arm/junyang/l20_ppo_u_archive_20261114/ppo_attrib/` 内，构造关驱动可直接在 DGX 复用 Stage 1 的 build_roots 路径重建/读取。
- 机器窗（2026-11-14 22:55 DGX 直读）：load ≈ 0、无 join/annotate 进程在跑，但 K0 标注回片仍在同步（k0_dist/returns c1 22:01、c2 21:56），**join 与 σ̂ 报告尚未落地**——作战室排的「join＋σ̂ 之后」条件未满足，本趟未上机跑关（亦因 §2 卡点本就不许跑）。DGX 配置：20 核、121 GB RAM。
- 本趟只读实测的线程口径：OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1（§3 检查）；venv `/home/arm/junyang/carbon/.venv/bin/python`。

## §6 待裁事项与建议

请策略组裁（PREREG 修订或确认）：价值头来源由「U.*」更正为本工件的 **tanh 值头 `value.*`**（û=(tanh+1)/2、M_eq=2000·atanh(2û−1)、消融＝v̂ 置零），其余注册条款（同件同源、除叶外全冻结、种子块、判读三格）不变。备选（不建议）：另行指定含 utility 头的工件——须换 md5 重新注册，且不再是 m3s2r1 本体件，与「本体起点」的外框冲突。

裁定一到，本线可立即动手：变体模块（探针侧、不改 main 本体）＋构造关驱动均按 §4 设计落码，DGX 窗一开即跑关、计数先报。本趟未产出任何探针产物、未改 main 任何文件、未启强度场/自博弈。

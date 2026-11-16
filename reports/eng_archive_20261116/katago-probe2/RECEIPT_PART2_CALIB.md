# RECEIPT — KataGo 新探针 Part 2 价值叶量纲校准（PREREG_SEARCH §2）

**状态：三件交付完成。注册量纲 a ＝ 3347.515730530027（对解析锚 3000 偏差 +11.58%，越 10% 线、归因见 §2.3）；根级质量验判读＝未通（改子根 ΔU 负占比 0.53 过半，按注册判读线「负占比过半＝量纲或接线未通」如实停报，未调参、未重跑）。** 实现＝工程组，验收＝策略组；产物全为诊断件，未入 main、未开任何对局场、未用注册种子块。

---

## §1 解析锚独立复推（SOP 闸一口径：亲读定义源，不抄注册文本）

源：仓库 main `code/cnet2_build.py`，GitHub API 读回亲验 md5 `e005152499b2ba30a56f92c7096e2707`（81,813 B，与在册验收值 e0051524 合；DGX 两处拷贝 carbon/code 与 carbon-repo/code 同值）。

读到的原文：

- :390（build_game 内、逐席逐行贴标签处）：`r["value"].append(np.tanh(mg / 3000.0)); r["teacher"].append(tid)`
- mg 的定义（build_game :340–356 上下文）：函数按席生成任务，`margins[p] = scores[p] − scores[1−p]`，其中 scores 是该局记录的终局分；:345–356 还把声明的 raw margin / u 与由 scores 算出的 margin 及 `_u_of_margin` 交叉校验（容差 1e-4 / 1e-6）。
- :721–722（文件内文档段）同义复述：`scores (value = tanh(margin/3000) with margins the per-side score differences)`。
- :168–174 `_u_of_margin`：`U = 0.5·WDL + 0.5·sigmoid(M/1000)`（WDL：M>0 取 1、M<0 取 0、M=0 取 0.5）——本回执全部 U 换算用此式。

推导：值头训练标签 v ≡ tanh(margin/3000)，margin 为按根席位定向的终局分差。头输出 v = tanh(z)，z 为预激活（value-diag HEAD_MAP 已证读法）。tanh(z) ≈ tanh(margin/3000) ⇒ z ≈ margin/3000 ⇒ **M_eq ＝ 3000·z**。解析锚 = 3000·z 独立成立（探针② 的 2000·z 系修订一注册笔误，本趟复推不依赖该注册文本）。

## §2 实测拟合（留出根集）

### §2.1 留出集 provenance

- 语料：ds_m3s2，DGX `/home/arm/junyang/carbon/data/ds_m3s2/meta.json` md5 `1d7d41febe1813467e1e83b312b8cae8`（实测合在册值）；回放在 `/home/arm/junyang/carbon/data/m3s2_corpus/replays/`。
- 构造：文件名序 [32:64] 共 32 局（Stage 1 根集 rootset32 为同序 [0:32]；构建脚本内断言 sorted[:32] 与在册 rootset32.json 逐字合，通过）。步 [200,299)、双席，status 口径经 build_game 特征化（cnet2_build `e0051524`、teacher 钉数据自带值 [2,2]，与 Stage 1 build_roots 同约定）；真值 = ds meta 记录的终局 scores 按根席位定向的 margin。
- 规模：6,336 根，每（局，席）恰 99 根（min=max=99，无缺根局）；teacher 全为 2。
- 清单：`heldout_manifest.json` md5 `de0ab6d83c2f3b05f09a360aa51930dd`（32 局 key＋逐局回放 md5＋scores 全在内，本地镜像 `~/workspace/katago-probe2/part2/data/`）；特征集 `heldout.npz` md5 `284d8892debd36263fa63b6f793f2327`。局 key 一览：40669bba…、43f6b9f4…、455da1a0…、47b55b74…、4c24327c…、4e40c1f6…、4e4f9c9d…、4f0c910a…、4f735257…、50b996e3…、523ae2da…、53ae5a80…、53d9e03d…、54ce2b60…、5846f162…、5b87b7dc…、61497a5e…、64c763b6…、66600af7…、68bbe064…、6c4770a4…、70057404…、71d7816b…、72d937eb…、76a6b30c…、79a5b80c…、7d9d15ef…、7da0ef21…、7ffa7e69…、805240eb…、835513be…、8374405e…（全名见清单文件）。

### §2.2 z 提取与管道校验

z 取法：m3s2r1 本体（DGX `/home/arm/junyang/s26stage/data/weights/cnet2_m3s2r1L.npz`，md5 实测 `b6db08c9b076257c54817377bb93130d` 合钉死值）经仓库 adapter（rl2_ppo_cnet2.Cnet2Adapter.make_model）torch 前向，v 为 tanh 头输出，z = atanh(v)（float64）；批次拼装照 value-diag common.batch 口径。

先行校验（拟合前）：同一管道在在册 roots.npz 前 4 局 792 根上复现构造回执的身份测量——Pearson(û, U) = **0.8485**（在册参考 0.849），û 均值 0.5734／标准差 0.1795／范围 [0.1875, 0.8138]（在册 0.573／0.180／[0.188, 0.814]）逐项合。管道可信后方跑留出集。

### §2.3 拟合结果与归因

过原点回归 M_eq = a·z（n = 6,336）：

| 量 | 值 |
|---|---|
| **a（注册值）** | **3347.515730530027** |
| 对 3000 偏差 | **+11.58%（越 10% 线）** |
| Pearson(z, margin) | 0.9545 |
| R²（过原点；margin 均值≈0 故 centered 同值） | 0.8270 |
| 残差分位 5/25/50/75/95 | −1858.3 / −972.2 / −402.4 / −132.1 / +231.9 |
| 残差均值／标准差 | −579.6 / 652.3 |
| MAE / RMSE（在 a 处） | 638.9 / 872.6 |
| MAE / RMSE（在 3000 处对照） | 642.6 / 894.8 |
| z 分布 均值／标准差／范围 | 0.1732 / 0.5430 / [−1.080, +1.308] |
| margin 分布 标准差／范围 | 2098.0 / [−3496.1, +3496.1] |
| 逐局 a：min／中位／max | −136.1 / 3253.0 / 3986.5 |

**归因（偏差 >10% 的成因，诊断量均为非注册口径、只作归因）**：失配不是纯尺度误差，而是「加性偏移＋负侧压缩」的结构失配，单一乘性常数吸收不了：

1. **零点偏移**：按 z 十分位分箱，箱均 margin 的零交叉在 z ≈ +0.28 而非 0；z∈[−0.03, +0.22] 的整箱均 margin 仍为 −710。带截距诊断拟合（非注册形式）给出斜率 3687.9、截距 −638.6 margin 单位——头的零点相对真值系统性偏高（对根席位乐观）。
2. **负侧（败势）压缩**：z<0 半区单独过原点斜率 4592.8（n=2605），z≥0 半区 2908.2（n=3731）；最负分箱（z 均 −0.694）箱均 margin −2768，而 3000·z 只给 −2083。正侧高分箱则与 3000·z 近合（z 均 0.789→margin 2367 vs 2368；0.972→3007 vs 2916）。
3. **斜率随 margin 窗口单调变**：|m|≤1000 子集 a=1313.6（n=1188）、≤1500 → 2168.2、≤2000 → 2550.8、≤2500 → 3074.6、全域 3347.5——不存在一个对全域成立的过原点尺度；注册 a 是全域最小二乘折中。
4. 残差系统性偏负（中位 −402）即偏移在过原点形式下的投影；逐局 a 有一局为负（−136.1，该局 z 与终局 margin 反相关，属后期逆转局）。

**Part 3 前置风险项（如实登记）**：① a 偏 3000 达 +11.58%，且成因是结构性的——即便用拟合 a，价值叶对败势叶的定价仍系统性偏低（量级不足）、对根席位整体偏乐观，MARGIN 在 Part 3 按残差尺度重钉时须把这条算进去；② §3 根级质量验在拟合 a 下仍未过注册判读线（见 §3.3）。两条均待策略组验收裁量，本趟不自行处置。

## §3 根级质量验（真续局）

### §3.1 变体与消融

- 变体：`part2/modules/cnet2_search_vleaf.py` md5 `1bf1848145dcbe485e518189adb8bda8`——探针②冻结件 `4a3b5f1b45fde9b659a383c1c3fba87f` 的模块拷贝，**仅**把价值叶量纲字面量 2000.0 改为拟合 a＝3347.515730530027 及其注释行（diff 全表存 `part2/data/variant_diff.txt`，两处 hunk 均在 leaf 块内）；冻结件本身未动、md5 复验仍 4a3b5f1b。其余四模块与冻结集逐字节同（feat 2a45b08b／np d8c15ad4／search 94f7fb64／fastsim aefe6c83）。
- 消融：rootset32（md5 `c0871c50…`）、构造关同口径（gate_driver.py `2858ea72` 原样、三臂同根同种子、冻结 DEFAULTS），DGX 16 worker：searched 6,006/臂、errors 0；**changed(value@a vs heur) = 100 根**；附带 changed(value@a vs const) = 85、changed(const vs heur) = 34；wall 1343.8 s。
- 负载注记：本趟 DGX 有他线在跑，单根均时 1.0468 s（探针②构造关趟为 0.4507 s／wall 574.5 s）；底座预算是墙钟口径，改子集对机器负载敏感——2000·z 趟的 120（vs const）／153（vs heur）与本趟的 85／100 **不可直接比**，本验判读只立在本趟 100 根的续局上，不作跨趟计数比较。

### §3.2 续局仪器（显著声明）

**两分支同仪器、对称使用**：改子分支与默认子分支的续局玩家同为 Echo3 部署包本体（md5 `7b7620499411a088cf4dc17c0e929bd7`，生产 CONF、SEARCH=False），双席皆然——包括根步的对手动作（在每个分支内由同一包在 pristine 根状态上算出）。选它的理由：强、确定性、部署形态，作测量仪器比搜索玩家干净（策略组验收已裁接受，骑缝即此对称性，本趟照此执行）。对称性实证：100 根两分支的根步对手动作逐根断言相等，**100/100 全等**（continuations.jsonl 内 opp_root_action_symmetric 全 true）。

- 续局构造：自记录回放的根状态（seat-0 条目全观察，含真实 full_carbon）以 FastCarbon.load_observation 重建，tid 计数 +500 照搜索层自带口径；根步根席位强制分支动作、对手席位走包，其后双席全走包至终局；每（根，分支）包实例状态全新重置（_AG=None），np 种子按根号同种子双分支同值。
- 终局计分：步限（step_count 达 300）＝终局现金差——经记录局实测校验（300 步局末状态现金与 meta scores 逐字合）；灭绝＝F01 原式（leaf 内原文：灭绝方 step_count−300−1、存活方带入终局步的现金）。事实注记：ds 语料 118 局中 0 局以灭绝终局，本趟 100 根 × 2 分支续局亦全部步限终局（limit 100／extinct 0）——F01 分支本趟未被触发，其口径靠代码引证、未经实测局校验。
- ΔU：根席位终局 margin 经 §1 的 `_u_of_margin` 换算，ΔU = U（新子分支） − U（默认子分支）。

### §3.3 结果与判读

改子根 100 根的 ΔU 分布：

| 量 | 值 |
|---|---|
| 均值 | −0.00136 |
| 下四分位（Q1） | −0.01636 |
| 中位 | −0.00035 |
| 上四分位（Q3） | +0.00183 |
| **负占比** | **0.53** |
| 零占比 / 正占比 | 0.12 / 0.35 |

**判读：负占比 0.53 过半——按注册判读线，量纲或接线未通。** 如实停报：未调参、未重跑、未换口径。事实附注（不改判读）：ΔU 量级整体贴零（中位 −0.0003、Q3 仅 +0.0018），即拟合量纲下被改写的子在强仪器续局下多为近中性、负侧略多；0.53 是贴着线过的半数，分布形态供策略组验收时 weighing，本趟判读以注册线为准。

### §3.4 限定（底座差异与口径）

- 本趟消融与变体坐在探针②的 DEFAULTS/teacher=1 墙钟底座上；Part 3 的对齐底座是 Echo3 CONF＋sims 时钟、TEACHER=2，两者不同——本验只校量纲与根级质量，不预判 Part 3 场效。
- 拟合侧 z 的特征化口径 = build_game 数据集特征化（teacher [2,2]），与在册身份测量同口径；消融里叶的 z 走搜索自身 Feat 路径，两口径之别已声明。
- 续局仪器自根步起以全新状态起算（Feat 历史自根起积累），双分支对称、已声明。

## §4 工件 md5 全表

| 工件 | md5 |
|---|---|
| 变体 part2/modules/cnet2_search_vleaf.py | `1bf1848145dcbe485e518189adb8bda8` |
| 冻结件 impl/modules/cnet2_search_vleaf.py（未动） | `4a3b5f1b45fde9b659a383c1c3fba87f` |
| Echo3 包（续局仪器，DGX 侧 echo3_pack.py＝本地 echo3_ref.py） | `7b7620499411a088cf4dc17c0e929bd7` |
| gate_driver.py（原样复用） | `2858ea720d9139d26f0feef9f0560c63` |
| build_heldout.py | `636d8bf7f1f5e4d2a3a9d1a423e8e9aa` |
| fit_z.py | `61c2ce4738fac14faf61f742b90ca3d6` |
| cont_driver.py | `bdd7b44fa7b72a29940c792e32a293aa` |
| heldout.npz | `284d8892debd36263fa63b6f793f2327` |
| heldout_manifest.json | `de0ab6d83c2f3b05f09a360aa51930dd` |
| heldout_z.npz（z＋margin 逐根） | `17f26aa926f9a686e34ab6fba3bc331e` |
| fit_results.json | `bdf65e9a05f2adab06c3ba4e2d58d97b` |
| out_gate_a/summary.json | `d0406f712e2d863f08e8c71a996db0c9` |
| out_gate_a/results.jsonl | `8057d96432bcb502dbd704cfc783a7cc` |
| out_cont/continuations.jsonl | `c00d1f0031fd35511a308448ae102eb7` |
| out_cont/continuation_summary.json | `284ab572fa5b6cb1d0f88db3f578de83` |
| variant_diff.txt | `90c7b6abefcb780b8bdd91de2771aaba` |
| 权重 cnet2_m3s2r1L.npz | `b6db08c9b076257c54817377bb93130d` |
| rootset32.json | `c0871c50d0cc1eea7e07f8aed6587838` |
| ds_m3s2 meta.json | `1d7d41febe1813467e1e83b312b8cae8` |
| cnet2_build.py（main，解析锚源） | `e005152499b2ba30a56f92c7096e2707` |

工件位置：本地 `~/workspace/katago-probe2/part2/`（modules/、scripts/、data/ 含 fit_results、manifest、消融 summary、续局全量 jsonl）；DGX `/home/arm/junyang/katago_probe_scratch/part2/`（含 heldout.npz／heldout_z.npz 与 out_gate_a／out_cont 全量）。

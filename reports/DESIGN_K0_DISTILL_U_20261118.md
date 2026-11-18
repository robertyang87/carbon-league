# 设计：k0_distill 改造版——终局 U 价值半环（策略组，2026-11-18）

- 依据：裁定 reports/RULING_KATAGO_CORE_U_HEAD_20261118.md（12b4a792）。目标：让终局 U 成为训练目标，补全 KataGo 全环的价值半环。隔离纪律：钉版 k0_distill 2728fee6 与封印格式 cnet2-k0-annotation/1 原件全不动，改造版另立文件与版本，BC 线继续用旧版。

## §1 封印 schema 变更面：新版式 cnet2-k0-annotation/2

- /2 在每根记录上加三个终局标签字段（从本根座位视角）：
  - `terminal_u`：本局终局 U（口径与设计逐字一致：U＝0.5·WDL 值＋0.5·sigmoid(M/1000)，WDL 值胜 1／平 0.5／负 0，M 为本座视角终局分差；每局先算 U 再作标签，禁止由 E[M] 代入 sigmoid 冒充）。
  - `terminal_wdl`：胜／平／负（0/1/2）。
  - `terminal_margin`：本座视角终局真实分差（原值，未裁剪）。
- 来源：产局侧 sidecar（terminal_u.json，每局 margin/u/终局状态，已在冒烟中落盘验通），由 join 改造环节并入每根；join 改造同样另立版本，不动钉版 join。适配器（7d5bd94a）已保证 sidecar 随语料同行。
- 改造版训练器**只吃 /2**；/1 语料喂入即 die 并点名版式，不许静默降级跑成 policy 单环。

## §2 价值头处置：解冻三头、终局标签监督

- 从 FROZEN_PREFIXES 移除 `utility.`／`wdl.`／`margin.` 三前缀——此三头与设计 §4.1 三目标同物（裁定 §2），全环改由终局标签直接监督：
  - utility 头 ← `terminal_u` 直回归；
  - wdl 头 ← `terminal_wdl` 分类监督；
  - margin 头 ← `terminal_margin` 回归（标签按 /1000 缩放，σ=1000 口径）。
- `aux.` 维持冻结（分布／灭绝／归属图等诊断头不进本环训练目标）。policy 头与 anchor 机制沿现行不动。头结构不动、权重热启动连续。

## §3 损失定义

- 总损失 ＝ policy CE ＋ anchor KL（沿现行口径与权重）＋ **λu·MSE(utility, terminal_u) ＋ λw·CE(wdl, terminal_wdl) ＋ λm·MSE(margin, terminal_margin/1000)**。
- 权重起点：λu＝1.0、λw＝0.5、λm＝0.5（起点值；首环读数后按纪律一次只动一个变量调整，改动留痕）。

## §4 输入槽与防假跑

- load_annotation 键集加 §1 三字段，**任一缺失即 die 点名**（防 U 半环假跑：有版式无标签不得开训）。
- 开训前坏根预扫硬闸照旧（训练器判据、全集逐根、报告入开工回执）。

## §5 版本与命名

- 训练器改造版新文件（建议名 `code/k0_distill_u.py`，工程组可按仓内惯例定名，回执报最终名与 md5 入册）；join 改造版同理另立。产物与权重落盘须自带版本戳，区分旧线。

## §6 实施验收（工程组冒烟回执必含）

1. 全链冒烟：产局→建集（/2）→双目标训练 10 步→落权重，RC=0。
2. **三头 loss 分项数值**逐项列明（utility／wdl／margin 三项非零且有限——证明 U 损失真在跑）。
3. **解冻对账**：训练前后参数比对——utility／wdl／margin 三头参数确有更新、aux 头参数逐位不变。
4. 新版文件 md5＋产物册登记。

## §7 钉版对账协议确认

- 工程组先行协议（同局 seed 987001、同窗 [200,230)、其余钉版件不动、只换 cnet2_search c3222eea ↔ 94f7fb64 两臂对照、逐字段报 max 差与 flips、独立件入 reports/eng_verify/、开正式环时并入开跑回执）：**口径无异议，照此执行**（一次只动一个变量，合纪律）。

# spot-29 任务书（作战室 2026-11-12 22:37 派定，本线 = 原 H23 线）

## 开考前提
- 推栈代理回执（标记 S29PUSH2230）＋ H27 md5 复核 ＋ 冒烟通过，齐了即开考，不再回等。
- harness 先验：eval_v53.py md5 6d24da0e、bigeval.py md5 a5822338，不符先换再考。
- 满载口径：62 worker ＋ OMP/MKL/OPENBLAS/NUMEXPR 线程全钉 1。
- 逐局线每 2 分钟机外快照到新分支 sync/h29。
- 战报大白话格式：先写清对战双方全名＋身份，胜负/margin 注明以谁为口径。
- 机器被回收立即报作战室、不硬撑；补机走常设规，H27 唯一执行。

## 被测方加载统一口径
- 每臂一张冻结 wrapper exp/<臂名>.py（make_search_agent 模板，SEARCH 关、TEACHER 2 全程、greedy），只改 docstring 臂名与 _W 权重文件名。
- 权重经中转落地后逐个 md5 对账再开考。

## Job 1 · Fable r1 定标
- 被测：Fable5.2-Max-V5 复刻注意力臂 r1。权重在 DGX：cnet2_fable_r1_fromstudio.npz，md5 8bec64d7618686f52f73d592e7099940（ep19）。
- 落地后先回作战室报文件名＋md5 存档。
- 双场：对 Master2 真身 seed 996000、对 Echo2 seed 999000，各 200 局。
- 桥基线：本尊对 Echo2 天梯 −187±317（n=5，≈平手带）。附报种树量/残树均值。

## Job 2 · M7Bs v2 双臂定标
- DGX 上 cnet2_m7bsv2r1 / cnet2_m7bsv2g1 的 ep19 文件（落地报 md5），各跑双场 {996000, 999000}。
- 桥基线 +88±287。

## Job 3 · Carago r1 定标
- DGX 注意力臂 ep19（best val 0.9086），先定位回报名＋md5。
- tag：caragor1_m2 / caragor1_echo2，种子同上。桥基线 −241±326。
- 裁决一出作战室转 L20，Carago r1 将以 0.5 入 snipe 池当压力型陪练。

## 裁决口径
- 标尺一校准分 ＝ 对 Master2 原始均分 − 785。
- 桥裁决与本尊天梯分差比：1 SE 内合格、1–2 SE 陪练可用、>2 SE 不合格。
- 逐场出数逐场报，不等全队列跑完。

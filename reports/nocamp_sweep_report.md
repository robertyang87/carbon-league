# no-camp 穷尽扫荡报告（nocamp_sweep，2026-11-04）

目的：nocamp_bench 已排除存档冠军 aux_seg2_zero（对 nocamp_recon
−14,414.6）。本单把仓内其余 Claude 侧 NN 本体逐一对 nocamp_recon
打完，钉死 OpenAI 侧所称「NN 对 no-camp 8 局 6 胜、场均 +409」是否
可能出自仓内任何一份 Claude 做的模型。

## 候选与实例化

| 候选 | 本体 | 原对 base 成绩 | 实例化口径 |
|---|---|---|---|
| 蒸馏混合版 | distilled_v1_hybrid.py（NN＋四盾，权重 distilled_v1_weights.npz） | −15,686 | make_agent()，harness 同 nocamp_bench |
| 纯蒸馏 NN | distilled_v1.py（权重同上） | −15,864 | 同上 |
| ctx seg2 冠军 | caseb_rl_ckpt/policy_ctx_seg2.py（权重 policy_ctx_seg2_weights.npz） | −9,899 | 同上 |
| 案 C 分配层 BC | casec_policy.py（权重 casec_ckpt/casec_bc_weights.npz） | −14,023（G1） | 按 casec_report §7：权重＋教师开局 N=40（warmup）＋plant_lock=False |

四份全部可独立加载运行，无一跑不起来。casec 实例化另做恒等校验：
扫荡前用同组装对 base 打校验局（seed 1000、side 0、不留痕），终分
[38.22, 11555.29] 与 §7 G1 记录逐分一致——plant_lock=False 即 §7
本体（与 P1.5 锁探针的恒等结论互证），实例化无歧义。

## 协议

与 nocamp_bench 一字不改：random.seed(7919*seed+104729*side+31)、
policy=被测方、seed 1000–1003 × 双边 = 每份 8 局、FastCarbon、
同款账本、全程 traces 留痕并登记 traces/INDEX.md「no-camp 穷尽
扫荡」节（32 局）。前三份走 caseb_rl_probe._job（open_n=0，与
bench 完全同路）；casec 走 casec_g1_probe.play 的同款账本与开局
组装、仅换对手。每份首局复跑核对：4/4 终分逐分 MATCH（+0.00）。
脚本 nocamp_sweep.py，逐局数据 nocamp_sweep_results.json。

预注册判读：复现 +409 ＝ 场均分差落在 +409 的 ±2 个样本 std 内
且胜场 ≥5/8。

## 结果（被测方口径，均值；对手统一 nocamp_recon）

| 候选 | 胜负 | 场均分差 ± 样本std | 存款 | 树收入 | 死亡丢碳 | 终局滞留 | 距 +409 |
|---|---|---|---|---|---|---|---|
| ctx seg2 冠军 | 0 胜 8 负 | **−13,627.9 ± 2,540.2** | 2,777.7 | 272.1 | 774.6 | 100.8 | 约 5.5σ |
| （已验）aux_seg2_zero | 0 胜 8 负 | −14,414.6 ± 1,543.4 | 2,898.5 | 311.7 | 720.2 | 55.0 | 约 9.6σ |
| 蒸馏混合版 | 0 胜 8 负 | **−15,034.7 ± 1,609.7** | 1,812.7 | 309.7 | 236.5 | 24.9 | 约 9.6σ |
| 纯蒸馏 NN | 0 胜 8 负 | **−17,356.2 ± 1,021.6** | 687.8 | 187.5 | 894.4 | 50.6 | 约 17.4σ |
| 案 C 分配层 BC | 0 胜 8 负 | **−19,898.2 ± 1,591.3** | 513.9 | 722.3 | 360.7 | 0.0 | 约 12.8σ |

逐局分差五份全负、无一局胜：ctx seg2 区间 −17,877 ~ −10,332；
混合版 −17,370 ~ −12,542；纯蒸馏 −19,338 ~ −16,189；案 C
−21,488 ~ −17,243。四份新验候选的预注册判读均为**不复现**
（场均与胜场两项判据同时落空）。

## 结论

**仓内不存在能复现 +409 的 Claude 侧 NN 本体。** 五份候选
（蒸馏纯/混合、RL ctx seg2、RL aux_seg2_zero、案 C 分配层 BC）
对同一个 nocamp_recon 合计 0 胜 40 负，场均分差在 −13.6k ~
−19.9k 之间，与 +409 的距离最小也有约 5.5 个样本 std（ctx
seg2）。排序与各自对 base 的强弱一致（ctx seg2 最强、案 C
最弱），传递性大体自洽（如 ctx seg2：对 base −9,899、nocamp
对 base +3,352，预示约 −13.3k，实测 −13,628）。

因此 OpenAI 侧 +409 所指的「NN」只能是仓外本体——其新建或
续训的另一份 NN / checkpoint，不是我方交付的任何一份模型。
我方五份 NN 本体对 no-camp 档（天梯约 1,950–2,080 带）的真实
位置全部为深负，与此前结论一致：约 2,000 分的估算套不到
任何一份仓内 NN 头上。

## 更正（2026-11-04）

更正：+409 后被证实来自当时在接手方分支 claude/nocamp-eval-g1 上的一份真实新 NN，现已并入 main（code/submission_nn_v1.py）；我方按规范协议复核为对 nocamp_recon 6 胜 2 负、场均 +317.3，+409 复现。本报告的 0 胜 40 负结论仅适用于当时 main 上的五份存档 NN，依然成立；『仓外本体』的定性作废。详见 reports/nn_v1_check_report.md。

# 裁定 · ③ 第二批 gen4 包 --verify 步时 FAIL 不挡触发（工程组裁定，2026-11-20 11:27 CST）

## 裁定请求

作战室派单：DGX 对 gen4_fixed_u.py 的 pack --verify 已结束（约 11:16，跑约 34 分钟，PID 2625484）。
batch2 的 Studio 搜索触发条件是「DGX 包 --verify 通过后才开两座搜索」——步时 FAIL 是否挡住触发？
gen4_fixed_u.py 是定额纠正包（K=12/TTA=4，从不去天梯），0.75s 是天梯步时口径。

## 直读核实（主代理 11:27，DGX 直读）

`lossprobe3_scratch/build/gen4_verify.log` 全文（1,359 B）：
- 打包：chassis apexcarbon_v1_fix3.py，value-scale bridge 已注入，clock-removal 补丁已打，
  post-condition：输出无 perf_counter。
- 前向对检查点参考网 max abs err **0.0000**（容差 0.25）→ **PASS**
- 确定性：1 局 × 2 runs **IDENTICAL actions**（final cash [8848.52, 8478.03]）→ **PASS**
- 步时（1,196 agent calls）：mean **1.669s**、p95 **3.708s**、max **23.745s**（limit 0.75s）→ **FAIL**
- PID 2625484 已不在册，verify 已结束。

包 md5（DGX 直读）：gen4_fixed_u.py **b4b55da0b1df7db49eae929cef15af16**；
仪表化版 gen4_fixed_u_lp3.py **5b2f5dcf2356a4c323ed65b706099844**。

## 裁定：口径错套，步时项不挡触发——Studio 开搜

**结论：开搜。** 触发条件的通过项为前向 PASS＋确定性 PASS（两项均已过）；步时项对本包为 report-only。

依据（两条，均为已落仓先例）：

1. **设计件原文已裁过同款情形**（reports/DESIGN_SEARCH_CORRECTION_20261120.md L42–45，
   第一批 gen3 包 --verify 实测）：「步时均值 1.159 s／p95 1.375 s／max 1.944 s（天梯线 0.75 s）→ FAIL。
   步时 FAIL 是天梯部署门（固定预算壳本就不适配天梯单步时限，Echo8 前例在案）」。
   第一批全量重搜即在同款步时 FAIL 下跑完并落仓定版。0.75 s 是天梯单步时限口径，
   与离线纠正包的正确性无关——本包从不去天梯，步时只影响墙钟成本。
2. **正确性两腿全过且无瑕疵**：前向误差恰为 0.0000（容差 0.25）；确定性整局逐动作相同
   （final cash 逐字同）。步时慢（均值 1.669 s，高于第一批的 1.159 s）与 max 23.745 s 单点毛刺，
   均属 DGX 争用下的墙钟读数（同驻：价值谱正式训＋消融 vec2 臂＋本包 verify），不是包缺陷。

## 执行口径（立为本批与今后定额纠正包的常设规则）

- 定额纠正包的 --verify 触发条件：前向 PASS ∧ 确定性 PASS；步时项只报数、不参闸。
- 本批：执行人立即开两座搜索（Studio＋DGX），按补充件（ab0d96aa）冻结的 11 局败局跑。
  DGX 侧逐位置耗时会高于第一批（争用），进度以落盘局数计，不以步时计。
- 包本身无需修。gen4_fixed_u.py b4b55da0… 与仪表化版 5b2f5dcf… 即本批定版包。

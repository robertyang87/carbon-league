# 补钉：预算赛 B 主体身份（Studio 线，2026-11-19）

- 补钉对象：reports/PREREG_BUDGET_RACE_20261118.md（5fa5c5eb）§1 的 B 主体身份占位（「权重与包 md5 出炉时补钉入本件补注，先于 B 场开打」）。
- 配对口径：按封印件 reports/SEAL_BUDGET_RACE_PAIRING_FIX_20261118.md（ee6290b1）——每对手 200 个种子各 1 局、side＝seed%2，与 A 同 (seed,side) 集逐行配对，d＝U_B−U_A；判读按所有者执行计划 §3（7190544b）。
- **B 主体身份钉定**：
  - 学生权重 loop2_student.npz md5 **4c49f28146bd3912282db8c80d77faab**；
  - 学生对局包 loop2_student_int8.py md5 **162d6142af95cffd7f71f31100fbf3f2**。
- 在册时点：上两值于发车前已在 L20 回填表与作战室派单（2026-11-19 01:28 CST）中在册；本补钉为发车后补记，不影响本场效力（作战室 2026-11-19 01:31 令）。
- 发车记录（Studio 第二考场）：学生包自 L20 直拉落 Studio，机上 md5 逐字合后用；发车前对手包逐字复核过闸（Master3s2 wrapper b4cde262695d38270e16bde33875a443、Echo6 25f144f3d9ecc87be9073d135d3ab8c8）；驱动 bigeval 354bcd1b、eval_v53 45792dfb、run_budget_b.sh 6c4667c7；2026-11-19 01:30:18 CST 发车，场一 vs Master3s2（962000–962199，tag budgetB_3s2_962000）、场二 vs Echo6（962200–962399，tag budgetB_echo6_962200），各 200 局。

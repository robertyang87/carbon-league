# 裁定：小教师定标比对——对齐键改为 (game, step, action)（策略组，2026-11-18）

- 凭据：工程组比对件 reports/eng_verify/compare_stage4_teacher_vs_student_396.json（commit 24f02550，策略组读回 md5 adc9cb0c 合）。注册口径：小教师注册（3e74e4ef）§20–23——测量用 Point B 附卷 compare 仪器、附卷集 994400／994401（396 根）；Tier 1 四项（flips=0 硬项／|bias|≤1e-4／max|Δq|≤1e-3／max|Δdu|≤1e-3）；Tier 2 宽屏（flips≤2%／|bias|≤1e-3／max|Δq|≤5e-2／max|Δdu|≤5e-2）＋下游验证闸。

## §1 事实（验合一致）

- 994400：arms_set_diff_roots 198/198，compared_roots **0**，全部度量为 0（无分母）。
- 994401：arms_set_diff_roots 197/198，compared_roots **1**，q_max_abs_delta 0.0907、argmax_flips 1——n=1 无统计意义。
- 身份面全合（config_sha256、games_sha256、replay_sha_match 均为 true，structural_mismatch null）；两趟标注各 396/396 根完工。**标注本身无问题，收敛侧（2,680 步、val total 1.52240）不受影响。**
- 工程组诊断成立：比对器按模型相关 arm 身份对齐，学生搜索走出的 arm 集与教师基本不交——这是对齐口径缺陷，不是学生质量问题。

## §2 裁定

- **对齐键改为 (game, step, action)**：同一局面、同一动作的 q/du 比较。Tier 1/2 四项度量的定义与阈值不变，只改 join 键；新比对件须明示 compared_roots 与可比动作条目数。
- **可比性下限（本裁定所定）**：新键下 compared_roots < **100**（396 根的约 25%）时，Tier 1/2 判据仍不具统计意义，不许硬判 Tier。
- **备选口径（仅当触发下限时）**：学生在教师 arm 动作上的直读 q 评估——学生模型对教师 arm 集合内每个动作直读 q（不经学生搜索），四项度量同口径计算；件中须明示「直读口径 ≠ 搜索臂一致性口径」，Tier 判读以直读四项为准并标注口径变更。备选口径不改变 Tier 1/2 的阈值本身。
- 旧比对件中 994401 单根的数值（q_max_abs_delta 0.0907、flip 1）**作废，不得引用**。

## §3 执行

- 工程组按新键重跑 compare（shard_compare v4），出 Tier 1/2 数值＋可比根数；若触发下限，另行提案备选口径读数。Tier 判读待新比对件到后由策略组落。

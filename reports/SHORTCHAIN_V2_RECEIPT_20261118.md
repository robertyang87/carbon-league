# k0_distill_u/2 真实短链验收回执（工程组，2026-11-18，L20）

ACCEPT_DISTILL_U_V2（aa48e46d）生效前置的执行回执：§10 配方逐项照办、无减项。工件在 L20 `/root/shortchain_v2/`，生产数据只读消费。

## 数据与环境

- 数据：L20 `/root/carbon/katago/loop_pilot/joined_u`，generation gen-90faee2ef4221d59，manifest_sha256 8bfc9284…，89,700 根（标注源 gen-a8e3b6661ec59257、build 源 gen-a58e4e5dd0ddcb75）；数据代 teacher_weights_md5＝06a03d9c…。
- 教师/环父网文件：`cnet2_m3s2r1_heads_v2.npz`，机上 md5 实测 `06a03d9cfeec239af3065a49fefec382`。
- 训练器：仓上 `code/k0_distill_u.py`（/2）机上 md5 `cfbf227e3cf7ad08795cc251ee9ba0fa`；配套件逐件验合。torch 2.15.0+cu132、device=cuda；开跑前 GPU 0%、load 0.02、无在跑进程。
- 两段各 20 updates（bs 64、lr 3e-4），段 1 约 13.1 s、段 2 约 13.1 s。段 1 产物 seg1.npz md5 `8d767c1d170d15cfd664ff266f7005dd`；段 2 产物 seg2.npz md5 `f1b0e4f545630d769acc30fd1dc6a3a2`。

## 六项验项（逐项读回判定）

1. **初值继承 PASS**：段 2 登记 student_init.md5＝8d767c1d… 与段 1 产物独立复算逐字合；段 2 trainable_moved 182 参数，独立复核 184 个非 aux 数组中 182 个与段 1 不同（确从段 1 字节出发并移动），aux 两件逐字节同。
2. **教师身份保持 PASS**：两段 label_teacher.md5 均 06a03d9c…，与数据代 teacher_weights_md5 逐字合；teacher_config_hash 两段同为 3cc2e67e…。
3. **锚点 PASS**：段 2 anchor_continuity.status＝"held"，previous_kl_reference_md5＝06a03d9c…；recipe_change＝[]（段 1 为 ring_anchor_baseline）。
4. **有效 lr/step PASS**：两段登记 lr＝0.0003、bs＝64、max_updates＝20 与命令逐项一致，updates_done＝20、log 尾 done 标记合。
5. **冻结闸 PASS**：两段 frozen_check.bitwise_identical_to_student_init＝true、moved＝[]；独立复核 aux 与各自 student_init 逐字节同。
6. **负例 PASS**：段 2 换 label_teacher＝loop1_student.npz（e28a5dd1…）→ 开跑前 die、RC=1、无产物。die 原文：「annotation producer teacher_weights_md5 '06a03d9c…' != --label-teacher file md5 'e28a5dd1…'; the recorded priors/q were computed under different weights than the declared label teacher; refusing to start」。

**结论：六项全 PASS，/2 入册生效前置已满足。**

—— 工程组 2026-11-18

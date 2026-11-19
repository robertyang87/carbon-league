# ApexCarbon-V4TTA-Search（fix2）— 归档记录（2026-11-19）

## 来源

- 所有者于 2026-11-19 17:41 CST 直接上传（文件名 ApexCarbon-V4TTA-Search-fix2.py），令考试。
- 谱系：第一版（Echo7）→ fix 分支（code/apexcarbon_v1_fixed.py）→ 本 fix2。与 fix1 代码相似度 0.988、与第一版 0.922，权重两行与第一版逐字相同。

## 入仓文件

- `code/apexcarbon_v1_fix2.py` — 与上传件逐字节一致
  - 大小：2,895,001 B（1,603 行）
  - md5：`800ba824c30f77acc252c3acf49db058`
  - sha256：`e06100820be708b3b65566dd8ebaf188d8e5a6c15ae462c3d0711a947510a920`
  - 入口：`agent(observation, configuration)`

## 我方核验（本 VM）

- AST 语法解析：PASS；导入冒烟：PASS（7.29 秒，本机读数偏慢，仅供参考）；静态扫描：无网络 / 文件 / 子进程调用。

## 考试

- Studio 钉版考场，考官 Master 3s2（opp_master3s2.py b4cde262），种子 975100–975103 试点，与 v1／fix1 同口径配对。
- **试点（tag gemini_apexv1fix2_m3s2_pilot4）：2 胜 2 负，平均分差 +672（SE 559，最差 −233），err=0、FAIL=false，单步最差 313 ms。** 逐局：+822／−98／−233／+2,196。台账：存入 8,014 vs 考官 7,988、树收入 4,113 vs 3,060（全家最高）、种树 32 vs 28。
- **对 v1 同种子配对：4 个种子全胜式占优（+240／+685／+478／+698），配对差 +525（SE 107.5）**——本家首个在试点全种子压过父版的候选。对 fix1 配对差 +391（SE 750），逐局大摆动、不判。
- 参照：v1 试点 +146、fix1 试点 +281、v2 试点 −1,007；fix2 的最差局 −233 为全家最佳下限。n=4 仍属试点，定论需 40 局配对（v1 的 40 局基准 +341.5 已在册）。
- 台账在本机 ~/workspace/gemini-apex-v1fix2-20261119/exam_pilot4_m3s2/。状态：未提交天梯。

# ppou2 检查点规程（开训前备忘，细节以开工令为准）

## 检查点考试（主臂每 20 轮／对照臂每 40 轮）
1. 取该轮 ckpt_N.npz（out 目录，ckpt-every 1 已开）。
2. 打包：L20 /root/carbon/code/packs/cnet2_search_pack.py（48392e98）
   量化核对齐已验（2026-11-14 22:3x）：packs/cnet2_pack.py 在 L20 与 DGX 同为 18bf47b7；
   仓库根版 cnet2_pack.py（119e9bb9）的 quantize/dequantize 与之 AST 源码级逐字同——
   L20 本地打包与 DGX 打包链（a4_pack_stage）量化口径一致，检查点包可 L20 自产，DGX 出包为备援。
   `python3 cnet2_search_pack.py <out>.py "<CONF 字面量>" ckpt_N.npz`
   CONF 字面量必须与 m3s2r1_base.py 内嵌的逐字一致（打包前从该 pack 文件内 grep 提取比对：
   SEARCH False／ROLL self／GUARD True／GUARD_NARROW True／MODE sweep／MIX False／TEACHER 2／
   TEACHER_LATE 2／LATE_STEP 240／BUDGET 0.7／HARD 0.85／ABORT_DL 0.95／START_DL None）。
   成品 md5 入检查点记录；打包在考试目录 /root/carbon/probe_m2/code/exp/ 下进行。
3. 两场（考试目录、线程全 1、EVO_WORKERS 24、顺序跑）：
   - vs Master2：种子 980000–981999，N=2000，tag ppou2_m_it<N>_m2（对照臂 ppou2_c_…）
   - vs Echo2：种子 982000–983999，N=2000，tag ppou2_m_it<N>_e2
4. 判读：与第 0 步冻结基线（ppou2_base_m2_a＋ppou2_base_m2_b 两批、ppou2_base_e2_a 一批）
   逐种子配对 ΔU；检查点读数对两批 M2 基线各配、均值定格。
   存活线：双考官合计 4000 种子簇，点估计 ≥+0.01 且 CI 下界 >−0.01。
   停线：连续两检查点合并 CI 上界 <0；或累计 KL 超顶（见下）；或单轮 KL ≥0.13（先停）。
   CI 口径：种子聚类 bootstrap（§17 先例，B=20000 级）。
5. 累计 KL(π‖π_起点）：用 /root/ppo_attrib/kl_measure.py＋rootset32.json（Stage 1 同工具同根集），
   参照＝本体 npz 导出；每检查点量一次，均值 >0.15 停线。
6. 结果表：主臂/对照臂并排报作战室＋策略组（双方全名＋口径，战报规矩）。

## 暖机锚（三件套，it80 口径）
- 双臂各：vwarm 3 轮毕、首个正式轮前 cp state_last.pt → out/state_anchor_frozen.pt，md5 入备案附件；
  验 actor 与 init npz 逐张量全等（导出比对）、train.jsonl 首行 kl_coef=0.20、league.json 池表与备案一致。

## 时间盒
- 硬停 2026-11-20 12:00 CST：停训 → 终局确认考（保留块 984000–984999，双考官各 500 簇，
  或按令提前触发条件）→ 归档。L20 周包 11-20 23:59:59 到期。

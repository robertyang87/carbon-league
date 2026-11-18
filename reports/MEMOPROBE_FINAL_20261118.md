# 跨根评估冗余度测量 · 终局报告

日期：2026-11-18。座位：L20 CPU 侧单核（taskset 1 核）＋nice 19＋BLAS 单线程，未占五座标注产能。数字全部出自原始 jsonl 聚合（scripts/aggregate.py ＋ scripts/final_extra.py），无估计值。

## 方法

- **仪器化副本**：钉版 code_pin 整套复制为 probe_code，仅两处纯包装补丁（计算零改动、不碰 RNG、只读数组）：
  - `cnet2_np.Policy.act`（md5 40ba55ce）→ 改名 `_act_impl`，新 `act()` 在每次前向记录：输入特征张量 x 的 sha256（按 CachePolicy._sig 的九字段序＋teacher，记为 **hx**）、aux 哈希、forced 哈希、调用标志（sample/forced/hook/want_u/temp）合成的完整输入哈希 **hfull**、输出哈希 **hout**（labels＋logits＋value＋u）与确定性输出哈希 **hnet**（logits＋value＋u）。
  - `cnet2_search.decode_b`（md5 ef36ce9f）→ 同式包装，记录守卫重解码前向（entry=gdec）。
  - 其余文件与钉版逐件 md5 全合（teacher c6ade5ce／search 其余部分／np 其余部分／feat 2a45b08b／fastsim aefe6c83／gate e922d9d8／model 3bb5857f／k0_annotate 7f567a2e／eval 45792dfb）；权重用 L20 座上件读回 06a03d9c；回放用 L20 prod_self 件、sha256 与 annot_a 身份段逐字合（300002 ed1c735a／300003 4d74c55a）。
- **仪器化点定点依据**（代码直读）：evaluate_root 的全部网络前向都经 `pol.act`——对手根动作（M=8，带 sample_rng）、gen_candidates 的贪心/forced 偏离/forced 重打分、续步双座贪心、_leaf_u（want_u）。守卫 BankruptcyGuard.apply 另可经 enc_for＋decode_b 重解码（enc_for 在本调用形下恒命中 act 刚写的签名缓存、不产生新 encode；decode_b 单独计数）。
- **副本验真 PASS**：300002 步 200/215 × 双座 4 根（注：任务书的步 211 不在 recon_pin.jsonl 的步集 {200,215,229,230,247,264,265,282,298} 内，取最近的 215 替代），与 recon_pin.jsonl 逐根比：priors/q/Q_mean/Q_std max diff **全 0.0**、arms 全等、n_act_calls 全等、仪器记录数＝计数器值（680/根）。采样全程逐根三方计数全等（200/200，聚合脚本断言通过）。
- **采样**：2 局 × 100 根＝200 根（中段步 200–229 双座 60/局、早段步 2–20 偶数步双座 20/局、晚段步 270–288 偶数步双座 20/局），生产口径 K16/M8/H2、teacher-style 2、belief sample、seed-teacher 7000。总采样墙时约 55 分钟（单根 eval 约 15.4–16.0 秒），**未触发缩减**——三步带全额采完。

## 结果

### 总量与 u（唯一 ÷ 总前向）

总前向（act 调用）**132,056** 次；守卫重解码 gdec **0** 次（200 根内守卫从未触发重解码，前向总数即 act 计数）。

| 范围 | 前向数 | 唯一 hx | u_hx | 唯一 hfull | u_hfull |
|---|---|---|---|---|---|
| 全集 | 132,056 | 121,988 | **0.9238** | 125,096 | **0.9473** |
| 中段 | 81,600 | 75,739 | 0.9282 | 77,659 | 0.9517 |
| 早段 | 23,256 | 21,088 | 0.9068 | 21,636 | 0.9303 |
| 晚段 | 27,200 | 25,161 | 0.9250 | 25,801 | 0.9486 |
| 300002 | 66,030 | 61,188 | 0.9267 | 62,743 | 0.9502 |
| 300003 | 66,026 | 60,800 | 0.9208 | 62,353 | 0.9444 |

### 分解：冗余全部在根内，跨根为零

- hx 口径：逐根内唯一之和 121,988 ＝ 全局唯一 121,988 → 根内冗余 10,068 次（7.62%)，**跨根冗余 0 次（0.00%）**；逐个核查：在 ≥2 根中出现的 hx 数 ＝ **0**。没有任何输入特征张量在两根之间重复——**跨根缓存零头寸**（与中段连续块的主信号假设相反：相邻根的局面张量逐字节全异）。
- 调用构成：贪心 99,592／forced（偏离＋重打分）6,016／叶 want_u 24,848／对手采样 1,600（＝200 根 × 8 场景，逐根核对合）。

### 恰零前提检查（同输入哈希 → 多输出）

| 键 → 输出 | 冲突组 | 组内调用 | 占比 | 归因 |
|---|---|---|---|---|
| hx → hout | 382 | 7,600 | 5.76% | 见下两行分解 |
| hx → hnet | 382 | 7,600 | 5.76% | 与 hout 同（冲突在 logits 层、非仅标签） |
| hfull → hout | 182 | 1,384 | 1.05% | **全部**为 sampled 对手调用 |
| hfull → hnet | 182 | 1,384 | 1.05% | 同上 |

hx 冲突 382 组分解：200 组由完整上下文不同解释（同 x、不同 forced 前缀/aux/标志——gen_candidates 对根局面的 forced 重解码，logits 随前缀自回归变化，构造必然）；182 组与 hfull 冲突集重合（同 hfull 仍多输出）。

hfull 残余冲突 182 组逐组验明：**全部** flags＝(sample=True, forced=False, want_u=False)、**全部**同根内、组内调用数典型 8（＝M 个场景）。机理：belief-sample 的 8 个场景里对手观察张量常逐字节相同（采样到的未知格不在对手编码状态内），但对手动作按各自 rng_m 采样、采样动作自回归改变后续 slot 的 logits——输出依赖不在输入哈希内的 RNG 状态。首例：根（300002，步 200，座 0）seq 1 起 8 次调用同 hfull、8 个不同输出。

**前提结论**：非采样调用类上，完整输入（x＋aux＋forced＋标志）→ 输出是严格函数：132,056 次调用中除上述采样组外零反例。恰零前提对 **hx 单键不成立**（forced 上下文）、对 **hfull 键仅因采样 RNG 不成立**；对非采样路径的 hfull 键成立。

### 真恰零可缓存量（act 级）

同 hfull 且输出全一致的非采样重复调用：**5,757 次（占总前向 4.36%**,5,148 组）——这是 act 级记忆化在恰零约束下的真实头寸。另 encode 级：每次 act 都全量重算 encode（act 路径不查签名缓存），同 hx 的重复 encode 10,068 次（7.62%）属主干重算——但其收益只及 encode 部分（剖析口径下 encode 约占网络评估的 2%)，且全部在根内、CachePolicy 的 last_enc 机制已部分覆盖守卫路径。

### 每根前向次数（与 n_act_calls≈609 互证）

200 根：均值 **660.3**、min 386、max 680；中段全 680（满臂）、晚段全 680、早段均值 581.4（min 386，浅步臂数少）。与 recon 集 36 根（均值 608.9、满臂 680、晚带均 466.7——其晚带步位更深、臂更少）同构，无矛盾。

### 缓存键成本参考

输入特征张量 x 恒定 **11,849 字节**（200 根 × 全部调用无例外，固定形状）；键材料另加 aux（逐调用哈希已记录）。hx 键即九字段原始字节序拼接，与既有 CachePolicy._sig 同源。

## 结论

1. **u ≈ 0.924（张量键）/ 0.947（完整输入键）**：全搜索标注的评估冗余总共只有约 5–8%，且**跨根冗余为精确的 0**——跨根/全局缓存方向没有头寸，先量后定规格的这一步把该方向判死。
2. 根内冗余里，act 级真恰零可缓存只有 **4.36%**（5,757 次）;hx 层面的重复大半被 forced 前缀与采样 RNG 区隔，硬缓存会破恰零。
3. 恰零前提本身：非采样路径在完整输入键下严格成立（零反例），采样路径因 RNG 在键外而不成立——任何记忆化规格必须把 forced 与采样标志纳入键、并把对手采样调用排除在外。

## 产物与证据

- 原始记录：results/sample_calls.jsonl（132,056 行 act 记录，sha256 8f68089e7d0a9abeb425416ce37b7f716bb6ca757d6828781b3ebd6ed2eca169；座上原件 /root/memoprobe/results/，本机件已由 .xz 传输落地并 sha256 验合）、results/sample_out.jsonl（200 根结果，sha256 971db6fb…)、results/verify_calls.jsonl（2,720 行）、results/verify_out.jsonl、results/summary.json。
- 脚本：scripts/probe_worker.py（采样）、scripts/apply_patch.py（仪器化补丁）、scripts/compare_verify.py（验真）、scripts/aggregate.py ＋ scripts/final_extra.py（聚合）、scripts/staged_agg.py（阶段快照聚合，阶段数见 STAGED.md）。
- 座上收尾：L20 memoprobe worker 进程已退出、直读确认无残留（NO_PROBE_PROCESS）；采样全程单核 nice 19，未与生产链抢占（盘点负载 0.0–1.0）。

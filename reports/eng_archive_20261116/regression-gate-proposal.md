# Carbon League 持续回归门禁方案（工程组提案，2026-11-14）

## 总约束

- **全线绕开 GitHub Actions**（所有者 2026-11-14 定规：不 dispatch workflow、不依赖其运行/日志通道）。本方案不含任何 workflow/CI 托管环节；所有门禁在舰队机器本地运行，结果以「入仓报告文件＋边聊回执」留痕。
- 门禁的意义是把历轮审查靠人肉复跑的验收变成**固定入口、固定口径、失败即阻断**。每层写明：跑什么、谁跑、多久、失败阻断什么。
- 耗时口径：标「实测」者有历史回执依据；标「推算」者按脚本结构（局数/步数/教师单价）估量级，未单独计时者如实标注——**先跑先测，数字以后续实测回执为准**。

## 第一层 · 快速集（入仓/刷盘前，执行线本机）

执行人：提交或刷盘的那条线，在自己机器上、动手前跑；结果（逐套 PASS/FAIL＋总耗时）随提交回执/刷盘回执发出。环境要求：torch venv（Studio/DGX/L20 均有；本 VM 无 torch，不承担本层）。

| 套件 | 入口（仓内现行） | 检查数 | 单趟耗时量级 | 状态 |
|---|---|---|---|---|
| PPO U 口径 | `automation/acceptance_s1/test_ppo_u_mode.py` | 18 | 秒级–1 分钟（内存造 tiny 随机网、不训练；推算） | 现成 |
| PPO resume/预热 | `automation/acceptance_s1/test_ppo_u_resume.py` | 23 | 分钟级（tiny 模型走 trainer 全程；未单独计时） | 现成 |
| 镜像身份 | `automation/acceptance_s1/test_mirror_identity.py` | 15 | 分钟级（小语料扫描；推算） | 现成 |
| gate 身份 | `automation/acceptance_s1/test_teacher_gate_identity.py` | 3 | 分钟级（确定性续弈 worker；推算） | 现成 |
| 历轮审查反例集 | `automation/acceptance_s1/test_review_fixes.py` | 随轮增长 | 分钟级（本地 fixture；推算） | 现成，**须持续追加** |
| 模拟器 parity | `code/test_fastsim_parity.py` | 36 | 分钟级（2 局×300 步双模拟对拍＋13 构造场景；需机侧 zerosum_env；推算） | 现成 |
| 标签契约 | `automation/acceptance_r5/test_labels.py` | 17 | 分钟级以下（推算） | 现成 |
| 发布原子性 | `automation/acceptance_r5/test_publish_f05.py` | — | 分钟级以下（推算） | 现成 |
| producer 接线 | `automation/acceptance_r5/test_ppo_wire.py` | — | 分钟级以下（推算） | 现成 |
| 双世界（小跑） | `automation/acceptance_s1/test_teacher_v2_twoworld.py`（默认权重） | 37 | 分钟级（推算） | 现成 |

规则与待建：

- **反例集纪律**：每轮审查确认的每个缺陷，修复提交必须同时往 `test_review_fixes.py`（或对应专题套件）追加一条能红能绿的反例；没有反例的修复不算验收完成。本轮待补：rl2_ppo_eval temp 钳制反例、split PPO 身份反例（均随策略组代码轮落地时补）。
- **test_derived 夹具改造（待建，工程组）**：`automation/acceptance_r5/test_derived.py` 的手搓 manifest fixture 在现行「指针在场即硬拒不完整 generation」口径下按设计失败，须改造为真实封代 generation 后才能进快速集；改造前该套件**不计入**门禁通过率，单独标注。
- **统一 runner（待建，工程组）**：`automation/run_fast_gate.sh`（本地脚本，非 workflow）——在钉住的 venv 里依次跑上表、逐套打印 PASS/FAIL＋wall time、任一失败 exit≠0，并产出一行机器可读摘要供回执粘贴。现各套件靠人肉逐个跑，是本层目前最大的执行风险。
- **失败阻断**：任一红 → 不许提交该修复、不许以该 SHA 刷盘；已入仓的按回退/前滚修复处理并在回执里点名红项。

## 第二层 · 慢回归（Studio/DGX 定时与触发式）

执行人：Studio 线（教师类）与 DGX 线（打包/建集类）分担；触发方式两条腿——① **事件触发**：凡触及 teacher/gate/search/build 的验收轮收口时，由执行线在本机跑对应子集；② **定时兜底**：两机本地 cron 每周一趟全量（本机脚本＋日志落盘，不经 Actions）。结果回执：当趟摘要发作战室＋策略组边聊，并把完整日志入仓 `reports/regression/<日期>-<机器>.md`（走 GitHub API 提交，非 Actions）。

| 套件 | 入口 | 检查数 | 单趟耗时量级 | 跑在哪 |
|---|---|---|---|---|
| 信息集 | `automation/acceptance_s1/test_teacher_information_set.py` | 20 | 数分钟（教师评估 5.55 s/局面量级×可达局面数；推算） | Studio |
| 双世界 canon | `automation/acceptance_s1/test_teacher_v2_twoworld.py --canon`（heads v2 正典权重，K16/M8，步 8/60/120/200） | 37 | 数十分钟（推算，未单独计时） | Studio |
| S1 接线 A1–A3 | `automation/acceptance_s1/test_s12_a1.py` / `a2` / `a3` / `a3_teacher` | — | 分钟级（需机侧 fixture：20 局 m3s2 回放＋heads_v2.npz） | Studio/DGX |
| 打包一致性 | `code/cnet2_search_pack.py --verify-pack` 逐包＋`code/certify_pack.py`；附 `cnet2_search.py` 双副本（code/ 与 code/packs/）md5 等值检查 | 逐包 | 秒–分钟/包 | DGX（打包已收敛 DGX 单线，runbook v2） |
| 建集兼容冒烟 | 新 builder 对既有 shard 语料 scratch 建集（§18 兼容回执的固定化） | — | 分钟级（语料大小相关；推算） | DGX |

- **失败阻断**：慢层红不回滚已入仓代码，但**冻结下一轮相关面刷盘与验收结论升级**（PASS-WITH-NOTES 不得转 PASS），直至复跑绿并回执；canon/信息集红另触发教师身份复核（按 §16(a) 撤回先例处理）。

## 第三层 · 制品验收（既有规矩固定化，不新增工具）

- **入仓**：put 前读回最新版带 sha、put 后读回验 md5 与字节——现行铁规，继续执行；大文件（>1MB）走 blob download_url 验，不用 Contents 内联内容。
- **刷盘**：回执必须附**目标路径全量 md5 扫**（不只 diff 文件；2026-11-14 fastsim 旧版漏刷教训），＋CODE_VERSION 戳更新＋旧版备份目录留痕。
- **打包**：每包入 `reports/live/packs_ledger.md` 台账（构建机 DGX、--verify-pack 通过、双跳 md5 一致才算交付）。
- **失败阻断**：任一 md5 对不上＝该次刷盘/交付作废重做，不许「大概没问题」放行。

## 责任矩阵（摘要）

| 层 | 执行 | 结果回执到 | 红的后果 |
|---|---|---|---|
| 一 · 快速集 | 提交/刷盘的执行线本机 | 提交/刷盘回执内 | 阻断提交与刷盘 |
| 二 · 慢回归 | Studio（教师类）／DGX（打包建集类） | 作战室＋策略组＋入仓报告 | 冻结下轮刷盘与验收升级 |
| 三 · 制品验收 | 入仓/刷盘/打包的执行线 | 各自回执＋台账 | 交付作废重做 |

## 与本轮审查的衔接

- 第 2 项（eval temp 钳制）、第 3 项（split PPO 身份）的修复归策略组下一代码轮；落地时按第一层「反例集纪律」各补一条反例进 `test_review_fixes.py`，本方案不另设专题套件。
- 第 5 项两件待动态验证（BC 数值保护、meta 完整性）不进任何一层门禁，直到验证完成、反例落地后再议入集。

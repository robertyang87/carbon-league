# 第五轮（REVIEW_FULL_20261113）最终验收证据归档

本文件把最终验收的证据从本地工作目录归档入仓，使验收结论可独立核对。
验收分两层，**不得混记**：

- **冻结验收**：对象是冻结 SHA `031d46c187518a6fbd255a7184025f4976867cd5` 的代码树。
- **冻结后追加**：冻结之后又有 12 笔代码提交（验收中修复 5 笔、验收后收尾 4 笔、gen_key 微修 1 笔、探针钉 2 笔），每笔的追加测试单列于 §2。验收中修复 5 笔的证据就是验收套件本身（修复在验收过程中完成、随即由同一套件验证），**不记作冻结版原测**。

运行环境：验收机本地 venv `~/workspace/integ-accept/venv`，torch 2.15.0+cpu；归档脚本保持原始形态（含本地绝对路径依赖），逐个登记于 §4，sha256 以本文件为准。大体积产物不入仓，只登记路径与 hash（§5）。

## 1. 冻结验收（冻结 SHA 031d46c1，验收后 HEAD fa180736）

| 项 | 测试脚本（automation/acceptance_r5/） | 结果 |
|---|---|---|
| 派生层消费端（set_uw 登记＋train load 校验） | test_derived.py | **13/13 PASS**：真 set_uw 登记 sidecar（cnet2-derived/1、base gen 对齐）；load verified 模式权重逐字节等；手工 loss 与消费端一致（weight_gate prod_n=144）；三篡改（旧代 base/rows/sha）＋镜像单方篡改均点名拒；gen 目录内 set_uw 被护栏拒；无 sidecar 旧集 legacy 模式可 load 且打印 legacy-unverified 警告 |
| F01＋F06 整合 | integ/test_integ.py ＋ integ/test_awr_chain.py | **27/27 ＋ 9/9 PASS**：迁移组含参数 +6（96→102，小 cfg）、policy/value 误差 0、name-swap 被拒、新格式 round-trip；F06 门 uw inf／w NaN／w·uw 溢出点名拒、极端有限 uw=428293.7 放行且 prod_max 入册、loss 回归值 316666688.0；AWR 全链 floor 1e-3 点名、anchor uw 有界（1000/375）、乘积 == scale·mean(w)=1.2802 |
| F01 PPO 接线（审计发现 adapter/resume 原为裸 strict 加载） | test_ppo_wire.py | 全 PASS（此项同时是验收中修复 e3f220b8/fed5adc0 的验证，见 §2） |
| 混合标签四类贯通 | test_labels.py（结果集 results/data-results.json 等） | **17/17 PASS**：builder manifest exact 4186 / approx 598 / unusable 598；fault 的 u == 纯 WDL {0., 1.}；unusable 与 approx 的 margin/u 为 NaN 不零填；heads EPOCH train_value_rows=**4186 = 4784 − 598**（unusable 不进 value 分母）；heads manifest approx_labels=true、label_source=mixed |
| F02/F03 | test_awr_f02.py ＋ recheck_labels_fa.py（Codex 原探针适配副本） | LB/DAgger 终局后行数 **0**（原 30/298）、margin **−349**、terminal=extinction；DAgger 前缀特征差 **0.0**（原 0.5） |
| F04 certify | test_certify_f04.py | **7/7**：首调用慢 3.05s FAIL、末次推进慢 FAIL、每步慢 FAIL、两局各 2 慢步 PASS、对手缺失 exit 2、n1r 短测 PASS（峰值 RSS 84.6 MiB） |
| F05/F11 | test_publish_f05.py ＋ test_identity_f11.py ＋ recheck_data_fa.py | replace 注入失败后 current.json 仍指旧代、旧代 2392 行完整；权重改 1 字节 resume → RECONCILE-MISMATCH（adopted=0）；seed0 carbon 非零格 103==103；terminal 短局被收录 |
| F07 | test_f07.sh | **9/9**：旧失败 seed 唯一重跑、ALREADY_COMPLETE 全集闸、跨 run 旧 DONE 归档 .stale、期望列表重复 seed exit 3 |
| bigeval 重复 seed | （F07 套件内） | EVO_SEEDS=17,17 → 显式 ERROR、**exit 2** |
| ladder 口径 | test_ladder.py | **4/4**：在册但 eval_failed 判 APPLIED；POST 含糊失败＋查无此名 → 恰 1 次 POST 停手 |
| 机械项 | — | 全 code/ 树 py_compile 全过；packs --check-source ok=true、mismatches=[] |

## 2. 冻结后追加（12 笔代码提交，逐笔）

| 提交 | 变更 | 追加测试（输入身份 → 结果） | 提交后文件 md5 |
|---|---|---|---|
| bd403d11 | set_uw 接入护栏＋register_derived 登记 | 验收套件 test_derived.py（冻结树＋本修复）→ 13/13，登记 sidecar 生成且 base gen 对齐 | code/set_uw.py `2acb148b8a620fadee969961b3df427b`（2,955 B） |
| 7e4ef5e3 | cnet2_train.load() 派生层身份校验 _validate_derived_layer | 同上套件：三篡改＋镜像篡改点名拒、legacy 警告在册 | code/cnet2_train.py `0389e7a5fc33078d731c48c82755493d`（45,332 B，后被 2a1b6ca3 取代） |
| e3f220b8 | PPO adapter make_model 改走 migrate_state_dict | test_ppo_wire.py → adapter 加载旧/新格式均可（对照组裸 strict 加载旧格式失败） | code/rl2_ppo_cnet2.py `7a68d699…`（全文见提交） |
| fed5adc0 | PPO resume 经 _resume_sd 走受检迁移 migrate_optimizer_checked | test_ppo_wire.py → 旧格式走 migrated（动量逐位、step 5.0、lr 3e-3 保留）、新格式 strict、损坏 checkpoint 抛 RuntimeError | code/rl2_ppo_train.py `f443a972…`（全文见提交） |
| 2c4c1fc7 | producer 裸 CLI 补 code/ 父目录入 sys.path | 冻结树上裸跑 --help 复现 ModuleNotFoundError → 修复后 `env -i` 裸跑 rc=0 | code/producer/selfplay_gen.py `5fe72063cb1a27df56a6eb6fa122bbc5`（35,096 B，后被 3c6a0467 取代） |
| 2a1b6ca3 | 残留3：train load() 先经 current.json 哈希验证 resolve 到 generation 目录 | test_r3.py → **16/16**：replace 注入失败后 load 只读旧代完整 2392 行、18 字段行数一致；回归 test_publish_f05_r/test_derived_r 在新码上 rc=0（results/pub_r.log、der_r*.log） | code/cnet2_train.py `b320a2ecb403961bcea30a84f1e7a5f5`（50,028 B，终值） |
| ab8db3e0 | 残留3 builder 侧：扁平 meta 盖 flat_generation/flat_total 戳、失败写 BUILD_FAILED.json | 同 test_r3.py 16/16（失败落盘 BUILD_FAILED、指针仍指旧代） | code/cnet2_build.py `b83df6daa27cced87a51259932e4b977`（61,489 B，后被 a7ed8f95 取代） |
| 3c6a0467 | 残留4/5：play() 本体错误增量检查（属性/globals/closure 三路去重）＋max-games 批界口径入 help | test_r45.py → **7/7**：裸 agent 直调 play()、root_errors 恰 2 → 该局被拒、无回放落盘；--help 含批界说明；--max-games 1 --workers 3 实产 3 局与新口径一致 | code/producer/selfplay_gen.py `95d2e29fbec8cfcebee57bf9bf0d25ed`（37,499 B，终值） |
| 0009ac7a | 残留6：stage0_train_heads 空 val 点名断言（main 划分后＋evaluate 入口） | 空 val 单元 → 点名 SystemExit（非裸 ValueError）；全 val=False 跑 main 开训前 rc=1（results/stage0_emptyval.log、stage0_main.log）；置 1 局 val 后 1 epoch 跑通（results/stage0_ok*.log、heads_ok.npz.manifest.json） | code/stage0_train_heads.py `14ba916651987fc2a7d8e6ed23c00fa7`（14,138 B，终值） |
| a7ed8f95 | gen_key 微修：逐局 scores/margin/u＋outcome 类纳入 _game_label_digest | 双跑同一套件：**补丁版 14/14、原版同套件 7 项失败精确复现原缺陷**；只改一局 margin（900/1100→1500/500）→ 换代 gen-33eca6a1、与全新目录逐值一致；改 fault 标记 → 再换代 gen-f32603b6、u_raw 变 WDL 1.0。**该套件脚本未在本地留存**（验收机工作目录已清理），证据为本行双跑数字记录，不补造脚本 | code/cnet2_build.py `b38786537f579f421c4a1a0dc97adcad`（63,761 B，终值） |
| 015002e6 | 探针钉：sp_probe_s29.sh 的 GEN_MD5 b275e952→95d2e29f（文件头注释同步） | 读回验证 GEN_MD5 行对、无旧值残留；全仓 code search 陈旧 producer 钉零命中（见下） | code/producer/sp_probe_s29.sh `406fd25e8be0413a9f062eeef285658a`（4,061 B） |
| b778d2ab | 探针钉：sp_probe_studio.sh 的 GEN_MD5 同步换 95d2e29f | 同上读回验证；**全仓 code search（归档时点 main=b778d2ab）：b275e952 / 8aed7f48 / f7dbb47e / 5fe72063 四个陈旧 producer md5 均 total_count=0** | code/producer/sp_probe_studio.sh `2f7e7e2d07322c7861b05fa9b42632d6`（4,262 B，仓内读回值；Studio 线本地副本另含其运维注释、md5 不同，以仓内为准） |

## 3. 输入身份（关键文件 md5）

冻结版（031d46c1）与最终版的业务代码差异仅 §2 的 12 笔；最终版（b778d2ab 时点）与 a7ed8f95 的业务代码相同，差异只有两支探针脚本的 GEN_MD5 钉。最终版关键文件 md5（归档时 API 读回核验）：

| 文件 | md5 |
|---|---|
| code/cnet2_model.py | 3bb5857f833a94797ac6cdc2a5a28963 |
| code/cnet2_train.py | b320a2ecb403961bcea30a84f1e7a5f5 |
| code/cnet2_build.py | b38786537f579f421c4a1a0dc97adcad |
| code/producer/selfplay_gen.py | 95d2e29fbec8cfcebee57bf9bf0d25ed |
| code/stage0_train_heads.py | 14ba916651987fc2a7d8e6ed23c00fa7 |
| code/certify_pack.py | 3c058ddc219272c389c4061907338b41 |
| code/ladder.py | 39a7257575f7a7eb25f3e59c41cd5069 |
| code/fastsim.py（与 code/packs/fastsim.py 一致） | aefe6c83826efae4de83bef8c643c80b |
| code/cnet2_search.py | 10a0674e5fd572801d2a06876f1de54b |
| code/set_uw.py | 2acb148b8a620fadee969961b3df427b |

## 4. 入仓工件登记（automation/acceptance_r5/，sha256）

脚本（保持原始证据形态，未改头部）：

| 文件 | 字节 | sha256 |
|---|---|---|
| test_derived.py | 7,128 | f4a8ffdd0b2b9690d592afd46a5f1201655a97a628f0819ce73dc9e89856a721 |
| test_ppo_wire.py | 5,002 | bd59bcf0f2868e322b39dda07b70e10875c7f607eccd35f74e04441d92cf3c3e |
| test_labels.py | 7,050 | 643568543260c43e2666b27b0796bea150919d4e6c800f24febef9b2c487a67f |
| test_certify_f04.py | 3,558 | 803cedc19421447c015c5123a335c0d88b3753090d8f6169cc7bb2d678fa0a33 |
| test_publish_f05.py | 2,733 | 81a9cf4156cc61d48770658788ef89e64bee3ce8b257d2aecff3deab58056b4f |
| test_identity_f11.py | 2,311 | 5158cd186114582c7846313b14c9545e0eade71c1610e1b24d90a24af74a03d8 |
| test_ladder.py | 1,761 | 315b2177462975c4ecc47d76cb6c9cdfc51c3c76cf3356e93c3518e86e8d1ffc |
| test_f07.sh | 2,766 | 715d7c454e07c0fc27fcc03b9cce5b3bd78edb09dfeb1ef11ca83f41c8b5ac37 |
| test_awr_f02.py | 3,750 | 77c9549c5ced713a95ed9e0d4cd7360674c6210d80c30986848a713579defd29 |
| recheck_data_fa.py | 12,924 | ce836c1d05710e69ad1c3629d790251bd7b7e8657fac9f32fbf9a6138f89449e |
| recheck_labels_fa.py | 3,464 | b455bcad53bd2cc00222087637470788a2258d3a2a538c52380aed6ac019366a |
| integ/test_integ.py | 11,581 | e41cd52392d45fcf21064f2e1b9d248528ed212a400a8647557d9fde1d6e97b9 |
| integ/test_awr_chain.py | 5,321 | fb7c2f47d242790ce4d63c3fea32eaf8ba103156bf1d9cd910766ec9d7d87e76 |
| test_r3.py | 5,406 | 8a181effb68caee21f804f49809e38b9d644134c982729c70b4a9d619c2d91d8 |
| test_r45.py | 3,437 | 167faf73ba98e4771141c02befcfde9471d58e3f80b852e0e240ec453c19a14c |
| test_derived_r.py | 7,139 | 91e9b403db7d5d2fa0f43eeaf8dc16f48579c77820dd19c310c95bb16f2ae450 |
| test_labels_r.py | 7,061 | 642b065aa457eb84db775a433e4588ae1ed6ff0de020f01a17000f84eddc7571 |
| test_ppo_wire_r.py | 5,013 | 38e080ace758817de149aa4a2460ed3a89fb1db481f804d0684a12766b139f2f |
| test_publish_f05_r.py | 2,744 | 16b34fc94d612f8f202e3d27058c9755c5e7bea5f62868a9157d60a5cefbf963 |
| test_identity_f11_r.py | 2,322 | 1ffcd07d8332952e0b46b2f930371f9b749e5f01208e014b9e22d8e56b1792b6 |

结果摘要（results/ 与根目录）：

| 文件 | 字节 | sha256 |
|---|---|---|
| results/data-results.json | 97,641 | a3c53f3fd7d70496af28cd7a9733ff70616a26c9c65b7e724e2232ecc30e2df2 |
| results/latest-selfplay-results.json | 24,745 | b6935d885f94710006d3f7b9f67d74247b435056143e69567f25a924ffe65307 |
| results/latest-selfplay-results2.json | 24,505 | 02b760adfcda45dbe7909ce5d1957fa0884ba0c7283ef17e92e708ebc2761b4c |
| results/identity_sample.json | 671 | 22bb5bd628eba1eb50de6432338da399d8b782647ae4674a4f703537f053b3ac |
| results/identity_index_sample.json | 2,434 | 56a372df13acdf8dc608afb04196fd40bf821e74e3cb5a0cf1416e3ef57d0101 |
| selfplay_err.log | 24,745 | b6935d885f94710006d3f7b9f67d74247b435056143e69567f25a924ffe65307 |
| heads_out2.npz.manifest.json | 405 | 1054b1675c08993f8778ba9a8e427d4622b06dec624bcaa393dbbd5c4f631c1e |
| results/heads_ok.npz.manifest.json | 396 | cdfea5f197ff598ccf517655b9a337464a95e8fd81ef6ccfaf43dc516532a5f9 |
| results/r3.log | 2,529 | 02de40abe3ac4b6f6839cdb7736dd0846cfa220e2471f8a0c1bb6671ddabf41a |
| results/r3b.log | 2,492 | ac55c7b8f1832ac577ba5973cc6cf213964c89d80cc912c32c2a0d01c9836fcd |
| results/r3c.log | 2,492 | 737ff01276a264165e68abdcf8b6dd9cedefee0f0834b7ac9c2feded76f7bfe4 |
| results/pub_r.log | 1,115 | 7901fcc87a7d7244458b1e405ca3e9d4fc1fc6516c5e983864dfea2f03b44004 |
| results/der_r.log | 1,400 | e28de8f185cefdcbae2f54881d9c186222fc7382a2f7410cc1fbefbac5651bbb |
| results/der_r2.log | 1,708 | d0551115537f6de3991684738fffc148d2591a5ec06cc571323a3373b9450c7d |
| results/ds_norm.log | 335 | 9392869bb01b26c25f1a5806b3f6e383ee1edd61698f577bb8838878f5e28da8 |
| results/stage0_emptyval.log | 504 | 2bf0f86f10ed6578c09ebe6111dce7ac405b89b09fa1e1d8655d09a5701938a1 |
| results/stage0_main.log | 370 | 10afdf25e7ba58d1d04eedf9108c4636d1e003884c705f211589224cabbe6cde |
| results/stage0_ok.log | 507 | f52311b80acc449532fee48014890f240f8428b11bfd785109860ae892fd37fa |
| results/stage0_ok2.log | 1,749 | 0382f250d3ba22d748a04fd2ee12d7ff186ec23a52d525a1edbea3179f58da6b |
| results/test_labels_r.log | 3,372 | ecabe0ae848711d5a62f91c9ac21ba0051655adcf39b46a42eb49551dfccccd3 |
| results/test_ppo_wire_r.log | 624 | d5c97cda57b92e93f06c87535ee599652212233159071589c3490ef0046e024e |
| results/test_identity_f11_r.log | 425 | 394a9e8cb12a58986bc2c91804500a9f3d485b94cbd3c43cf203325eed16de05 |

注：integ/ 工作副本（验收时点的 cnet2_*.py 快照）未入仓——最终代码以 main 为准，验收脚本以本目录登记版本为准。

## 5. 未入仓大件登记（只记 hash，本地留存）

| 路径（验收机本地） | 字节 | sha256 / 清单 hash |
|---|---|---|
| ~/workspace/final-accept/work/heads_out2.npz | 658,810 | 3c3fd9328ec5198d900e4212fbe1491418bb05c1c6a9799028e17f32ffebba2d |
| ~/workspace/final-accept/work/init_heads.npz | 658,810 | 7395636e1c74b97a736e10f3037a4db405ec722eac19299c4f83b2428126a329 |
| ~/workspace/fix4-residual/heads_ok.npz | 658,810 | a1a7c53150ce29e68d836fa1484361616e6c1e84e607e60cd577f0b8aaa8d9d0 |
| ~/workspace/final-accept/work/ds_mixed/（53 文件，混合标签语料） | 78,255,624 | 清单 sha256 7e615ba523e1ca514c85b64e31d67bda37f0214038ed083311950188c1f8001f |
| ~/workspace/final-accept/work/ds_mixed2/（58 文件） | 176,071,049 | 清单 sha256 9f3ba465a8fd38c47bf9da71ffb522b7a00c3ee014b45fead33dcedf10aae3ac |
| ~/workspace/final-accept/work/ds_pub/（54 文件） | 78,255,858 | 清单 sha256 b3d4a5cafbb4d022e0164d8d02f554b70ecf284e8f092a5c10bf9282cc4f630c |
| ~/workspace/fix4-residual/ds_r3/（77 文件，残留3 撕裂复验集） | 91,300,980 | 清单 sha256 ae08ae3b5498a311d666b8a7ffb949e2d16e1d2dcafd89855e34d804576e83f3 |
| ~/workspace/fix4-residual/ds_fresh_b/（53 文件） | 78,255,619 | 清单 sha256 704eddd9ae658e3600047b884d55bae5cc31aa223cc90489ad38a107b7c5188a |
| ~/workspace/fix4-residual/ds_heads_ok/（53 文件） | 78,255,676 | 清单 sha256 2cefd93b97e1b07f0f564a18d45a17385fb87218163c182992f5ced1c338651d |
| ~/workspace/fix4-residual/ds_legacy/（23 文件，无 sidecar 旧集） | 26,073,592 | 清单 sha256 722206b736eb8ab7ba803ef6300e3ac363a9273461bb9da65731daf6c5791e38 |

清单 hash 口径：目录内相对路径排序后逐行「相对路径 空格 文件 sha256」拼接再取 sha256。

## 6. 未关闭项（不构成验收通过的一部分）

1. **完整平台资源认证未做**：1.5 CPU / 512 MiB cgroup 下单步 1s／硬顶 3s、整局 1500s 的整场认证未执行；F04 数字均为无约束宿主机短测。
2. **冻结裸网 ΔU 收益验证未做**：属 Stage 实验范畴，另行排期。
3. **Codex 证据 harness 需按最终树刷新**：automation/probe.py 及 recheck 脚本多处按第五轮源码行号/标记抽取，在最终树上不能端到端原样运行；本次验收用的是本地适配副本（已入仓，见 §4）。第六轮复核前须先刷新证据 harness。

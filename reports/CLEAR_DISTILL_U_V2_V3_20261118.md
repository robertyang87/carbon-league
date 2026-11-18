# 热启 /2 生效销项与 trainer /3 入册（策略组）

策略组，2026-11-18。依据：工程组短链回执 `reports/SHORTCHAIN_V2_RECEIPT_20261118.md`（3fd16a84，策略组亲读全文）；原验收件 `reports/ACCEPT_DISTILL_U_V2_20261118.md`（aa48e46d）生效前置；钩子裁定 `reports/RULING_PREFLIGHT_HOOK_20261118.md`（ebfd6991）。

## §1 /2 生效销项

短链在真实数据（环 1 joined gen-90faee2e、89,700 根）上以冻结 /2（cfbf227e）执行，六项验项逐项读回全 PASS，策略组核对无误：初值继承（段 2 登记 init＝段 1 产物 8d767c1d、182/184 非 aux 数组移动、aux 逐字节同）；教师身份两段同值 06a03d9c 且合数据代；锚 held、recipe_change 为空；lr/step 合注册、updates_done 满额；冻结闸 bitwise 合；负例换错教师（e28a5dd1）在开跑前 die、点名两 md5、无产物。

**裁定：aa48e46d 的生效前置已满足，三身份热启接口（k0-distill-u/2 设计）入册生效。** 步段链训练自此可使用三身份接口（其余前置如试点参数钉定各自照旧）。

## §2 trainer /3 同批入册

短链基线落地后，钩子版同批替换入仓：`code/k0_distill_u.py`（commit d558b368）——**策略组直读仓内文件验合：md5 `5d28e2f71728aeb4e4f2f5d754eff4ee`、TRAINER_VERSION＝`k0-distill-u/3`**。

**入册：现行生效 trainer 版本为 /3**——在 /2 三身份接口之上叠加 ebfd6991 主案的无条件 preflight 钩子（进程内验 sidecar 自封＋绑定、失败 die 点名、无警告路径；自审 9/9＋主代理复跑、/2 回归套件除版本断言外全过在案）。/2（cfbf227e）自此为被取代的历史基线，其短链验证结论由 /3 继承（钩子为纯增量、训练数学与既有闸逐字不动）。

**命名消歧（重申）**：本件的 /3 是 TRAINER_VERSION；稀疏设计中的 `cnet2-k0-annotation/3` 是 joined 版式号，两个命名空间不许混称。

—— 策略组 2026-11-18

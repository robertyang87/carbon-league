# 裁定：试点拼接双闸 (a) 系统性失败——k0_join _game_stem 补剥天梯 -1 后缀（2026-11-17，策略组）

承：RULING_GEN1_PILOT_ASSEMBLY_20261116（95f67ae7）双闸；作战室 2026-11-17 04:20 请裁。

## §1 事实（策略组自钉版源码直读核验）

- 仓内 `code/k0_join.py` md5 **27c11e16**（blob 2aa20075e5）即钉版 join。`_game_stem`（341–347 行）docstring 自述产出「标注 games 表用的裸 stem」，实现只剥 `.json.gz`／`.gz`／`.json`；调用点唯一（400 行），join 按名匹配（398–412 行：`build_by_stem[_game_stem(key)]` 查标注裸 id，查无即 die「no counterpart」）。
- 天梯回放文件命名为 `<match-id>-1.json.gz`（-1 为回放文件后缀、从不是 match-id 的一部分；match-id 为十六进制串，不含连字符）。build 代 gen-bc0cf40561fae1f4 的 200 个 games-table 键全为此形（L20 侧 200/200 量化），故 stem 全剩 `<id>-1`，与标注裸 id 永不相等——**函数违背其自述目的，属钉版工具自 bf2aac72 引入以来未变的既存缺陷**，本次 gate (a) 为该 build 的首次执行，此前从未跑通过。
- 试点数据与 build 代均有效、不动：对账 PASS（200/200 局、39,600 根、身份全合钉版）、6 缺根 reseal 全过、合并 PASS（gen-26d6fda809835820、200 局／39,600 根／9 源）、已交 L20（MERGE_RECORD md5 009ed075 两边同；build 代 L20 侧验合）。双闸 (b) 未跑——它只认证 join 后产物，(a) 不过即无产物。

## §2 裁定：采 (a)，驳 (b)(c)

- **(a) 通过**：仅改 `code/k0_join.py` 的 `_game_stem` 一处——剥扩展名后，若剩余 stem 以 `-1` 结尾且该 `-1` 紧贴被剥的扩展名（即键形为 `<id>-1.json.gz`／`-1.gz`／`-1.json`），剥掉该 `-1`；无扩展名的键原样返回；其余逻辑、docstring 本意之外一字不动。改后 `_game_stem` 在天梯命名键上产出真裸 stem；非 `-1` 键行为逐字不变（十六进制 match-id 不可能以 `-1` 结尾，误剥无口）。
- **(b) 驳回**：全量重建 build 代只治表象，docstring 与实现的矛盾留在 §6.4（生产 build 同为天梯命名键、同墙必再撞），且代价为整代重跑＋收据重验。
- **(c) 驳回**：拷 build 改键破封存，不合法，无此路。

## §3 执行序（两步， identity 先验后用）

1. DGX 按 §2 合同打补丁，入仓 `code/k0_join.py`（单文件提交），报新 md5＋diff stat；策略组 API 直读验合（单文件、diff 仅限 `_game_stem`、md5）后**同轮授权重跑**，DGX 收到授权即续跑。
2. 重跑 gate (a)：k0_join --verify 须 exit 0，200/200 局匹配、每局 198 根；回执附回归证据——先前失败键 `0001b27b…-1.json.gz` 现映射＋对照组（`<id>.json.gz`→`<id>`、无扩展名键原样、`x-1` 无扩展名原样）。随后 gate (b) 照旧（join --verify＋load_annotation），组装回执交 L20。
3. **重钉版**：新 k0_join md5 自本裁定起为 K0 链（试点＋§6.4 生产）钉版 join，取代 27c11e16；§6.4 派单的 join 身份以本件为准。旧版留 git 历史（blob 2aa20075e5）备查。
4. 08:00 开训闸：join 本身约 1 分钟；裁定及时＋两步衔接顺，仍可达（估计；以实跑回执为准）。

## §4 责任界定

此为工具缺陷被闸按设计抓住，非数据缺陷、非执行失误：标注批、build 代、拼接合并均按裁定合格完成，DGX 叫停正确。等价类教训：凡「按名匹配」类工具，开工前须以真实命名形（天梯 `-1` 后缀在内）跑一遍匹配冒烟，不只以 docstring 为据。

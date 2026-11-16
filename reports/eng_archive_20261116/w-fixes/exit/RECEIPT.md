# W05 修复回执 — exit_merge generation 复用漏验 meta.json

**结论**：W05 成立（merge 层），已修复落仓。trainer 侧验封覆盖 meta 哈希、实测为真第二道闸，`cnet2_exit_train.py` 不需改、未改。

## 落仓

- `code/exit_merge.py`：commit `1aad458b2a8675f7ff96d77fd87e5c88e2b37f27`，基线 md5 `2146afb92312eff6774b8357b3c557cc`（blob 43c6c32b，动手前读回核对一致）→ 新 md5 **`49c32b25ea8dd544e9e56ad4b94d6c59`**（11,118 B），sha 钉基线提交、API 读回逐字节一致。
- `code/cnet2_exit_train.py`：基线 `23b92d70889038c527ad89ce12019d9d` 读回核对，**未改动**（理由见下）。

## 复现（基线版，evidence/repro_w05_baseline.txt）

真实 `exit_gen.record()` 造分片 → 发布 gen-e8441cca2094cc32 → 已发布代 `meta.json` 原地改坏（total 与首局 n 各 +1，文件 sha 改变）→ 同输入重发布：

- merge 返回 `reused: true`、发布成功——复用核验只比 manifest 之间记录的哈希（self-seal＋core 等值）与各字段 `.npy` 的 bytes＋sha256，**已发布代里的实际 meta.json 从未被重算哈希**，坏代被「核验通过、原样复用」且指针被重新翻到它上面。W05 在 merge 层成立。
- trainer 兜底实测：`_verify_exit_generation`（cnet2_exit_train.py:212-216 对 meta.json 重算 sha256 与 manifest 的 meta_sha256 比对）拒绝消费、点名 "meta seal mismatch"；完整加载路径（resolve_exit_arg → load_exit）同样拒绝。故坏代训不进去——trainer 是真第二道闸；但 merge 层的「已验复用」结论为假、指针会被翻到自称已验的坏代上，缺陷在 merge 层照修。

## 修法（仅 exit_merge.py 复用分支）

字段文件循环之后、`reused = True` 之前，补一项：已发布代 `meta.json` 必须存在且 `_sha256_file(meta.json) == manifest.meta_sha256`，否则 die 点名文件 `'meta.json'`，绝不复用坏代。manifest 缺 meta_sha256 键时 `.get()` 返回 None、比对必不等 → fail-closed。V02 发布规则注释同步更新为全文件口径。修复后复用核验覆盖已发布代每一个文件：字段 `.npy`（bytes＋sha256 对 manifest）、`meta.json`（sha256 对 manifest 封印）、`manifest.json` 本体（self-seal＋与新构建 core 等值，其字节在指针翻转时另行哈希入 current.json）。

注：gen id 内容地址绑生产者代码 md5，修复版首发 gen id 随之变更（gen-a0f1755617e498d8），属内容寻址应然，同版内重发稳定性由回归 B1/W3 证明。

## 回归（evidence/regress_w05_fixed.json；落仓后以 main 字节再跑 evidence/regress_main.json，14/14 项判定全 PASS）

- **W1** 损坏 meta.json 后重发布 → die 点名 'meta.json'；meta 修复回原字节后 → 恢复复用。
- **W2** 损坏任一数据文件（value.npy 翻字节）后重发布 → die 点名字段（原覆盖复证）；**W2b** 篡改 manifest 内容（rows+1 未重封）后重发布 → die（self-seal）。
- **W3** 未损坏同输入重发布 → 仍为复用，manifest 字节与 gen 目录 inode 均不变。
- **V01**：全链经指针可加载（10 行，含 NaN 哨兵的真实 record() 产物）；q 真损坏五型（已评估槽 Inf／pi_slot 在而 q 全 NaN／pi 正项压 NaN q／未搜索行带有限 q／pi 有项而 pi_slot=-1）在 legacy 平坦目录上全 die 点名；行数错位、scal NaN 仍 die。
- **V02**：复用后旧指针 verify PASS；新代 rename 前故障注入 → 旧代完好、指针仍可验、解除后新代落位且旧代 manifest 不变；指针翻转点故障注入 → 旧指针仍可验。
- **D1 四项**：封印代字段翻字节经指针拒、manifest 改 rows 经指针拒、未封印 legacy 直路点名可载、封印代直指拒。

## 证据

`~/workspace/w-fixes/exit/`：repro_w05.py、regress_w05.py、make_shards.py、evidence/{repro_w05_baseline.txt, repro_w05_fixed.txt, regress_w05_fixed.json, regress_main.json}、src/（基线与修复版三件套）。

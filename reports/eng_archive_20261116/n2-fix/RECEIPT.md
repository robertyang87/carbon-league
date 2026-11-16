# N2 单读闸修复回执（2026-11-14）

仓库 `robertyang87/carbon-league`（main）。裁定口径：校验须自单次读入的字节完成、K0 消费字段去 mmap 且已验字节驻留内存；k0_train 自身残留（u_raw/label_status 训练路＋screen 路）并入同趟；`check_aux_sidecar` 一处 CLI 不可达 fallback 只记录不修。

## 落地（两件单文件提交，put 前读回基线 sha、put 后读回验 md5）

| 文件 | 基线 md5（blob sha） | 新 md5（blob sha） | 提交 |
|---|---|---|---|
| `code/cnet2_train.py` | `5310439c845b388da670131569a29453`（`ed4183da39e88233ba3b3e8950f5f3a062ceb09`，148,544 B） | `4fab36723ac4585e8e5a127a4a2d05db`（`6c497e7a06d6d64074246874541e57c3fdf3e396`，151,263 B） | `3546c2d31d75ca728d166cdfa9d6f98e406220a6` |
| `code/k0_train.py` | `4d5b3b67b55ac5df376e0b224e3f2cb7`（`eb9590d4662a0e0ffa6ef16450b367027f4e86c0`，65,216 B） | `97feaa354c8699b7f036063277d97487`（`3ddd26647c2fbd3fab3ed0d6f8df3ce98e6cd9cf`，66,476 B） | `7f97a4f8072cec7663c0e9bf92d681d11f9df1fb` |

## 基线反例（换盘窗坐实，证据在 `~/workspace/n2-fix/evidence/`）

形态（基线）：`cnet2_train.load()` 先由 `_resolve_load_source` 按路径哈希校验整代数据，随后 `D = {k: np.load(path, mmap_mode=mm)}` 按路径 mmap 再读——已验字节 ≠ 消费字节，且 mmap 惰性分页使内容随后续磁盘写入漂移；k0_train 在 `T.load` 校验之后再以 `np.load(path)` 按路径重开 u_raw/label_status（训练路基线 :496/:499、screen 路 :1020/:1023）。

- `load_base_swap.json`：校验后将 scal.npy 换为 B（A×0.5）→ load 成功且消费 == B（`consumed_eq_A=false, consumed_eq_B=true`），scal 为 memmap；load 成功后再换盘为 C（×0.25）→ 已载入的 D 随之变为 C。`load_base_ctrl.json`：无换盘时消费 == A，但 post-load 换盘仍漂移。
- `k0_base_swap.json`：np.load 钩子在 T.load 校验后、k0 按路径读 u_raw 前把文件改写为 B（sha256 `67da674697ef…`，清单 A 为 `9592491e0ed0…`）→ 进程 exit 0，消费的正是 B，两步 u-loss 随之偏移（0.11047136783599854 / 0.026951543986797333，对照组 0.06324618309736252 / 0.06239991635084152）。
- aux（`BC_AUX_W=1`，基线）：`_load_aux_bound` 先 `_sha256_path` 校验再 mmap 按路径读，post-load 换盘后 aux 数组跟随磁盘变化（`aux_follows_disk_after_swap=true`，memmap）。

## 修法

### cnet2_train.py（新行号）
- 新增 `_read_bytes`（:214）与 `_load_gen_field_bytes(src, man, name)`（:219）：清单声明字段单次读字节 → 对该字节算 sha256 与清单条目比对 → `np.load(io.BytesIO(raw))` 解析同一缓冲；不一致拒绝报文：`dataset generation file failed verification: {p} (sha256 mismatch -- file bytes changed after the generation was sealed); refusing to train`。
- `load()`：`man is not None` 时，清单声明的 FIELDS 全部走上式（驻留内存、零 mmap）；w/uw 的 generation 分支同走上式（缺文件仍先报基线原 SystemExit）；root-derived w/uw 改为单次读字节 BytesIO 解析（其文件级哈希随后仍由 `_validate_derived_layer` 按原口径校验）。
- meta.json：单次读字节；清单带 `meta_sha256` 时先验后 `json.loads` 同一字节（全路径一致）。拒绝报文：`dataset generation meta failed verification: {mp} (meta_sha256 mismatch -- meta.json changed after the generation was sealed); refusing to train`。
- `_load_aux_bound`：每个 aux 标签文件单次读字节 → sha256 与 sidecar 比对（原拒绝报文逐字不变）→ BytesIO 解析。
- 保持不变（有意）：清单未声明字段的按路径回退读（:671，该类文件无封印身份可绑）；legacy 无清单路径（:673，基线语义）；root meta 的 `derived_layer` 镜像读（:766，try/except 的信息性读取，其值随后由 `_validate_derived_layer` 对 sidecar 校验）。diff 审查：删除行恰为 6 处目标按路径加载，余皆新增。

### k0_train.py（新行号）
- 新增 `_load_gen_field(gen_dir, man, name, tag, refuse)`（:265），建在既有 `_read_bytes` 上（W02 先例同路）：按清单条目定位文件 → 单次读字节 → 清单 sha256 验同一字节 → BytesIO 解析；拒绝报文沿用调用方前缀：`k0: generation field 'u_raw' failed verification: {p} (sha256 mismatch -- file bytes changed after the generation was sealed); refusing to start`（screen 路为 `k0 screen: …; refusing to run`）。
- 调用点：训练路 :518（u_raw）/:521（label_status）；screen 路 :1042/:1045。删除行恰为原 4 处按路径 `np.load`。

## 修复后探针（同攻击面）
- load 探针 --swap：校验后换 scal → **拒绝**（上引 generation file 报文点名 scal.npy）；无换盘控制：消费 == A、非 memmap、post-load 换盘不再漂移（`after_disk_swap_still_first=true`）；aux 同：驻留、post-load 换盘不跟随。
- k0 mid-window 探针（T.load 返回瞬间换 u_raw 为同一 B，`67da6746…`）：训练路 **拒绝**（`k0: generation field 'u_raw' failed verification … refusing to start`，exit 1）；screen 路 **拒绝**（`k0 screen: … refusing to run`）。原 in-flight 钩子探针在修复版上无按路径读可挂，消费恒为真字节、loss 与基线对照逐字同。
- 注：整代预换盘（开跑前已换）由 `T.load` 内 build 的 `load_generation(verify=True)` 先行以 `ValueError: generation file hash mismatch` 拦下，基线/修复版同——漏洞窗只在校验后重开，本趟已闭。

## 回归全表（最终字节 vs 基线，同机同夹具）
| 项 | 基线 | 修复版 |
|---|---|---|
| T.load 等值（夹具全字段 sha256＋meta＋is_val，18 字段、is_val 和 598） | — | 与基线**全等**（`equiv_*.json`） |
| k0 短跑 2 步 loss（--max-updates 2） | total 0.9335246682167053 / 0.45114707946777344 | **逐字同** |
| k0 `--screen`（w02 pristine.pt 臂，全 JSON：各 horizon log_loss/auc＋KL） | — | 与基线**零差异键** |
| prove_ties（regress/fixture） | ALL_PASS（duel top 0.24250224919029495 / second 0.23608215419009607） | ALL_PASS，duel 读数**逐字同** |
| test_review_fixes（regress/root） | 9/9 OK | 9/9 OK |
| probe_w_repro（fixed 模式） | 11/11 pass | 11/11 pass |
| proof_real_examiner（真实 opp_master2r） | PASS：identity_ok、冻结集恰 {master2_src.py, md5 `404114a0b5e422516322f62e9ddb6870`, kind code}、快照字节合、exec=快照、eval 4/4、mean_u 5.442151047233296e-07 | PASS，同上，mean_u **逐字同** |
| probe_v06（原版探针，v03v05v06-fix/probes） | mean_u 两轮 `0.20390815759383646`、snapshot_md5 `84face63ef3c617fe8a3fa682e058a79`、recorded md5 换盘后不变 | **逐字同**（全部不变式键与基线及 W04 记录全等） |

## 记录项（只记录、不修，按裁定）
`check_aux_sidecar`（最终文件 def 在 `code/k0_train.py:304`）的两处非预取 fallback——仅当 `prefetched` 为 None 或缺条目时才走，两个 CLI 调用点（训练路、run_screen）恒传 `prefetched=aux_raw`（`_read_aux_bundle` 单读字节），故 CLI 不可达：
- sidecar 按路径重读，`code/k0_train.py:321`：`sc = json.load(open(sc_path))`（else 分支）；
- 标签文件按路径重哈希，`code/k0_train.py:381`：`actual = _sha256_file(p)`（else 分支）。

## 同族点名（扫到、未静默扩大范围，报上级知悉）
- k0_train 训练路 :509 与 screen 路 :1034：`man = json.load(open(man_path))` 在 T.load 之后按路径重读 generation 清单（清单是全场哈希的锚点文件）。属同族「校验后按路径重开」，但不在本趟点名残留（u_raw/label_status）范围内，未动；如需收口建议另趟把 `man` 改由 T.load 的已验来源传递。
- cnet2_train :766 root meta 镜像读：见「修法」保持不变条目，信息性、后有 sidecar 校验兜底。

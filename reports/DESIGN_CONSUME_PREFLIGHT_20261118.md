# 设计：消费端 preflight 与封印 sidecar（合并代消费闭环）

- 凭据：Codex 复核 `reports/DECISION_REVIEW_FOR_LOOP3_20261118.md`（3f28c966）§5；裁定 `reports/RULING_MERGED_GEN_PROVENANCE_SAMPLER_20261118.md`（7283a3f3）§1；验收 `reports/ACCEPT_MERGE_PUBLISHER_20261118.md`（fdf47383）。
- 交付物：`tools/consume_preflight.py`（待审副本，md5 `f95823896e02ba73b934bbfb20756fe3`，578 行）＋本设计件＋自证 12/12（`tests/RESULTS.md`）。
- 纪律：钉版/已验收文件只 import 不改（k0_join_u `9054aadc`、k0_distill_u `06826f59`、merge_annot_generations `f1f5ab9c`，自证前后 md5 不变）。

## §0 口径说明（先行明示）

现行 REGISTRATION.json（已验收封印格式）只记源代**身份**（generation、manifest_sha256、config_sha256、window、mode/rate、源/实抽根数、派生种子），**不记文件系统路径**。本设计据此定口径：源代路径在消费时经 `--source` 给出，preflight 逐个经真 loader 读出其身份后与注册条目按 (generation, manifest_sha256) **绑定核验**——路径只是到达方式，身份只认注册件与源代自身封印，调用方自报不算数。不可达、注册内未提供、提供而未注册（替换）三种情形全部 die 点名。此口径与裁定 7283a3f3 无冲突（裁定未涉路径），且不动已验收 publisher 的封印格式；安全属性与「路径由注册件给出」等价：消费只能在注册清单内的源代上发生，且每个源代都被真 loader 验过。

## §1 缺口（复核 §5 原样）

publisher 的 `--verify` 是自愿动作：合成实例上，删 REGISTRATION 后 publisher verify 拒绝，但真实 `k0_join_u` loader 照样接受该合并代；训练 `k0_distill_u.py:1118–1125` 的 effective 记录只写 joined 代的 generation 与 teacher_config_hash，**不展开来源清单**——裁定 §1 要求的「训练 → 合并代 → 源代」两层引用在消费端没有任何必经核验点。补法（复核原话）：必经消费 preflight ＋封印 sidecar，未必大改钉版核心。

## §2 preflight 设计

### 2.1 位置：两个消费点同一道闸

| 消费点 | 消费者 | target | 附加输入 |
|---|---|---|---|
| (a) join 前 | k0_join_u 消费标注代 | 合并代数据集根 | `--source` 逐源代 |
| (b) 训练前 | k0_distill_u 消费 joined 产物 | joined 数据集根 | `--annotation-root`（joined 绑定的标注代根）＋`--source` 逐源代 |

(b) 的训练端必经性由 §3 的 distill_u 钩子补丁（待裁）转为代码强制；(a) 现阶段由链路卡把 preflight 登记为具名步骤、sidecar 为其产物（与链路卡制度一致）。两处跑的是同一个工具、同一种 sidecar。

### 2.2 核验清单（任一失败 die 点名，不产 sidecar）

1. **target 真 loader 验**：标注代走 `k0_join_u.load_annotation_source`（指针/manifest 自封/meta 封/逐字段 sha/rows/记录闸）；joined 产物走 `k0_distill_u.verify_annotation_generation`（消费端自己的验证器，非镜像复刻），并核 joined 的 annotation 绑定在 manifest 与 meta 两处一致、与 `--annotation-root` 实际加载的代 (generation, manifest_sha256) 全等、joined 的 teacher_config_hash 与标注代一致。
2. **合并链验**（target 侧标注代带 merged_from 时）：直接 import publisher 的 `verify_dataset`（与发布方同一实现，非另写一份）——来源清单哈希复算对三处身份槽、记录闸（并集窗口）、REGISTRATION 存在＋generation/hash/清单 canonical 逐字一致、派生种子复算。
3. **源代可达验**（publisher 从未做的一环）：注册清单逐条目绑定一个经真 loader 验过的提供源代，并逐项对账 config_sha256（且由 config 字典复算）、window、源根数；源代本身是合并代的，其注册链一并递归验。缺一、重复、未注册混入，皆 die 点名条目/代号。
4. **普通代**：`--require-merged` 时 die 点名；默认模式明示「非合并代、无注册链」放行并记录（sidecar 的 merged=null、checks.notice 在案），使训练出处对普通代同样有据可查。

### 2.3 sidecar 格式与封印（CONSUMPTION.json）

落点：target 数据集根（与 REGISTRATION.json 同层，代目录本身永不写）。内容寻址口径与 manifest 一致：`sidecar_sha256 = sha256(canonical(core))`，core 排除 `sidecar_sha256` 本身与 `created`、`locations`（路径/时刻是信息字段，不入身份——同一代在另一路径复验应得同一 sidecar 身份，绑定靠 generation＋manifest_sha256）。

core 字段：

- `format`: `k0-consumption-sidecar/1`
- `target`：实际将加载的代 {kind: annotation|joined, generation, manifest_sha256, total, teacher_config_hash, teacher_weights_md5}（训练时即 joined 代——绑的是真正被加载者）
- `annotation`：标注代同上字段组（kind=annotation 时与 target 同一）
- `merged`：null 或 {teacher_config_hash（清单哈希）, registration: {file_sha256（注册件字节哈希）, format, seed}, merged_from: 来源清单**全文**, sources: 逐源 {generation, manifest_sha256, config_sha256, window, mode, rate, source_roots, drawn_roots, loader_verified, is_merged}}
- `checks`：逐项具名布尔（target_loader_verified / joined_annotation_binding / registration_present / registration_list_byte_identical / merged_hash_recomputed / sources_all_loader_verified / derived_seeds_recomputed / notice）
- `tool`：consume_preflight 自身与三个被 import 件的 md5

已存在 sidecar 同 hash → 复用不重写；异 hash → die（消费记录不许静默覆盖，与「已发布不覆盖」同纪律）。

### 2.4 两跳追溯路径

```
训练产物 eff.data {generation(joined), teacher_config_hash,
                   consumption: {sidecar_sha256, merged_from 全文}}   ← §3 补丁后
  └─ 第 1 跳（sidecar_sha256 绑定）→ CONSUMPTION.json
       {target=joined 代 id+manifest sha, annotation=合并代 id+manifest sha,
        merged.registration.file_sha256, merged_from 全文, 逐源验讫}
       └─ 第 2 跳（file_sha256 / (generation, manifest_sha256) 绑定）
            → REGISTRATION.json（合并代根）＋逐个源代（真 loader 可直验）
```

每一跳都由哈希绑定，没有一跳靠「某次人工 verify 看过」。补丁落地前，eff 里只有第 0 层（generation＋hash），sidecar 已可独立完成 1、2 跳；补丁把第 0→1 跳也钉死。

### 2.5 --check-sidecar 模式

对既定 sidecar 重跑全量核验并逐字段比对新鲜记录与封印 core（先验 sidecar 自封，再比对；分歧 die 点名字段）。用途：训练开工前复核（钩子强形式）、段间衔接复查、验收抽查。它保证 sidecar 不是「某时点的一张纸」：源代被删/被换、注册件被改、代被重指，都会在 check 时现形（自证 N3a/N5 覆盖）。

## §3 distill_u 强制钩子补丁案（草案，未落地，待策略组裁）

对象：`code/k0_distill_u.py`（现行 md5 `06826f59`）。三处改动，均为增补，不动任何既有闸：

**Hunk 1 — 新增核验函数**（置于 `load_annotation` 之后、main 之前）：

```python
def _require_consumption_sidecar(data_arg, ds):
    """Consumption gate (ruling 7283a3f3 two-layer provenance):
    refuse to train on a generation whose mandatory consumption
    preflight record (CONSUMPTION.json, sealed by
    tools/consume_preflight.py) is missing, fails its own seal, or
    does not bind the generation actually loaded. Returns the
    consumption summary sealed into the effective record."""
    root, ptr = _resolve_dataset_root(data_arg)   # mirror of
    # verify_annotation_generation's pointer resolution
    sc_path = os.path.join(root, "CONSUMPTION.json")
    if not os.path.exists(sc_path):
        raise SystemExit(
            f"k0_distill_u data: no CONSUMPTION.json at dataset "
            f"root {root!r}; the mandatory consumption preflight "
            f"(tools/consume_preflight.py) has not sealed this "
            f"generation for consumption; refusing to start")
    sc = json.load(open(sc_path))
    core = {k: v for k, v in sc.items()
            if k not in ("sidecar_sha256", "created", "locations")}
    if sc.get("sidecar_sha256") != _canonical_sha256(core):
        raise SystemExit(
            "k0_distill_u data: CONSUMPTION.json fails its "
            "sidecar_sha256 seal; refusing to start")
    man_sha = _sha256_file(os.path.join(ds["gen_dir"],
                                        "manifest.json"))
    tgt = sc.get("target") or {}
    if tgt.get("generation") != ds["info"]["generation"] \
            or tgt.get("manifest_sha256") != man_sha:
        raise SystemExit(
            f"k0_distill_u data: CONSUMPTION.json binds generation "
            f"{tgt.get('generation')} / manifest "
            f"{tgt.get('manifest_sha256')}, but the loaded "
            f"generation is {ds['info']['generation']} / "
            f"{man_sha}; refusing to start")
    merged = sc.get("merged")
    return {"sidecar": "CONSUMPTION.json",
            "sidecar_sha256": sc["sidecar_sha256"],
            "merged_teacher_config_hash":
                merged["teacher_config_hash"] if merged else None,
            "registration_sha256":
                merged["registration"]["file_sha256"]
                if merged else None,
            "merged_from": merged["merged_from"] if merged else None}
```

**Hunk 2 — 调用点**（现行 L1042 `ds = load_annotation(...)` 之后一行）：

```python
     ds = load_annotation(args.data, q_temp=args.q_temp)
+    consumption = _require_consumption_sidecar(args.data, ds)
```

**Hunk 3 — effective 记录封印来源清单引用**（现行 L1118–1125 的 `data` 块内，`belief` 后增一项）：

```python
                  "teacher_weights_md5": info["teacher_weights_md5"],
                  "belief": info["belief"],
+                 "consumption": consumption},
```

`merged_from` 全文随 eff 封入训练产物（条目数个、体积可忽略），使训练产物自带完整来源清单；sidecar_sha256 与 registration_sha256 同时给出引用锚——「封印完整来源清单引用」两种读法（全文＋引用）都满足。

**钩子强度口径（请裁定二选一）**：主案如上——钩子在进程内验 sidecar 自封＋与实际加载代的绑定（便宜、确定性、挡住「没跑 preflight / 跑错代 / sidecar 被换」）；源代全链复验由 preflight 本体与 `--check-sidecar` 承担（建议链路卡在每段开工时各跑一次）。备选强案：钩子改为子进程调 `consume_preflight.py --check-sidecar`，每次开训全链重验，代价是开训前多一次全量 loader 验（分钟级以下）。主案＋链路卡 check 已闭合全部已见反例，备选只在「sidecar 落盘后、开训前源代可能被动过且无人跑 check」被认为不可接受时才需要。

**连带说明**：钩子对普通代同样生效（preflight 默认模式为其出 merged=null 的 sidecar）——这是有意为之：必经性不靠「先知道是不是合并代」判断（joined 产物本身看不出源代是否合并代），统一一道闸、无旁路。既有普通代训练只需开训前补跑一次 preflight（离线、秒级）。

## §4 与真实代冒烟的关系（闭环条件）

ACCEPT §2 的真实代补验（段 1 首次正式合并前）扩为四步：真实源代合并冒烟 → publisher `--verify` → **consume_preflight（--source 逐源）** → 真实 meta 逐字段对齐读回，回执一并报策略组备案。本工具在合成 fixtures 上自证 12/12 不替代该冒烟；**钩子补丁未裁、真实代冒烟未跑之前，合并 publisher 不称闭环**（与复核 §5 口径一致）。

## §5 自证摘要（tests/run_tests.py，合成数据）

正例：合并代 preflight PASS＋sidecar 封印且绑定值与 loader 直读逐项合（P1）；check-sidecar PASS（P2）；普通代默认放行＋明示（P3）；joined 目标 PASS 且 sidecar 双绑 joined＋合并代（P5）；同 sidecar 复用（P6）。
负例：删 REGISTRATION 必拒（N1，复核反例本体）；篡改注册清单必拒（N2）；sidecar 对另一代 check 必拒并点名字段（N3a）；sidecar 篡改必拒（N3b，自封）；joined 绑定错配必拒（N4）；注册源代未提供必拒并点名代号（N5）；普通代 `--require-merged` 必拒（P4）。

# C02 数据构建修复复核

复核基线：`5463f504048bcca28868ebce0a063ac8d6c51b8e`；本项修复提交 `ea34364651b25f66218220dc67772eebb63caff1` / `639253c0ba9020399f8bc71d2cdc39904dedca61`。对照 `reports/REVIEW_FIXES_20261113.md:53–69` 的原 C02。业务仓库只读；未执行整个 `replica_chain.sh`、任何训练程序、工作流或远端任务。

**结论：原 C02 的坏回放→exit 0→自动链放行路径已经关闭。** 本次 21 项预期检查均通过，包括新增边界的观测断言；“通过”不表示下述边界不存在。仍发现一个独立的数据完整性问题：scanner 对索引中缺失的 replay 静默跳过，standalone builder 会把不完整输入发布为 `partial=false`。当前 replica 链的数量门会拦住这个例子，不能把它写成原自动开训漏洞未修。

## 1. 已通过

证据：`c02-results.json`；可重跑脚本 `recheck_c02.py`。脚本调用真实 `build_top.py`、真实 `cnet2_build.main`、真实 `set_uw.py`；启动侧只抽取 `replica_chain.sh:91–107`，在进入 set_uw 前终止，用 marker 表示 gate 通过，不含训练入口。

| 用例 / JSON 键 | 实测 | 对应源码 |
|---|---|---|
| `healthy`、`healthy_set_uw` | 两局/六行，build=0、set_uw=0、valid_games=2、partial=false、gate=0 | `build_top.py:31–33`；`cnet2_build.py:467–470` |
| `bad_gzip_default`、`bad_gzip_retry` | 原 C02 两局坏一局：两次 build=1；meta 与基础 npy 均移除；坏 shard 不存在且未认证；好 shard 留作重试 | `cnet2_build.py:144–157`、`:398–443`；`replica_chain.sh:82–83` |
| `bad_gzip_allow_partial` | 显式开关后 build=0；`partial=true`、`valid_games=1`、`total=3`；failed_games 含 demo-0 原 ID 与 BadGzipFile 原因；gate=1，显示拒绝 partial | `cnet2_build.py:421–470`；`replica_chain.sh:94–107` |
| `processing_default`、`processing_allow_partial` | 有效 gzip/JSON 内缺 players：默认 build=1 且无 meta；显式 partial 记录 `KeyError('players')`，gate 拒绝 | `cnet2_build.py:145–157`、`:169–171` |
| `restored_retry` | 修复同一 replay 后 build=0、六行、valid_games=2、failed_games=[]；gate 通过 | `cnet2_build.py:361–413` |
| `zero_window_first`、`zero_window_cached_retry` | STEP_MIN=100 连跑两次均 build=0、零行、valid_games=0、partial=false、failed_games=[]，无旧 shard 复活；gate=1 且列出两个零行 ID | `cnet2_build.py:205–215`、`:375–379`、`:422–426`；`replica_chain.sh:98–107` |
| `all_bad_default`、`all_bad_partial` | 全坏默认 build=1；显式 partial build=0、零行、两个 failed_games；gate 拒绝，未把“零行失败”伪装成合法零窗口 | `cnet2_build.py:421–470` |
| `gate_contract_boundaries` | metadata 缺失、旧 schema、不可读 JSON、缺 valid_games，全部 gate 非零且不出现放行 marker | `replica_chain.sh:91–107` |

旧产物清理的准确范围：默认坏 gzip 后 `meta.json` 和当前 builder 基础字段全部被删除。之前由 `set_uw` 生成的 `w.npy` / `uw.npy` 仍保留，因为不在本次默认 `fields` 中；此时没有 meta/y 等主体，不能被当作一个完整数据集。`bad_gzip_allow_partial` 也展示复用目录时这些附属数组仍是六行，而新主体三行。建议以后用发布目录版本化或清除所有依赖行布局的派生数组，但不能因此声称原“完整旧数据集残留”复现仍成立。

## 2. 剩余问题：缺失 replay 被悄悄排除（P2，范围限 builder / 直接消费路径）

**位置：** `code/cnet2_build.py:252–254`；对比失败声明 `:421`、`:467–470`。

**触发条件：** index.json 仍含两局已满足 steps/teacher 条件的合法索引，但其中一局的 replay 文件未下载成功、被移走或丢失。不是合法 teacher/长度过滤，也没有设置 `CNET2_ALLOW_PARTIAL=1`。

**复现：** 脚本 `missing_replay_default` 先构建两局成功的目录，然后删除 `replays/demo-0.json.gz`，保留原 index，再调用真实 build_top。

结果：

- index entries=2，但 build_top 返回 0；真实 set_uw 也返回 0（`missing_replay_set_uw`）。
- 发布 `valid_games=1`、`total=3`、`partial=false`、`failed_games=[]`。
- meta.games 只剩 demo-1；丢失的 demo-0 ID 不出现在任何失败声明中。

**原因与后果：** scan 在 `:253–254` 直接 continue，因此这一缺失输入根本不进入 per-game `errs` 和新失败契约。独立构建任务或只读 metadata/返回码的消费者会得到“成功且非 partial”的不完整数据，无法区分主动筛选和输入缺失。这是修复未覆盖的既有扫描边界，不是此次引入的新回归。

**已有限制：** 当前 `replica_chain.sh:64–74` 对 replay/index 数量先检查，构建后 `:107` 又核对 valid_games==index。本例抽取后一道 gate 返回 1（1 != 2）；完整生产链通常在前一道检查就停止。因此这里没有证明当前自动链仍会启动，也未实际启动。

**建议：** 对原本应参与构建的 indexed replay 缺失，保留原 ID 并纳入失败集合；默认非零；显式 allow-partial 才发布 `partial=true`。合法 teacher/min_steps/重复源过滤与缺失输入分别记账，避免直接把所有 index 行都机械视作必须训练的局。

## 3. 事务发布边界（P3 / 非原 C02 的自动链绕过）

**位置：** `code/cnet2_build.py:324`、`:382`、`:445–454`、`:467–470`；当前失败清理只在 `:427–443` 的 per-game errs 分支。

两个受控例子均返回 1，因此当前 replica 链在 `replica_chain.sh:82–83` 已停止；且真实链在 `:81` 先删除 DS。以下是复用输出目录的独立 builder 边界，不能拿“独立 metadata gate 通过”宣称全链放行。

1. `scan_error_old_output`：成功构建后令 index 为非法 JSON。build=1，但此前 meta 与所有基础 npy 原样留下；新失败不在输出目录状态中体现。
2. `cache_merge_old_output`：成功构建后仅从缓存 shard 删除 maps 字段，保留 y 和其认证行数。`:382` 的缓存命中条件仍满足；`:445` 先用 w+ 打开/覆盖所有合并数组，随后 `:454` 因缺 maps 抛 KeyError。build=1，旧 meta 仍宣称完整六行，数组内容已经变化；不是保留一个完好的旧版本。输入回放本身正常；故障注入的是缓存字段完整性，用来验证发布顺序，不表示线上自然出现频率。

**建议：** 合并前验证完整 shard schema/形状；在独立临时版本目录完成 arrays、meta 和一致性检查，再原子切换发布指针。失败时保留一个明确未发布的候选或完好的旧版本，不能留下旧 meta 配新/半写数组。此项为低优先级健壮性边界，C02 原 per-game 失败门的有效性不受影响。

## 4. 运行方法与证据边界

```bash
CARBON_REPO=/path/to/carbon-league \
CARBON_REVIEW_OUT=/path/to/review-output \
/path/to/python recheck_c02.py
```

依赖当前仓库运行依赖（NumPy、FastCarbon 等）；本次 Python 为 `/workspace/carbon-review-venv/bin/python`。不给 CARBON_REVIEW_OUT 时仅输出 stdout；CARBON_REPO 未设置时从脚本祖先寻找 `code/fastsim.py`。输入和数据产物均在临时目录创建，结束清理，业务源码未写。

fixture 使用真实 FastCarbon 观察与三步决策，index 声称 300 步只是为了通过现有 min_steps=200 扫描过滤。它验证失败/缓存/发布控制流，不是完整赛事回放、不用于估计自然错误率或训练质量。整份检查没有运行训练，也没有把当前线上权重、外部数据盘或生产发布状态当作已验证。

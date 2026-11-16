# X05 复现探针（cnet2_build 发布路径）

对象：`code/cnet2_build.py`，现行 main md5 `8598f26de678450a2dd24e308107416e`（探针每次运行都经
`~/workspace/gh_api.py` 从 GitHub main 实时取字节并在日志中打印 md5；漂移时打 WARN、仍照跑）。

## 缺陷一句话

发布路径用 `_gen_complete(path)` 判断同 id 旧代能否整代采纳：它只做
`json.load(manifest.json)` ＋逐字段 `os.path.exists`，**从不比对 manifest 自带的逐字段 sha256**
（同文件 `load_generation(verify=True)` 却逐个验——发布侧与消费侧口径不一致）。于是：

- 字段齐全但字节已被原地改坏的同 id 旧代 → 被「完整」采纳、指针翻转到它、平面镜像照它刷新；
- 字段缺失的同 id 不完整旧代 → 发布路径 `shutil.rmtree(gen_dir)` 直删后以 staging 改名顶替，
  旧代连同其中完好文件与任何痕迹一并销毁，无隔离、无留痕。

边界（按 r4 定性）：`gen_key` 经回放指纹绑定，**输入漂移会换新 gen id**，不属本洞；
两件探针构造的都是**发布后原地篡改 / 代码漂移型的同 id 异字节**，输入全程不变。

## 构造（两件共用，见 x05_common.py）

- 语料：本 VM 既有真实天梯语料 `~/workspace/carbon-league/replays` 中确定性选取的一局
  （≥200 步、分数齐全、无 producer 溯源键的普通 ladder 记录，按 (steps, file) 排序取首条：
  `012b624dc91fd21633f3cd9d3e1f6da0-1.json.gz`，299 步）。仅在**副本的 index** 里把先手位
  策略显示名改为教师名 `claude-AlphaCarbon-Master` 使 scan 保留该侧；回放字节原样复制。
- 构建：子进程真跑取回的 `cnet2_build.py` CLI（`BUILD_WORKERS=1`），产一代
  `gen-b1983771706c0003`（299 行）。首建约 30 秒，复跑约 10 秒。
- 依赖：python3 + numpy；GitHub API 取码（gh_api）；源语料路径如上。探针不写任何仓内文件，
  全部工件落在 `runs/<probe>/`（vendor/corpus/out/probe.log），每次运行先清空重建。

## probe_adopt.py — 坏代被整代采纳

步骤：真建一代 → 在已发布代内原地翻转 `value.npy` 第 200 字节（大小不变、manifest 不动，
md5 `ba024afd…` → `cc7d22c1…`，与 manifest 声明 sha256 `29994e29…` 矛盾）→ 同输入重跑。

实跑结果（2026-11-14，日志 `results/probe_adopt.log`，15/15 断言 PASS）：

- 重跑 rc=0、报告同一 gen id、current.json 仍指向该代；
- 代目录与坏文件 **inode 均未变**（未重建的直接证据），坏 md5 重跑后原样存活；
- manifest.json 未被重写（没有发生任何重封/复验）；
- **平面镜像 `out/value.npy` 被刷成坏 md5**——坏字节经镜像扩散给 flat 消费者；
- 对照：同模块 `--resolve`（verify=True）对同一批字节报 `generation file hash mismatch` 拒绝——
  数据客观上坏，唯独发布路径不看。

修复后期望（探针应转为 NOT-REPRODUCED）：采纳前逐字段验 sha256（与 meta_sha256 一并），
不符必须 **die 点名**（指出哪一代、哪个字段、期望/实际哈希），不得翻指针、不得刷镜像。

## probe_rmtree.py — 不完整旧代被直删

步骤：真建一代 → 删代内已声明字段 `step.npy` 使其不完整，并在代目录内种哨兵文件
`X05_SENTINEL.txt` → 同输入重跑。

实跑结果（2026-11-14，日志 `results/probe_rmtree.log`，11/11 断言 PASS）：

- 重跑 rc=0、同 gen id；`step.npy` 被重建且 sha256 与 manifest 重新一致；
- **哨兵文件消失**；代目录 inode 改变；连**完好无损的旁观字段** `value.npy` 的 inode 也改变——
  整目录被 rmtree＋改名顶替，而非原地修复或隔离留存。

修复后期望（探针应转为 NOT-REPRODUCED）：不完整旧代应**隔离改名**（如
`gen-<id>.incomplete-<ts>`）保留待查，再发布新代；哨兵在隔离目录中存活即为验收信号。

## 运行

```bash
cd ~/workspace/r4-fixes/x05-probes
python3 probe_adopt.py    # 退出码 0 = 缺陷复现；2 = 未复现（修复后应如此）
python3 probe_rmtree.py
```

## 文件清单

- `x05_common.py` — 共用线束（取码验 md5、语料构造、子进程构建、断言记录）
- `probe_adopt.py` / `probe_rmtree.py` — 两件探针
- `results/probe_adopt.log` / `results/probe_rmtree.log` — 本轮实跑断言日志（归档副本；
  每次复跑会在 `runs/<probe>/probe.log` 重新生成）
- `runs/` — 最近一次实跑的完整工件（vendor 字节、语料、构建产物），可删、复跑自动重建

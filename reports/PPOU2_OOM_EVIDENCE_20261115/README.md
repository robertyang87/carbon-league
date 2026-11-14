# ppou2_m OOM 证据包 — 2026-11-15

主臂 ppou2_m（λ0.95，PID/PGID 91223）死于 2026-11-15 00:41:18，torch OutOfMemoryError。
本包为作战室处置令三件之①的保全证据；诊断与重发方案见报作战室的回执。

## 保全文件（本目录，md5 已核）
- `ppou2_m.log` — md5 65826381f93b5e24164aeed7bb5aa217，43 行，全部是崩溃回溯（无启动行，缓冲未刷出）
- `ppou2_m_config.json` — md5 cc826d8d2141cd9bc351ef1be1826b58
- `ppou2_m_effective_params.json` — md5 12b4cfe19f075ad10426d9d699d69219
- `ppou2_m_league.json` — md5 eadeba50083679353e81ddfed2100853

## ① 死亡确认（L20 直读，2026-11-15 01:27–01:28）
- 崩溃点：`rl2_ppo_train.py:859` `ad.evaluate(vnet, bt)` → `rl2_ppo_cnet2.py:266` `model(b)` → `cnet2_model.py:289 forward` → `encode:181 tattn` → `cnet2_model.py:126` `F.linear`，要 170 MiB，卡剩 111.38 MiB。
- 崩溃时显存表（OOM 报文自带）：m 31.91 GiB；对照臂 c（91224）10.29 GiB；server 91238/91251 各 1.03 GiB；PyTorch 已分配 31.35 GiB，预留未分配 206.84 MiB；expandable_segments 已开。
- 主臂目录现状：全部文件停在 00:06:28；train/eval/warmup.jsonl 均为 0 行；零产出，锚 state 未捕获。
- 死亡复核：01:27:28 与 01:28:43 两次 nvidia-smi 均只剩 c 的两个进程（91224 + 91251），m 已不在。
- 首发崩溃（23:38 pool 编码错）残留：无。OOM 报文只列四个在世进程；当前 GPU 进程表亦只有 c 的两个。

## ② 根因诊断（不对称 31.91 vs 10.29）
- config diff：两臂除 `out` 路径与 `lam`（0.95 vs 1.00）外逐字相同；effective_params.json 逐字相同。λ 只改 GAE 优势加权（CPU 侧），不改 GPU 分配量——λ 不是成因。
- 崩溃位置是更新段小批量循环（mb=768）：`evaluate(net, bt)` 的输出被保留用于算 loss 之上，又叠加 `evaluate(vnet, bt)` 的整前向＋反向。更新段是每轮 GPU 峰值。
- 时序重建：m 00:06:26 启动，死于 00:41:18（约 35 分钟），正处暖机 it=1 的更新段（c 的 it=1：roll 1549s + upd 768s；m 更新段约 00:32 开始）。同一时刻 c 仍在 it=1 的 rollout 段（GPU 只用 server 前向），故只占 10.29 GiB——3 倍差是相位差，不是结构差。
- 可证伪判据已兑现：若 λ/代码不对称是成因，c 应一直停在 ~10 GiB。实际 c 未动过，至 01:27 已长到 36,468 MiB（01:27:28 与 01:28:43 两读相同，稳定），与 m 致死量 31.91 GiB 同量级。
- 结论：**单臂峰值水位 ≈32–36 GiB；双臂更新峰值重叠 ≈64–72 GiB > 44.39 GiB 卡**。原样重发只要更新段重叠就必再崩，不是概率问题。
- expandable_segments：launcher 只设了 `expandable_segments:True`；本次是真实物理 OOM（47.66 GB 总量只剩 11.9 MB 空闲），不是虚拟地址超留——PyTorch 报文里的 reserve 调节建议不适用。预留未分配仅 206.84 MiB，碎片不是故事。
- 附带风险：c 单臂现占 36.5 GiB，只剩约 6.8 GiB 余量；它扛过了 it=1、it=2 两个更新段，但更长的对局批仍可能顶穿。监视在位（anchor pid 15619 / first-round pid 15620）。

## ③ 重发方案（只动一个变量：并发调度）
- 主推：**按轮次边界串行**——同一时刻 GPU 上只跑一个 learner 进程，另一臂停机（进程退出），到轮次边界再 `--resume state_last.pt` 交接。**SIGSTOP 无效**（停住的进程照样占显存），必须是进程退出＋ resume。
- resume 代价已核代码：`rl2_ppo_train.py:563 it0 = stt["it"]`（续轮次）；`:586 warm_iters = resume_vwarm if resume else vwarm`，`--resume-vwarm` 默认 0；`:768 warm = resume_warm or (not resume and it <= vwarm)`——resume 默认不重跑暖机，只剩进程启动＋模型/池加载开销。
- 交接点：只在轮次边界（train.jsonl 新行落地、state_last.pt 已写后）SIGTERM 整进程组，验 GPU 空了再起另一臂；切换粒度按检查点间隔（m 20 轮 / c 40 轮）摊薄重启开销。
- 节奏：每臂约 46+ 分钟一轮；至 11-20 12:00 硬停约 165 轮/臂，m 8 个检查点、c 4 个检查点，时间盒仍成立。
- m 的锚：重发后全新启动，暖机 it=3 落后由现有 anchor watcher 捕获（已同时盯两臂）；c 的锚独立，待其 it=3（约 01:30）。
- c 现在不停（作战室令）。第一次切换放在裁定后的 c 的下一个轮次边界。
- 备选（不主推）：mb 768→384，更新段激活峰值减半，双臂可共存；但改的是备案 pin（需策略组批）且改 PPO 更新动力学。仅当串行不可行时再提。

## 附：对照臂锚捕获（01:29–01:33 直读核验）
- `state_anchor_frozen.pt` 落 01:29，54,613,777 B，md5 **573fb3af6a423f4fe8ac066f23a2b793**（与当时 state_last.pt 同值）；train.jsonl it=3 warm=true，checkpoint sha256 **2cd353f0dc9947599cce6e06879c2462d5d11875cbab7c061bdb54bc49abd513**，与 watcher 记录一致。
- actor==init 已验：`check_anchor_equal.py` 逐键比对 180 个键，最大绝对差 0.0，无缺失（state 里多出的 utility/wdl/margin 六个头是模型自带辅助头，init npz 不存，不影响结论）。暖机 kl=0.0、gnorm=0.0 佐证 policy 未动。
- 补充观测：c 显存 01:27–01:28 稳在 36,468 MiB，01:31:38（it=3 落后）回落至 16,566 MiB——峰值在更新段内，轮间回落。双臂峰值重叠仍是致死机制，不改结论。

## 时间线
- 23:36:42 首发（v2，pool 编码错）→ 23:38 双臂崩
- 00:06:26 重发（v3，SNAP 编码）m 91223 / c 91224
- 00:41:18 m OOM 死（暖机 it=1 更新段），零产出
- 00:45:06 / 01:07:51 c 暖机 it=1 / it=2 落地；01:27 c 显存 36,468 MiB
- 01:26 作战室下处置令；01:31 三件报作战室，待裁重发

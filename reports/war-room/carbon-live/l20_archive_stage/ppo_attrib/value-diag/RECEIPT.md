# 备制回执 — PPO-U 价值侧诊断（PREREG_VALUE 过签版，六点口径）

状态：**备制完成、机制验证全过，未跑任何聚合统计**（备制期唯一计算
是每检查点 ≤8 根的机制验证，只出布尔结论）。策略组已令 A+B 立即
执行——下方一键命令可直接转 L20 线执行，不再等二次确认。

## 一键执行命令（L20 / gpu-l20-6）

```
cd /root/ppo_attrib/value-diag && /root/carbon/.venv/bin/python run_all.py
```

流水线：step 0 五个 state 副本就位复核 + 全部工件 md5 校验（六个
ckpt npz 对 Stage 1 回执、五个 state 对备制记录、roots.npz、
rootset32.json）→ realized_u.py → extract_vhat.py → ab_tables.py。
产物落 `/root/ppo_attrib/value-diag/out/`：`realized_u.npz`、
`realized_u_provenance.json`、`vhat_{74,78,79,85,95}.npz/.json`、
`vhat_82.json`（缺失标记）、`results.json`（A/B 表 + verdict +
C 触发标记）。

## HEAD_MAP 结论（全文 HEAD_MAP.md，逐点张量＋代码行/结构证据）

- **ckpt npz 里没有 critic**：全程 sep_value=1，critic 是独立
  `vnet`，只存于 trainer state .pt；npz 只导出 actor，其值头是
  冻结的 init 工件（value.weight 与 init 逐字同、utility 头冷启动
  常量，全程未变）。v̂ 一律从配对 state 的 vnet 读；配对锚定已验：
  四个可用 state 的 net 与同号 ckpt npz 186/186 数组全等（74 由
  Stage 1 验），提取脚本内嵌该断言、执行时复验。
- 读出分界**在 it79 不在 78**：74/78 = vnet tanh 值头（换头前
  adapter 的 evaluate 根本没有 utility 分支），v̂ = (tanh+1)/2；
  79 起 = vnet utility 头，v̂ = u 本身（adapter 约定 v = 2u−1）。
  行为证据：vnet 值头 md5 在 ≤78 持续变、79 起冻结；utility 头
  ≤78 冻结、79 起每轮变。
- r19（82 起）utility-source=checkpoint 实证闭环：trainer 的来源
  哈希方案对 it81 备份的 vnet 复算，与 r19 重拉日志记的 head
  sha256 `a592d452…`、critic sha256 `bb700cea…` 逐字复现——82 的
  critic 是 it81 vnet 续训，读出仍为 vnet utility 头，无换源。
- **卡点（唯一）：ckpt_82 的 v̂ 数值工件不存在**。it82 终局 trainer
  state 无任何存留：备份点只有 74/78/79、85 是冻结副本、滚动
  prev1 冻结在 it81、state_last 已是 it96，全盘 .pt 扫描点名完毕。
  缺的证据就是一份内嵌 it=82 的 trainer state .pt（含 vnet）。
  脚本处置：82 写 ARTIFACT_MISSING 标记、其余五点照跑；A 表 82 列
  与 79→82/82→85 单步变化不可观测，但 **r19 步组合计分享率按
  m85 − m79 望远镜式精确可算**（带符号和恒等），B 表判读线不用
  82。若策略组裁定必须补点，需要的工件已确认不存在——请其就
  「五点＋组合计」口径或放弃 82 点裁定。

## 脚本清单（本地 ~/workspace/ppo-u-attribution/value-diag/，已同步 L20 /root/ppo_attrib/value-diag/）

- `common.py` — Stage 1 同款批次组装、critic 构建（vnet＋配对 npz
  meta、net≡ckpt 恒等断言）、md5 钉版表、HEAD_MAP 读出表。
- `realized_u.py` — 按 PROVENANCE.md 口径构造每根 realized U
  （源局已记录实际对局的终局、席位定向 margin → 平台公式；与
  `cnet2_build._u_of_margin` 逐（局，席）断言等值；四组自检硬断言，
  含双席和=1 与 margin 符号自洽）；逐局 steps 总数实读回放供 B 表
  残局距离。
- `extract_vhat.py` — 六点逐点 v̂＋所读张量名落盘（82 出缺失标记）。
- `ab_tables.py` — A 表（全体＋席位×步段的 bias/MAE/Pearson/
  Spearman）、B 表（逐点符号不一致率、残局/非残局、剔除计数），
  verdict 按过签线机械打标（指认（换头步）/指认（r19 步）/证伪/
  中间，及未签的量级间隙如实记「无判定」不自创线）；C 触发标记
  只记旗（C 本身不备不跑；种子 990400–990499＋不可构造骑缝已写
  进脚本注释与 results 的 C_note）。
- `mech_check.py` — 备制期机制验证脚本（下述结果即其输出）。
- `run_all.py` — 一键编排（含 step 0 拷贝/md5 门）。
- 文档：`HEAD_MAP.md`、`PROVENANCE.md`、本回执。

## 机制验证结果（mech_check，≤8 根/点，只报机制结论）

- ckpt_74：PASS — 读出 tanh，vnet[value.weight, value.bias]；形状
  (8,)、有限、值域 [0,1] 内、非恒定、同批前向可取策略 logits、
  state.net≡ckpt_74、内嵌 it=74。
- ckpt_78：PASS — 同上（tanh / value 头；恒等对 ckpt_78）。
- ckpt_79：PASS — 读出 utility，vnet[utility.weight, utility.bias]；
  其余各项同上（恒等对 ckpt_79）。
- ckpt_82：ARTIFACT_MISSING — state 槽缺失，与 HEAD_MAP §6 的盘点
  结论一致（读出口径已证为 utility / vnet utility 头）。
- ckpt_85：PASS — utility / utility 头（恒等对 ckpt_85）。
- ckpt_95：PASS — utility / utility 头（恒等对 ckpt_95）。
- 总：MECH_CHECK ALL PASS。

## 纪律

L20 训练目录零写入（state 只读复制到 /root/ppo_attrib/value-diag/
state/，源文件未动）；未起训练/eval 对局进程；全量 A+B 未在本趟
运行，等 L20 线照一键命令执行。

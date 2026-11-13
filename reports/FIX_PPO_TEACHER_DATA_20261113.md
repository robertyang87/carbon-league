# PPO 恢复、教师信息集合与镜像身份修复

2026-11-13。实现基线 `fd2e791d436a911a834d17657db5119a84ae215d`；
整合了后续 `fa7b41c` 的双线规则与 `4ac6318` 的 30 轮预算记录。代码与脚本的精确 SHA256 见
[source-manifest.json](review_evidence/20261113_ppo_teacher_data/source-manifest.json)。

本轮修复三处已复现的问题，并订正文档中会影响实验结论的表述。
代码验收通过；没有远程部署、重建真实数据集、启动训练或正式教师收益实验。
“修复通过”不等于新策略已获得棋力增益。

## 1. PPO：以恢复后有效 critic 为准

原路径先检查 init 的 utility 键，再恢复 checkpoint，存在两个方向的问题：
init 无头而 checkpoint 已有训练头时误拒绝；init 有校准头但旧 checkpoint
覆盖该头时，init 检查不能证明实际 critic 的来源。resume 还会跳过预热。

修复在 `code/rl2_ppo_train.py`：

| 参数 | 行为 |
| --- | --- |
| `--utility-source auto` | fresh 使用 init；resume 只有目标/读头/尺度/独立价值网模式元信息匹配才自动接受 checkpoint |
| `--utility-source checkpoint` | 显式接受旧 checkpoint 的实际 critic；独立模式检查 vnet，共享模式检查 net；缺头、非有限值或无效形状拒绝 |
| `--utility-source init` | resume 时完整重置独立 vnet 并清其 optimizer；不只换头，不覆盖 actor；共享 critic 模式拒绝 |
| `--resume-vwarm N` | 恢复后显式预热；独立模式只训练 vnet，共享模式只训练选定价值头，策略保持不变 |

effective_params 与 checkpoint 保存训练契约及来源；记录源文件、完整有效
critic 和 utility 头 SHA256。`calibration_verified=false` 明确表示键存在
或来源声明不等于通过校准实验。正常同模式恢复保留参数及 optimizer 状态。

**L20 旧 checkpoint 的迁移**：保留原 init，不再需要制作“两键合并 init”。
在原任务参数上明确追加或替换：

```text
--resume <state_last.pt> --reward u --gamma 1 --rscale 1
--value-head utility --utility-source checkpoint
```

需要切换后预热时显式增加 `--resume-vwarm N`，N 由既定实验预算确定。
旧文件没有 training 元信息，默认 auto 会拒绝含糊恢复，须采用上面的明确
来源声明；新同契约 checkpoint 后续可使用 auto。若要引入另一份校准 critic，
则使用完整匹配的 init 和 `--sep-value 1 --utility-source init`。

仍有的范围限制：默认独立 critic 没有随 actor NPZ 导出，不能将该 actor
包内的 utility 两键直接视为训练后的评论家。gamma/rscale 自检仍按既有
约定警告并记录，正式 U 任务必须核对实际参数。

## 2. 教师：阻断真实对手私有输入

原 v2 双世界测试固定了对手 x/Feat，仅证明不直接读取模拟器隐藏碳图。
实际调用方会重建对手真实私有观察，因此仍存在另一条信息通道。可达反例中，
我方整个合法观察历史与公共状态相同，仅对手观察不同，得到
`max|ΔQ|=0.0014981473`、`max|Δq|=0.0043435661`。

修复将 `sample`/`mean` 标记为 **v3**：

- 从我方合法 Feat 构建每个假设世界，再从该世界的对手观测初始化对手模型。
- 对手根动作在我方候选生成之前采样，场景及随机数仍由候选共享。
- 分支继承我方真实历史、我方 guard，以及场景内的对手模型历史。
- 完全忽略调用方真实对手 x、aux、Feat、动作转换 shim 与 guard 快照。
- 旧 oracle/v1 路径保留为诊断；gate 与 pilot 绑定当前教师版本。

修复后，上述反例在两席、sample/mean 四种组合中 Q/q 差均严格为 0；
毒化所有对手私有输入也不被读取，输入世界与 Feat 不被修改。

**这是一个明确的对手历史近似**：模型在场景根初始化，之后随分支演化，
没有恢复对手根前的真实私有信念。信息流正确不保证教师更强。旧 v2 pilot、
方差、N 与收益不能直接认证 v3，需重新测量。正典 heads v2 未在本机复跑；
本次神经网络测试使用 b128 权重与固定种子的测试 utility 头。

## 3. builder：策略身份与显示名称分离

原 `_scan` 丢弃 producer identity/mirror 声明，随后以同名判断同策略自博。
同一权重设置两个合法教师名称便能漏过第二道检查。

现按 producer 的单权重、配置及代码身份识别，校验 index、replay 与 corpus
identity，保留 diversity、开局种子和镜像声明；交换席位或仅保留一个教师侧
也不会绕过检查。身份矛盾不可通过演练 override 放行。

缺少足够 provenance 的旧自博明确标记 legacy/unverified，正式建集拒绝；
显式 `CNET2_ALLOW_MIRROR=1` 只允许演练。普通梯子对局不因同名被推定为
同权重自博。等分仍是退化签名而非逐行镜像证明，原 >50% 阈值未修改。
producer 孤儿回放恢复也保留已有的 open_seed、mirror_signature。

兼容影响：producer 源码 hash 变化会改变生成身份；builder 源码参与 shard
fingerprint，旧 shard 缓存会重建。本轮没有在真实 ds_gen1 或生产机器执行
重建。恢复生成前必须核对新旧 identity，不应把配置不符的旧产物当作已完成。

## 4. 验收与证据

以下均在本修复的最终代码文件上执行，来源 manifest 绑定字节版本：

| 验证 | 结果 | 证据 |
| --- | --- | --- |
| 原 PPO U 奖励/读头验收 | 18/18 | [日志](review_evidence/20261113_ppo_teacher_data/ppo_u_mode.log) |
| PPO 恢复与真实训练步骤 | 23/23 | [日志](review_evidence/20261113_ppo_teacher_data/ppo_resume.log) |
| 原双世界与信念测试 | 37/37 | [日志](review_evidence/20261113_ppo_teacher_data/teacher_twoworld.log) |
| 新可达信息集合、毒化与双席测试 | 20/20 | [日志](review_evidence/20261113_ppo_teacher_data/teacher_information_set.log) |
| oracle/v1 两席、带/不带历史兼容 | 4/4 逐位一致 | [日志](review_evidence/20261113_ppo_teacher_data/teacher_legacy.log) |
| 镜像身份与建集准入 | 15/15 | [日志](review_evidence/20261113_ppo_teacher_data/mirror_identity.log) |
| gate/pilot 版本与正式资格 | 3/3 | [日志](review_evidence/20261113_ppo_teacher_data/gate_identity.log) |
| 原 S01 终局与守卫反例 | 17 个边界案例及到时/灭绝拒绝、双席隔离通过 | [日志](review_evidence/20261113_ppo_teacher_data/terminal.log) |

PPO 新测试实际运行 main→rollout→GAE→backward→checkpoint，两种 critic
模式都验证预热期间 policy 和原策略 optimizer 的动量/LR/step 逐位不变，
utility 确有更新。镜像测试跑真实 `_scan→main` 准入路径，正例在 worker
边界停止，检查旧 generation 指针不变；不等于完整模拟器建集验收。

可复跑入口位于 `automation/acceptance_s1/`：

```bash
python test_ppo_u_mode.py
python test_ppo_u_resume.py
python test_teacher_v2_twoworld.py /path/to/weights.npz
python test_teacher_information_set.py /path/to/weights.npz
python test_teacher_gate_identity.py
python test_mirror_identity.py
```

历史双世界脚本保留 v2 文件名，内部验证当前 v3。正典模型运行需传正典
权重，并对双世界脚本加 `--canon`。终局回归入口在证据目录；只移除了
旧归档中“guided_match 自 7b025 未变”的过期文件身份断言，保留全部行为断言。

## 5. 文档与后续实验口径

`STAGE1_TEACHER_GATES_20261113.md` §18 撤回“对手私有 Feat 非泄漏”的
旧定性；§17 同步收紧三项实验口径：PPO critic 准确性须验证；监控挑选与
独立最终验收分开；v3 pilot 按来源局聚类估算样本量，不能用根数冒充独立局。

硬 argmax 引导收益不作为 q 蒸馏上限。pilot 点估计低于 0.02 且区间跨零
记为证据不足，可按预算暂停，但不据此证明方法无效。旧 1,187 秒/根是完整
续弈评估成本，旧 492 局/约 48 小时不能直接作为 v3 终局实验报价。

正式平台资源认证、正典 v3 收益、冻结裸网 ΔU 与机器部署仍是后续任务，
本报告不将其计为关闭。

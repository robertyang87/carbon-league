# 修复复核证据（2026-11-12）

被测源码：c64bbeda8a00b2f4d79ace589f201bab3cca5163。
报告：[REVIEW_FIXES_20261112.md](../../REVIEW_FIXES_20261112.md)。
原始报告和原始证据另存，不覆盖。

## 文件

| 文件 | 内容 |
|---|---|
| commit-syntax-summary.json | 22 笔修复提交及 11 个 Python 修改文件语法检查 |
| simulator_fix_review.py / simulator-fix-results.json | 原版 env.run 与 eval 的 299 步/灭绝/账本对照，search leaf 原样函数检查 |
| planter-ledger-residual.json | 两步种植员招募自撞后的分类账本记录 |
| review_ppo_fixes.py / ppo-fixes-results.json | 真实 resume 代码块、rollout、打包轨迹/GAE、奖励和温度角色检查 |
| recheck_data.py / data-results.json | 真实 builder/merge 的窗口、教师表、空 shard、index 标签与 split 检查 |
| docs-results.json | 基线 Git tree 文件、入口指针与 eval hash 核对 |
| ecs_checks.py / ecs-results.json | 纯本地 API mock；未连接云 API、未释放实例 |
| runtime/ | 当前 packer、历史 b128 固定样本的错误闸、纯网计时、TTA 与采样对照 |

## 运行

Python 3.12、NumPy 1.26.4、PyTorch 2.5.1 CPU（PPO 探针需要）及仓内原版环境依赖。
可用独立虚拟环境安装依赖，不要更改生产训练环境。
完整检出仓库 code/ 和 tools/；runtime 还需要已跟踪的历史包与修复前 Git 对象，详见其 [README](runtime/README.md)。

脚本优先使用 CARBON_REPO 环境变量；否则从脚本父级向上寻找 code/fastsim.py。
以下在本证据目录运行，输出保存到临时目录，不覆盖历史 JSON：

```bash
export CARBON_REPO=/path/to/carbon-league
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p /tmp/carbon-fix-recheck
python simulator_fix_review.py > /tmp/carbon-fix-recheck/simulator.json
python review_ppo_fixes.py > /tmp/carbon-fix-recheck/ppo.json
python recheck_data.py > /tmp/carbon-fix-recheck/data.json
python ecs_checks.py > /tmp/carbon-fix-recheck/ecs.json
```

种植员分类账本复现：将 simulator_fix_review.py 中两步 suicide 策略的 RECCOLLECTOR 改为 RECPLANTER，读取 ledger_extra.pol 的 pla_rec/pla_dead/pla_dead0/rec60_p/e_prec25，均应为 2，基线实得 1。只改探针副本，不改业务代码。

路径移植后的脚本只做语法/路径校验；原始动态结果由相同被测函数在同一基线采集。脚本没有包含完整训练、线上性能认证、最新发布包或部署检查。resume 使用真实 AST 代码块而非启动完整训练器；search leaf 测试也执行原样提取的函数。构造边界状态的测试证明函数语义差异，不量化对局发生率。

**探针退出 0 只说明采集完成。** 某些输出刻意记录尚未修好的反例，不是统一验收测试套件。应根据报告的预期行为判定。未来修复后请记录新源码 SHA 和新结果，保留此处快照。


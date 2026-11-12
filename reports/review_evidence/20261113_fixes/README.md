# 第三轮复核证据

源码基线：14d53320b20d58bdb82676f7fa866b8e11a55062。
日期：2026-11-12 UTC / 2026-11-13 北京时间。
总报告：[REVIEW_FIXES_20261113.md](../../REVIEW_FIXES_20261113.md)。

## 范围与入口

- [ppo/README.md](ppo/README.md)：终局奖励、SELF 镜像、resume、actor 角色、混合长度与推理次数。
- [simulator/README.md](simulator/README.md)：原版 env.run 对拍、search leaf 与 sweep 终局深度反例。
- [data/README.md](data/README.md)：真实 builder/merge、缓存与 split、坏回放仍通过开训前检查。
- [runtime/README.md](runtime/README.md)：当前打包器、历史 b128 样本、采样/TTA/cache/考试错误门。
- ecs_checks_round3.py：复用上一轮已入库的纯 mock 脚本，检查当前 ECS 工具；不访问云 API。
- [method/math-evidence.json](method/math-evidence.json)：候选 CE 反例、tanh 数值、溢出箱信息缺失、PCR 预算与文件截断记录。
- [syntax-and-commits.json](syntax-and-commits.json)：本轮变更提交和 8 个 Python 文件语法检查。
- ecs-results.json 与 ecs-extra-results.json：原四个用例已通过，新增缺 InstanceId 的结构边界仍误报。

## 使用

Python 3.12、NumPy 1.26.4、PyTorch 2.5.1 CPU（PPO），以及仓内原版引擎依赖。
设置 CARBON_REPO 为仓库根路径，OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS=1。
各脚本也可从自身父级向上查找 code/fastsim.py；具体临时输出变量见子目录 README。
运行时请使用新临时输出目录，保留此目录内的历史结果。

证据中的绝对路径是当次采集语境；整理后的脚本已改为可移植路径，只进行了静态语法/链接检查。
runtime 的计时循环只在内存 AST 缩短为两步，验证统计和退出码，不能用于性能认证。
构造状态、固定策略输出与数学反例均已在报告标注；没有用它们估算线上发生率。

## 结果解释

这些文件既包括通过断言，也包括残留缺陷的反例，不是统一“全绿即上线”的套件。
尤其某些采集脚本正常退出，仅意味着它成功记录了错误行为。
新 KataGo 方案尚未实现，数学反例审查的是设计承诺，不能当作现有生产模块的测试失败。

本轮未启动训练、部署、发布策略或操作云资源。对 builder 开训链的检查只实际运行构建与加权；
后续是否会放行由 replica_chain.sh 的代码路径证明，未启动该脚本。


# Terminal-U 切换清单（L20 线记账）

切换执行：2026-11-13 20:49:02，it74 边界，PID 28238（--reward u --rscale 1.0 --gamma 1.0），
fork 快照 state_pre_termU_it74.pt（md5 683d87588ae597920fe20a8c095c6d14）。

## 策略组 Review 补项（2026-11-13 21:10 转入，本线处置口径）

1. **训练/部署 guard 不一致**（学习方无 guard、部署包 GUARD=True）
   - 记账：snipe_n1 训练分布为无 guard 行为；快照打包部署时按正典 CONF 带 guard。
     在训 EVAL 测的是无 guard 裸网；定标测的是带 guard 的部署件——两读口径不同，引用时必须标明。
   - 处置：Tier 1 读数（it85/it90 对 it74 分叉锚）出报时显式挂此差异，不把 guard 效应
     混进 U 制归因；本轮对照内不改任一侧（guard 开关本身是变量，中途改会污染对照）。

2. **evaluate 只返回 tanh 头**（U 制下价值应读 utility 头）
   - 记账：与去 clamp（u+γ=1.0 同域不裁）、γ=1.0 同趟处置。
   - 处置：EVAL 的 mean_u 列由逐局终局边际按 U 公式算（结果口径），不经价值头，
     故重新锚定读数不受此项影响；今后凡用价值头读数（诊断/ΔU 类）一律改读 utility 头，
     不用 tanh 头。PPO 自身 vnet 在 U 制下回归 U 尺度回报，与此一致。

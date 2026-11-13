# 截止 57557a8：正式 heads 训练器与校准指标补充

固定基线 `57557a83a73cd4d6b26b56ebebb4c0df8e13a155`。新增 `code/stage0_train_heads.py`。只提取真实 `evaluate/targets` 做小型内存fixture；没有调用main、训练或写模型。脚本 `check_latest_calibration.py` 支持 `--repo/--out/--ref`；证据 `latest575-calibration-evidence.json`。

## 1. Huber问题现已进入会保存权重的训练入口

`stage0_train_heads.py:20/:81` 导入并调用 `stage0_smoke_heads.head_losses`，`:89–92` 导出训练后的state_dict到npz。被复用文件内容与62e审查一致：`:49`仍是margin的`Fn.smooth_l1_loss`，标签仍按旧float16 tanh逆变换。

因此此前“仅smoke、不保存”的范围描述**只适用于62e版本**，不适用于本轮最终575截止点：已经有正式头部训练入口沿用Huber并保存权重。margin仍在model中宣称E[M]，与设计MSE条件均值目标不一致。62e数学fixture仍直接适用（恢复标签均值1006.066，Huber最优点约111.111）。不能称文档M01没改；应称后续实现没有遵守已修目标。未证明有人已运行该入口或把产物接到线上。

## 2. 新确定问题：按真实target分箱却作为预测校准指标

位置：`stage0_train_heads.py:43–48/:53`。代码用 `bins=clip((ut*10).astype(int),0,9)`，然后累计 `sum_b p(b)*abs(mean(u|b)−mean(ut|b))`，输出名为`val_u_calib_binned`。

预测均值校准要检验 `E[U | predicted_u]=predicted_u`。按真实U分箱得到的是另一种“结果区间内的预测残差”，不是该校准条件；数值会同时出现误报与漏报。真实evaluate函数（仅CT.loader和net用内存fixture替代）得到：

| fixture | 实际U标签 | 预测u | 当前val_u_calib_binned | 按预测分箱的校准偏差 |
|---|---|---|---:|---:|
| 均值已校准，信息不足 | .09997284 / .90002716，各一条（合法正常局分差±1386.294经过现有标签管线） | .5 / .5 | **.40003** | **0** |
| 两组都预测错，残差相消 | .5 / .5（合法平局） | .2 / .8 | **0** | **.3** |

第一例模型对所有输入预测正确的总体条件均值.5，但按结果分箱错误报出约.4“校准误差”；第二例真实均值都为.5，.2与.8明显各自失准，却在同一个真实target箱内抵消，报告0。它不是普通采样误差或小样本置信度问题，而是分箱变量选错后指标测量了不同对象。

**最小修复：** 可靠性图/分箱校准按预测u分箱，固定边界并同时输出每箱n、mean(u)、mean(U)。现有总体偏差与MSE可以保留；若结果分层残差有诊断用途，可另命名并保留，但不能用它替代预测校准或给bare价值头签发校准通过结论。加上述两个fixture作为回归例即可，不需要真实训练。

新evaluate遍历整个验证loader，比62e smoke只取首batch有所改进；但同一种反推标签依旧不能验证原始终局标签真实性、正常/异常路由或bare/mixed续弈匹配。该限制与本节分箱错误分开，不把“验证集更完整”否定掉。

# 最新 62e0dee 的 heads 增量复核

固定增量基线：`62e0dee16419725ece6f18a22f095c83bb6fcd0f`，隔离目录 `/workspace/carbon-review-round5/latest`。本节与原 `98ffadfa` 的文档验收分开：原 M01/M02 文字修复属实；新的 heads 原型来自 `2e0ffdb`，损失实现出现偏离。仅源码阅读及两次小张量反向求导，没有 optimizer step、数据集训练或业务修改。复现：`check_latest_heads.py --repo <repo> --out <new-dir>`；环境需 NumPy/Torch，证据 `latest-heads-evidence.json`。

## 新增确定项：margin 仍叫条件期望，实际 smoke 用 Huber

- `code/cnet2_model.py:164,209–212` 把新 margin 输出定义为 E[M]/1000。
- `code/stage0_smoke_heads.py:49` 调用 `smooth_l1_loss`（默认 beta=1），而非已经在设计修正的未裁标签 MSE；`stage0_heads_patch.py:8` 也直接写“expectation…Huber”。
- 与 `katago_architecture_design.md:99–100,113` 的 MSE 条件均值/Huber仅诊断约定不一致。这是**新原型损失的实际偏离**，不是推翻此前文档已修的结论。

提取该提交原样 `head_losses`，使用常量输出 stub，输入9个M=0与1个M=10000（按真实builder存float16 tanh，再按smoke反推）。恢复标签的均值为1006.066；下表两次头输出分别是0.111111与1.006066，梯度针对这个缩放输出：

| 常量 margin 预测（原始单位） | 对缩放头输出 m_k=M/1000 的实际总loss梯度 | 含义 |
|---|---:|---|
| 111.111 | −1.4e−9，约为0 | Huber稳健最优点 |
| 1006.066 | +0.200000 | 梯度下降会离开标签均值、向下推 |

这不是浮点微差，而是损失的统计目标不同。若 smoke 是后续正式头训练的模板，建议在继续接教师前改为MSE；若有意保留Huber，就必须更名稳健诊断头，不把其输出叫E[M]或做均值校准验收。单纯冻结trunk不消除该目标问题；目前smoke不保存权重，也未证明此头已经用于生产决策。

## 旧 tanh 标签反推只能是近似通路验证

`cnet2_build.py:58` 存 float16；`cnet2_train.py:64–65` 才提升成float32；smoke:40再做 `3000*atanh(clamp(v,±.999))`。提升精度不能找回量化信息，clamp又产生约11400.6的硬上限：

| 原M | 实际恢复M | 误差 |
|---|---:|---:|
| 10000 | 10060.660 | +60.660 |
| 13320 | 11400.623 | −1919.377 |
| 20000 | 11400.623 | −8599.377 |

smoke:9–11已如实说明尾部不能恢复，故不把这件已声明限制单独升格成未披露事故；但它不能成为“未裁raw-margin期望头已验收”的证据。不可逆不仅发生在约11400以外，float16在10000时已经有误差。相反，示例中的正常局U误差只有约1e−6至6e−6，因为sigmoid尾部已接近饱和；**不能把raw-margin大误差直接宣传成同量级U错误或证明效用头必坏。** 正式标签应从真实终局reward/结果元数据恢复M、WDL、outcome_kind和U，或重建dataset；不要把旧value列改名raw。

此外，smoke仅从M符号导出WDL并一律使用正常局软分，没有异常类型。相同M=100在正常胜局U=.762490，单方故障获胜U=1；只靠value字段无法区分。未查明ds_m3s2是否含异常局，故只是正式数据契约缺口。若用作正常局smoke，应明确并核查数据过滤；正式训练需outcome mask/真实U。

## 已合理的部分与 smoke 边界

`utility`的sigmoid+直接U MSE、WDL三分类CE、无界线性margin输出，形式上符合正常局目标；三者读取ctx而不把未来标签输入actor。当前只是一个utility头，没有bare/mixed双路；旧普通forward仍返回旧value，单独forward_heads才返回新头。将单头用于独立接口smoke合理，但不能声称已经实现匹配续弈的bare校准、双头路由或完整教师闭环。

smoke训练损失有限/下降与U落[0,1]最多验证可训练通路；sigmoid的范围是结构保证，不是校准或对战效果证据。其验证只取第一个batch（smoke:107），且样本仍用同一种近似反推标签，不能独立证实标签正确、尾部均值、bare续弈价值或裸网ΔU。主BC新头梯度、严格加载、smoke退出码和NumPy打包兼容由训练链代理另查，本节不重复计项。

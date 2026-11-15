"""叶接口适配（S0b 原型）。

统一叶接口：callable (leaf_state) -> float，返回 U 尺度（部署口径
reward = 2U - 1，U ∈ [0, 1]）。两种叶：

  ① HeuristicLeafAdapter —— 现行启发式叶（对照/低成本臂）。
  ② VNetLeafAdapter      —— 完整 PPO critic（学习叶，骨架，验收前禁用）。

超域纪律：映射后 U 若落在 [0, 1] 外，**不静默裁剪**——返回原值并置
last_out_of_range 标志、累加 out_of_range_count 供遥测/验收读取。
静默裁剪会把量纲错误伪装成正常值（旧嫁接线 2000·z 的教训之一）。

有限性纪律（SI06，与超域检查分离）：已声明的原始分差尺度可以超出
[0, 1]，但不可为非有限值。NaN / ±Inf 由 last_nonfinite 标志与
nonfinite_count 单独记账，并**原样返回**给比较器——由比较器在整轮
收齐后判该轮无效（abort_reason="nonfinite_leaf"）。适配器不把非有限
值混进超域计数，也不把它裁剪/映射成合法端点（尤其 vnet 的 tanh：
Inf 经 tanh 会变成合法的 1.0，必须在 tanh 之前拦截）。
"""

from __future__ import annotations

import math
from typing import Callable, Optional


class HeuristicLeafAdapter:
    """包装现行启发式叶函数，经 affine 映射到 U 尺度。

    heuristic_fn: (leaf_state) -> float，值域为启发式叶的原始尺度。
    affine 参数 (a, b)：U = a * x + b。

    注意：(a, b) 默认 (1, 0) 只是占位。正式参数必须经本线校准流程
    测定并注册（S0/S3 口径）；**不得从 Echo3 的拟合 affine 参数迁移**
    ——教师/主体不同，迁移参数无身份依据（Codex 设计复核与预注册
    均已钉死此条）。
    """

    def __init__(
        self,
        heuristic_fn: Callable[[object], float],
        a: float = 1.0,
        b: float = 0.0,
    ) -> None:
        self._fn = heuristic_fn
        self.a = a
        self.b = b
        self.last_out_of_range = False
        self.out_of_range_count = 0
        self.last_nonfinite = False
        self.nonfinite_count = 0
        self.calls = 0

    def __call__(self, leaf_state: object) -> float:
        self.calls += 1
        u = self.a * float(self._fn(leaf_state)) + self.b
        # 有限性与超域分离：非有限值单独记账、原样返回（比较器判轮无效）；
        # 有限但超 [0,1] 只记超域（原始分差尺度的预期现象）。
        self.last_nonfinite = not math.isfinite(u)
        if self.last_nonfinite:
            self.nonfinite_count += 1
            self.last_out_of_range = False
            return u
        self.last_out_of_range = not (0.0 <= u <= 1.0)
        if self.last_out_of_range:
            self.out_of_range_count += 1
        return u  # 返回映射后原值，不裁剪；超域由标志/计数报警


class VNetLeafAdapter:
    """完整 PPO vnet 叶（骨架）。

    身份 = (checkpoint, state_md5)（MS04 裁定）。U 映射固定为
    U = (tanh(Z_vnet) + 1) / 2，其中 Z_vnet 为 vnet 原始输出；
    reward 口径 = 2U - 1。

    backend: (leaf_state) -> float，返回 Z_vnet（tanh 前）。
    backend 未注入时调用即抛 NotImplementedError —— L-vnet33 四项
    验收（身份/行为匹配/校准/成本）未过前禁用，不许用占位实现
    悄悄顶替真叶跑出「结果」。
    """

    def __init__(
        self,
        checkpoint: int,
        state_md5: str,
        backend: Optional[Callable[[object], float]] = None,
    ) -> None:
        self.checkpoint = int(checkpoint)
        self.state_md5 = str(state_md5)
        self._backend = backend
        self.last_out_of_range = False  # tanh 映射恒在 (0,1)，保留字段对齐接口
        self.last_nonfinite = False
        self.nonfinite_count = 0
        self.calls = 0

    def identity(self) -> dict:
        return {"checkpoint": self.checkpoint, "state_md5": self.state_md5}

    def __call__(self, leaf_state: object) -> float:
        if self._backend is None:
            raise NotImplementedError(
                "VNetLeafAdapter backend 未注入：L-vnet33 四项验收未过前禁用"
            )
        self.calls += 1
        z = float(self._backend(leaf_state))
        # 有限性必须在 tanh 之前查：tanh(±Inf)=±1 会把坏值压成合法端点。
        # 非有限 z 原样返回，比较器据此判整轮无效（nonfinite_leaf）。
        self.last_nonfinite = not math.isfinite(z)
        if self.last_nonfinite:
            self.nonfinite_count += 1
            return z
        return (math.tanh(z) + 1.0) / 2.0

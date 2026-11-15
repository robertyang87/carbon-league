"""只读桥接：钉版搜索源 cnet2_search.py（S0b 原型）。

从 GitHub 仓库 robertyang87/carbon-league 的 main 拉取
code/cnet2_search.py，md5 必须为 94f7fb64（钉版身份），不合即报错停。
副本落 proto/vendor/cnet2_search.py —— 只读副本，本桥从不回写仓库、
从不修改副本内容（仅在文件头加来源注释时以注释行前缀注入，不动代码行；
见 fetch_vendor 的实现说明）。

用途：S0b 比较器原型以 importlib 从 vendor 副本加载钉版模块，
复用其启发式叶与搜索入口，避免凭记忆重写或漂移到未钉版源码。
"""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import os
import sys

sys.path.insert(0, os.path.expanduser("~/workspace"))
from gh_api import get_file  # noqa: E402

OWNER = "robertyang87"
REPO = "carbon-league"
REMOTE_PATH = "code/cnet2_search.py"
EXPECTED_MD5 = "94f7fb64"

_HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR_PATH = os.path.join(_HERE, "vendor", "cnet2_search.py")

_HEADER = (
    "# [READ-ONLY VENDOR COPY] source: github.com/{owner}/{repo}@{ref}:{path}\n"
    "# [READ-ONLY VENDOR COPY] code md5 (header excluded): {md5}\n"
    "# Do not edit; refresh via pack_bridge.fetch_vendor().\n"
).format(owner=OWNER, repo=REPO, ref="main", path=REMOTE_PATH, md5=EXPECTED_MD5)


def fetch_vendor() -> str:
    """拉取远端钉版源、自验 md5、落 vendor 副本，返回副本路径。

    md5 对原始字节核验（加头注释之前）；不合即抛 RuntimeError 停。
    """
    r = get_file(OWNER, REPO, REMOTE_PATH, ref="main")
    raw = base64.b64decode(r["content"])
    md5 = hashlib.md5(raw).hexdigest()[:8]
    if md5 != EXPECTED_MD5:
        raise RuntimeError(
            f"cnet2_search.py md5 不合：got {md5}, expected {EXPECTED_MD5} —— 停"
        )
    os.makedirs(os.path.dirname(VENDOR_PATH), exist_ok=True)
    with open(VENDOR_PATH, "wb") as f:
        f.write(_HEADER.encode())
        f.write(raw)
    return VENDOR_PATH


def load_pack_sources():
    """从 vendor 副本以 importlib 加载并返回模块对象。副本不存在先拉取。

    实测注意（2026-11-16 验证）：cnet2_search 顶部 import 同级模块
    （cnet2_feat / cnet2_np / fastsim 等包内源码），仅 vendor 本文件时
    exec 会缺模块。本函数不做静默兜底：缺依赖时抛 RuntimeError 并
    点名缺失模块——调用方须先把钉版同级源（feat 2a45b08b 等）置于
    sys.path 后再调用。叶函数定位不依赖 exec，用 list_top_level_functions
    的 ast 清单即可（leaf 是 make_search_agent 内的嵌套函数，行 666）。
    """
    if not os.path.exists(VENDOR_PATH):
        fetch_vendor()
    spec = importlib.util.spec_from_file_location("cnet2_search_vendored", VENDOR_PATH)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as e:
        raise RuntimeError(
            f"load_pack_sources: 钉版模块同级依赖缺失（{e}）；"
            "须先把包内同级钉版源置于 sys.path，本桥不静默兜底"
        ) from e
    return mod


def list_top_level_functions(path: str = VENDOR_PATH) -> list[dict]:
    """ast 解析副本，列出顶层函数名并标注疑似叶函数/搜索入口。

    标注规则（启发式，仅作定位清单）：名字含 leaf/value/eval/heur 的
    标 leaf_candidate；含 search/agent/make 的标 entry_candidate。
    """
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            low = name.lower()
            tags = []
            if any(k in low for k in ("leaf", "value", "eval", "heur")):
                tags.append("leaf_candidate")
            if any(k in low for k in ("search", "agent", "make")):
                tags.append("entry_candidate")
            out.append({"name": name, "lineno": node.lineno, "tags": tags})
    return out


if __name__ == "__main__":
    p = fetch_vendor()
    print(f"vendor copy: {p}")
    for fn in list_top_level_functions():
        print(fn)

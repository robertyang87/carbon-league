"""只读桥接：钉版四源（cnet2_search / cnet2_feat / cnet2_np / fastsim）。

SI08 修复版（2026-11-16）：旧版只拉 cnet2_search 一件、依赖
~/workspace/gh_api 在场、且加载缓存时不复核 md5。本版：

- 四源全量：fetch_vendor() 拉取全部四源，逐件按钉版 md5 校验后落
  proto/vendor/ 副本；任一不合即抛 RuntimeError 停，不落半成品。
- 缓存复核：load_pack_sources() 加载前先 verify_vendor_cache() 逐件
  复核在盘副本的代码 md5，不合即停（哨兵替换/漂移都会被拦）。
- 离线 bootstrap：bootstrap_offline(source_dir) 从本地冻结 checkout
  （含四源 .py 的目录，如仓库 code/ 的冻结副本或另一 vendor 目录）
  复制并校验，全程不碰网络、不依赖 gh_api。gh_api 仅在在线拉取时
  惰性 import，缺失时报错并指向离线路径。

md5 口径与 real_env.verify_vendor / 包内 _BUILD_HASHES 一致：
cnet2_search 的 vendor 副本带 3 行头注释，代码 md5 ＝去前 3 行后
计算；其余三源为整文件 md5。副本只读，本桥从不回写仓库。
"""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import os
import sys

OWNER = "robertyang87"
REPO = "carbon-league"

# name -> (仓库路径, 钉版代码 md5 全值, vendor 副本是否带 3 行头注释)
SOURCES = {
    "cnet2_search": ("code/cnet2_search.py", "94f7fb6438bb7577b141e388f068e122", True),
    "cnet2_feat": ("code/cnet2_feat.py", "2a45b08b6f97508541e62867e408748e", False),
    "cnet2_np": ("code/cnet2_np.py", "d8c15ad4fdd8ee079363cf215d52daf3", False),
    "fastsim": ("code/fastsim.py", "aefe6c83826efae4de83bef8c643c80b", False),
}

_HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(_HERE, "vendor")
# 兼容旧名（cnet2_search 的 vendor 路径）
VENDOR_PATH = os.path.join(VENDOR_DIR, "cnet2_search.py")

_HEADER_TEMPLATE = (
    "# [READ-ONLY VENDOR COPY] source: github.com/{owner}/{repo}@{ref}:{path}\n"
    "# [READ-ONLY VENDOR COPY] code md5 (header excluded): {md5}\n"
    "# Do not edit; refresh via pack_bridge.fetch_vendor().\n"
)
_HEADER_MARK = "# [READ-ONLY VENDOR COPY]"


def _header(name: str) -> bytes:
    remote, md5, _ = SOURCES[name]
    return _HEADER_TEMPLATE.format(
        owner=OWNER, repo=REPO, ref="main", path=remote, md5=md5[:8]
    ).encode()


def code_bytes(name: str, raw: bytes) -> bytes:
    """从任意形态字节（仓库原始/vendor 副本）取代码字节。

    带头注释的源（cnet2_search）：若字节以头注释标记开头，去前 3 行；
    其余情况原样返回。与 real_env.verify_vendor 的 tail -n +4 口径一致。
    """
    _remote, _md5, has_header = SOURCES[name]
    if has_header and raw.startswith(_HEADER_MARK.encode()):
        return b"".join(raw.splitlines(keepends=True)[3:])
    return raw


def code_md5(name: str, raw: bytes) -> str:
    return hashlib.md5(code_bytes(name, raw)).hexdigest()


def _vendor_bytes(name: str, raw_code: bytes) -> bytes:
    """代码字节 -> vendor 副本字节（仅 cnet2_search 加 3 行头注释）。"""
    _remote, _md5, has_header = SOURCES[name]
    if has_header:
        return _header(name) + raw_code
    return raw_code


def _check(name: str, raw: bytes, origin: str) -> bytes:
    """校验 raw 的代码 md5 合钉版，返回代码字节；不合抛 RuntimeError。"""
    want = SOURCES[name][1]
    got = code_md5(name, raw)
    if got != want:
        raise RuntimeError(
            f"{name} md5 不合（{origin}）：got {got}, expected {want} —— 停"
        )
    return code_bytes(name, raw)


def _fetch_remote(remote_path: str) -> bytes:
    """在线拉取一件远端源。gh_api 惰性 import；不可用时指向离线路径。"""
    try:
        sys.path.insert(0, os.path.expanduser("~/workspace"))
        from gh_api import get_file
    except ImportError as e:
        raise RuntimeError(
            "gh_api 不可用，无法在线拉取；改用离线 bootstrap："
            "pack_bridge.bootstrap_offline(<本地冻结 checkout 目录>)"
        ) from e
    r = get_file(OWNER, REPO, remote_path, ref="main")
    return base64.b64decode(r["content"])


def fetch_vendor(names=None) -> dict:
    """在线拉取钉版四源（或 names 子集）、逐件验 md5、落 vendor 副本。

    全部校验通过后才逐件落盘；任一不合即抛 RuntimeError，已落盘的
    旧副本不被半成品覆盖（先全拉全验、后统一写）。
    返回 {name: vendor 路径}。
    """
    names = list(names) if names else list(SOURCES)
    staged = {}
    for name in names:
        raw = _fetch_remote(SOURCES[name][0])
        staged[name] = _check(name, raw, f"github {SOURCES[name][0]}")
    os.makedirs(VENDOR_DIR, exist_ok=True)
    out = {}
    for name, code in staged.items():
        path = os.path.join(VENDOR_DIR, name + ".py")
        with open(path, "wb") as f:
            f.write(_vendor_bytes(name, code))
        out[name] = path
    verify_vendor_cache(names)
    return out


def bootstrap_offline(source_dir: str, names=None) -> dict:
    """离线 bootstrap：从本地冻结目录复制四源到 vendor 并逐件校验。

    source_dir 须含 {name}.py 四件（仓库 code/ 冻结副本、另一 vendor
    目录均可，带头注释的副本会被正确识别）。不碰网络、不依赖 gh_api。
    返回 {name: vendor 路径}。
    """
    names = list(names) if names else list(SOURCES)
    staged = {}
    for name in names:
        src = os.path.join(source_dir, name + ".py")
        if not os.path.exists(src):
            raise RuntimeError(f"离线源缺文件：{src} —— 停")
        with open(src, "rb") as f:
            raw = f.read()
        staged[name] = _check(name, raw, f"offline {src}")
    os.makedirs(VENDOR_DIR, exist_ok=True)
    out = {}
    for name, code in staged.items():
        path = os.path.join(VENDOR_DIR, name + ".py")
        with open(path, "wb") as f:
            f.write(_vendor_bytes(name, code))
        out[name] = path
    verify_vendor_cache(names)
    return out


def verify_vendor_cache(names=None) -> dict:
    """逐件复核在盘 vendor 副本的代码 md5，不合/缺失即抛 RuntimeError。

    返回 {name: 实测代码 md5}。加载前必经此关（SI08：缓存不复核已修）。
    """
    names = list(names) if names else list(SOURCES)
    out = {}
    for name in names:
        path = os.path.join(VENDOR_DIR, name + ".py")
        if not os.path.exists(path):
            raise RuntimeError(
                f"vendor 副本缺失：{path}；先 fetch_vendor() 或 "
                "bootstrap_offline(<冻结目录>) —— 停"
            )
        with open(path, "rb") as f:
            raw = f.read()
        want = SOURCES[name][1]
        got = code_md5(name, raw)
        if got != want:
            raise RuntimeError(
                f"vendor {name} md5 不合：got {got}, expected {want} —— 停"
            )
        out[name] = got
    return out


def _load_module(name: str):
    path = os.path.join(VENDOR_DIR, name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_all_sources() -> dict:
    """校验缓存后加载四源，返回 {name: module}。

    缺件时先尝试在线拉取（gh_api 不可用则报错指向离线 bootstrap）；
    拉取后、加载前均逐件复核 md5。vendor 目录同时置于 sys.path 首位，
    使 cnet2_search 的同级 import 解析到钉版副本。
    """
    try:
        verify_vendor_cache()
    except RuntimeError:
        fetch_vendor()
        verify_vendor_cache()
    if VENDOR_DIR not in sys.path:
        sys.path.insert(0, VENDOR_DIR)
    mods = {}
    for name in ("cnet2_np", "cnet2_feat", "fastsim", "cnet2_search"):
        mods[name] = _load_module(name)
    return mods


def load_pack_sources():
    """兼容旧签名：校验并加载四源，返回 cnet2_search 模块对象。"""
    return load_all_sources()["cnet2_search"]


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
    import argparse

    ap = argparse.ArgumentParser(description="钉版四源只读桥接")
    ap.add_argument("--verify", action="store_true", help="只复核在盘缓存")
    ap.add_argument("--offline", metavar="DIR",
                    help="离线 bootstrap：从本地冻结目录复制四源")
    args = ap.parse_args()
    if args.offline:
        paths = bootstrap_offline(args.offline)
    elif args.verify:
        verify_vendor_cache()
        paths = {n: os.path.join(VENDOR_DIR, n + ".py") for n in SOURCES}
    else:
        paths = fetch_vendor()
    for name, p in paths.items():
        print(f"{name}: {p}  md5 {verify_vendor_cache([name])[name]}")
    for fn in list_top_level_functions():
        print(fn)

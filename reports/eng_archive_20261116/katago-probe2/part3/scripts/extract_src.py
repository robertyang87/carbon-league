#!/usr/bin/env python3
"""Extract embedded module sources + CONF/DEFAULTS from a deployment pack file.
Read-only on the pack; writes extracted sources under part3/work/<tag>/."""
import ast, hashlib, json, sys, os

def load_pack(path):
    with open(path, "r") as f:
        text = f.read()
    tree = ast.parse(text)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("_SRC", "CONF"):
                out[name] = ast.literal_eval(node.value)
    return text, out

def main():
    path, tag = sys.argv[1], sys.argv[2]
    text, parts = load_pack(path)
    src = parts["_SRC"]; conf = parts["CONF"]
    wdir = os.path.join(os.path.dirname(__file__), "..", "work", tag)
    os.makedirs(wdir, exist_ok=True)
    man = {"pack_path": path, "pack_md5": hashlib.md5(text.encode()).hexdigest(),
           "conf": conf, "modules": {}}
    for k, v in src.items():
        p = os.path.join(wdir, k if k.endswith(".py") else k + ".py")
        with open(p, "w") as f:
            f.write(v)
        man["modules"][k] = {"md5": hashlib.md5(v.encode()).hexdigest(), "bytes": len(v.encode()),
                             "lines": v.count("\n") + (0 if v.endswith("\n") else 1)}
    with open(os.path.join(wdir, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1, sort_keys=True)
    print(json.dumps(man, indent=1, sort_keys=True))

if __name__ == "__main__":
    main()

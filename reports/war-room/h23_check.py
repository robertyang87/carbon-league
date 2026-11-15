"""h23 定检直读驱动：读 sync/h23 分支头、顶层与 bigeval_live 清单、
三份 cal.out、末 5 个 live 文件尾行；与上一趟基线逐项比对。
用法: python3 h23_check.py <mark> ; 输出 JSON 到同目录 h23-check-<mark>-1115.json
"""
import sys, os, json, hashlib, base64, urllib.request
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BRANCH = "robertyang87", "carbon-league", "sync/h23"
BASE = "synced/h23"
HF = os.path.dirname(os.path.abspath(__file__))

def md5_names(names):
    return hashlib.md5("\n".join(sorted(names)).encode()).hexdigest()

def list_dir(path):
    d = gh_api.get_file(OWNER, REPO, path, ref=BRANCH)
    if isinstance(d, dict) and d.get("type") == "file":
        return [d["name"]]
    return [e["name"] for e in d]

def read_file(path):
    """返回 (text_or_None, size, via)。>1MB 文件走 download_url。"""
    d = gh_api.get_file(OWNER, REPO, path, ref=BRANCH)
    size = d.get("size", 0)
    if size and size > 900_000:
        url = d.get("download_url")
        with urllib.request.urlopen(url, timeout=60) as r:
            raw = r.read()
        return raw.decode("utf-8", errors="replace"), size, "download_url"
    content = d.get("content", "")
    if not content:
        return "", size, "inline-empty"
    return base64.b64decode(content).decode("utf-8", errors="replace"), size, "inline"

def tail_info(path):
    text, size, via = read_file(path)
    lines = [l for l in text.splitlines() if l.strip()]
    last = None
    if lines:
        try:
            obj = json.loads(lines[-1])
            last = {"seed": obj.get("seed"), "margin": obj.get("margin")}
        except Exception:
            last = {"raw_tail": lines[-1][:120]}
    return {"lines": len(lines), "last": last, "via": via, "size": size}

def main():
    mark = sys.argv[1] if len(sys.argv) > 1 else "H23STATUS"
    commits = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={BRANCH}&per_page=3")
    head = [{"sha": c["sha"][:8], "date": c["commit"]["committer"]["date"],
             "msg": c["commit"]["message"]} for c in commits]
    top_names = list_dir(BASE)
    live_names = list_dir(BASE + "/bigeval_live")
    cal = {}
    for f in ["lineA_cal.out", "m3s2r1pre_cal.out", "m7bsr1_cal.out"]:
        text, _, _ = read_file(f"{BASE}/{f}")
        cal[f] = text
    last5 = sorted(live_names)[-5:]
    tails = {n: tail_info(f"{BASE}/bigeval_live/{n}") for n in last5}
    rate = gh_api.req("GET", "/rate_limit")
    core = rate["resources"]["core"]
    out = {
        "trip": "2026-11-15 11:27 CST",
        "mark": mark,
        "h23_head": head,
        "top_level_items": len(top_names),
        "top_level_names_md5": md5_names(top_names),
        "top_level_names": sorted(top_names),
        "cal": cal,
        "live_items": len(live_names),
        "live_names_md5": md5_names(live_names),
        "live_last5": last5,
        "live_tails": tails,
        "rate_remaining": core["remaining"],
        "rate_reset": core["reset"],
    }
    p = os.path.join(HF, f"h23-check-{mark.lower()}-1115.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in
        ["trip", "mark", "h23_head", "top_level_items", "top_level_names_md5",
         "live_items", "live_names_md5", "live_last5", "live_tails",
         "rate_remaining", "rate_reset"]}, indent=1, ensure_ascii=False))
    print("CAL lineA:", repr(cal["lineA_cal.out"][:200]))
    print("CAL m3s2r1pre:", repr(cal["m3s2r1pre_cal.out"][:300]))
    print("CAL m7bsr1:", repr(cal["m7bsr1_cal.out"][:200]))
    print("WROTE", p)

if __name__ == "__main__":
    main()

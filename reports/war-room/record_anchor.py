"""Record the new E0 baseline anchor (+785) and re-based calibration scores
into reports/live/exams.md (branch cnet2-search-v12), then read back."""
import sys, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, REF = "robertyang87", "carbon-league", "cnet2-search-v12"
PATH = "reports/live/exams.md"

f = gh_api.get_file(OWNER, REPO, PATH, ref=REF)
old_sha = f["sha"]
text = base64.b64decode(f["content"]).decode()
assert text.startswith("# exams  2026-11-12 07:07"), "header changed, abort"

section = """## 18:22 E0 基线复测（新锚）与校准重基

- 基线场：E0 vs Master2 真身，200 局，seed 999200，h23，PID 119669，18:22 落盘：**167 胜、mean +785、se 66、lo2se +654、err 0**（live 聚合核对一致）。
- 旧锚 −778 作废：其原始场次无记录；新锚显示 E0 实际比 Master2 强 785 分（方向与旧锚完全反向）。
- 传递性验证：按新锚，m3s2r1 对 E0 的传递预测 = 1197 − 785 = **+412**，与 h24 直接实测（140/200 时 +387.7，终值待出）一致；n1r 传递预测 = 969 − 785 = **+184**，与直接实测 +245 在 1 SE 内。旧锚下的约 1400 分传递缺口是锚的符号错误所致，现已闭合。
- 重基后校准分（原始分 − 785）：m3s2r1 +1197 → **+412**（旧 +419）；n1r +969 → **+184**（旧 +191，仍过上榜闸 ≥150）；sb100 +896 → +111（旧 +118）；m3s2r1 ep10 +1115 → +330（旧 +337）。直接对 E0 的实测场为准绳，校准分仅作跨考官比较。

"""

new_text = text.replace("# exams  2026-11-12 07:07",
                       "# exams  2026-11-12 18:40", 1)
new_text = new_text.replace("## 配对结果（同种子减基准，最新在上）",
                            section + "## 配对结果（同种子减基准，最新在上）", 1)
assert new_text != text and section.strip() in new_text

res = gh_api.put_file(OWNER, REPO, PATH, new_text.encode(),
                      "exams: record new E0 baseline anchor +785 (200 games, se 66) and re-based calibration scores",
                      sha=old_sha, branch=REF)
print("put commit:", res["commit"]["sha"])

back = gh_api.get_file(OWNER, REPO, PATH, ref=REF)
back_text = base64.b64decode(back["content"]).decode()
assert "18:22 E0 基线复测" in back_text and "+785" in back_text
assert back_text.startswith("# exams  2026-11-12 18:40")
print("read-back OK, new sha:", back["sha"])

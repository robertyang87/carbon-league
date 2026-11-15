import json
import sys

sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER = "robertyang87"
REPO = "carbon-league"

g1 = {
    "exam": "a4g1_pair_vs_master2",
    "seed_domain": [996000, 996199],
    "n_per_field": 200,
    "harness_md5": "354bcd1bc2dc8bc188c813f53d28c893",
    "opponent": "Master2 (opp_master2r, 2aebfbc1008d3475ece8e178696e292d)",
    "du_definition": "mean over 200 games of U(margin_a4) - U(margin_base), U(m)=0.5*W+0.5*sigmoid(m/1000), same-seed paired, computed by ops room",
    "fields": {
        "a4g1_a4_m2": {
            "package": "exp/m3s2r1_a4.py (ep17 winner)",
            "a_md5": "cc2ea7d60ac005731431da675d99d6f3",
            "wins": 173,
            "losses": 27,
            "mean_margin": 993,
            "mean_margin_recomputed": 993.32,
            "se": 58,
            "lo2se": 878,
            "worst_ms": 85.0,
            "err": 0,
            "FAIL": False,
            "u_field_max_dev": 0.0,
            "archive_md5": "c0388cd1021c0b0cdbe1241cf1d9ef87",
            "archive_readback_match": True,
            "archive_head": "7248a5e9",
            "bytes": 882609,
        },
        "a4g1_base_m2": {
            "package": "exp/m3s2g1_base.py (ep19 incumbent)",
            "a_md5": "bb997d7137395e6e66f7b03134d96897",
            "wins": 180,
            "losses": 20,
            "mean_margin": 1034,
            "mean_margin_recomputed": 1033.91,
            "se": 53,
            "lo2se": 927,
            "worst_ms": 84.0,
            "err": 0,
            "FAIL": False,
            "u_field_max_dev": 0.0,
            "archive_md5": "49de96a2899821e8f77e1c5af435c897",
            "archive_readback_match": True,
            "archive_head": "01bd7dbb",
            "bytes": 882366,
        },
    },
}

r1 = {
    "exam": "a4r1_pair_vs_master2",
    "seed_domain": [996000, 996199],
    "n_per_field": 200,
    "harness_md5": "354bcd1bc2dc8bc188c813f53d28c893",
    "opponent": "Master2 (opp_master2r, 2aebfbc1008d3475ece8e178696e292d)",
    "du_definition": "mean over 200 games of U(margin_a4) - U(margin_base), U(m)=0.5*W+0.5*sigmoid(m/1000), same-seed paired, computed by ops room",
    "fields": {
        "a4r1_a4_m2": {
            "package": "exp/m3s2r1_a4.py (ep18 winner)",
            "a_md5": "4a85855ba8878ec79d8c820f87e418cb",
            "wins": 189,
            "losses": 11,
            "mean_margin": 1173,
            "mean_margin_recomputed": 1172.87,
            "se": 54,
            "lo2se": 1065,
            "worst_ms": 124.0,
            "err": 0,
            "FAIL": False,
            "u_field_max_dev": 0.0,
            "archive_md5": "e5e7b6eb8d8c9703b8c2dbb282de5a6a",
            "archive_readback_match": True,
            "archive_head": "76a82549",
            "bytes": 882313,
        },
        "a4r1_base_m2": {
            "package": "exp/m3s2r1_base.py (ep19 incumbent)",
            "a_md5": "11d8005930b298290c2e150949349647",
            "wins": 187,
            "losses": 13,
            "mean_margin": 1180,
            "mean_margin_recomputed": 1179.68,
            "se": 57,
            "lo2se": 1065,
            "worst_ms": 120.0,
            "err": 0,
            "FAIL": False,
            "u_field_max_dev": 0.0,
            "archive_md5": "9ba8ab31720064844816236fbb65885e",
            "archive_readback_match": True,
            "archive_head": "89a919e8",
            "bytes": 881691,
        },
    },
}

for name, doc in [("a4g1", g1), ("a4r1", r1)]:
    path = f"synced/h29/summaries/{name}.json"
    raw = json.dumps(doc, indent=1, ensure_ascii=False).encode()
    sha = None
    try:
        cur = gh_api.get_file(OWNER, REPO, path, ref="sync/h29")
        sha = cur["sha"]
    except Exception:
        sha = None
    r = gh_api.put_file(OWNER, REPO, path, raw,
                        f"summary {name} paired A4 fields (C1)",
                        sha=sha, branch="sync/h29")
    print(name, r["commit"]["sha"])

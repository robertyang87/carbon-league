"""Export realized preferences only for the exact two evaluated joint commands on every registered map."""
from statistics import mean

import responses
import select_orders as sel


def compile_pairs():
    plan = responses.check_plan()
    roots = sel.read_json(sel.HERE / "roots.json")["tickets"]
    sel.need([r["seed"] for r in roots] == list(sel.TRAIN_SEEDS), "Dropped or reordered registered map")
    pairs = []
    for root in roots:
        if not root["selected"]:
            pairs.append({"seed": root["seed"], "selected": False})
            continue
        directory = sel.HERE / "tickets" / str(root["seed"])
        s = sel.read_json(directory / "summary.json")
        sel.need(s["root_public"] == root["root_public_sha256"] and
                 s["protocol"] == sel.sha(sel.HERE / "protocol.json") and
                 s["root_and_suffix_replayed"] and set(s["arms"]) == {"parent", "alternative"},
                 "Unverified response")
        arms = {}
        for name, receipt in s["arms"].items():
            order = receipt["root_order"]
            sel.need(order == root["orders"][name] and
                     sel.sha(directory / (name + ".json.gz")) == receipt["sha256"], "Action identity changed")
            option = next(r for r in root["menu"] if r["action"] == order)
            arms[name] = {
                "joint_order": order,
                "origin": name,
                "fitted_probability": option["probability"],
                "prior_probability": option["prior_probability"],
                "scores_by_seat": receipt["terminal"]["scores"],
                "terminal": receipt["value"],
                "native_fresh_calls": receipt["native_fresh_calls"],
                "mean_k": mean(receipt["K"]),
                "mean_h": mean(receipt["H"]),
                "aborted": receipt["aborts"],
                "response_sha256": receipt["sha256"],
            }
        gains = {k: arms["alternative"]["terminal"][k] - arms["parent"]["terminal"][k]
                 for k in ("win", "margin", "U")}
        sel.need(gains == s["alternative_minus_parent"], "Wrong signed label")
        pairs.append({
            "selected": True,
            "seed": s["seed"], "seat": s["seat"], "terrain": s["terrain"], "step": s["root"],
            "source_sha256": s["source_sha256"], "public_sha256": s["root_public"],
            "pivot_uid": root["pivot_uid"], "pivot_role": root["pivot_role"],
            "sampler_sha256": plan["proposer"], "continuation_sha256": plan["continuation"],
            "examiner_sha256": plan["examiner"], "two_actions": arms,
            "alternative_minus_parent": gains,
            "unmeasured_orders": len(root["menu"]) - 2,
            "summary_sha256": sel.sha(directory / "summary.json"),
        })
    return {
        "schema": "sapling-two-order-realized-training-returns/1",
        "scope": "One natural coupled response per exact joint order on a training root. "
                 "No option outside the pair is labeled. No game from step zero is added. "
                 "Realized terminal response differences need not agree with expected action values.",
        "maps": len(roots), "pairs": pairs,
    }


if __name__ == "__main__":
    combined = compile_pairs()
    sel.common.save(sel.HERE / "training_pairs.json", combined)
    for p in combined["pairs"]:
        print(p["seed"], p.get("alternative_minus_parent"))

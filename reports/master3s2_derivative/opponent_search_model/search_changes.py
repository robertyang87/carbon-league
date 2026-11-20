"""Separate actual opponent search changes using successive archived public worker counters."""
import json
import zipfile

import study


def main():
    cards = study.read(study.D/"opponent_rollout_calibration/protocol.json")["tickets"]
    rows = study.read(study.HERE/"held_roots.json.gz")
    weights = study.read(study.HERE/"weights.json")
    sides = {}
    for card in [c for c in cards if c["split"]=="held"]:
        path = study.D/card["rel"]
        assert study.sha(path.read_bytes()) == card["archive_sha256"]
        with zipfile.ZipFile(path) as archive:
            calls = [json.loads(line)["calls"][1-card["seat"]]["response"] for line in archive.read(card["arm"]+"/turns.jsonl").splitlines()]
        cumulative = 0
        for t, response in enumerate(calls):
            delta = response["search_counts"]["changed"] - cumulative
            assert delta in (0,1)
            cumulative += delta
            sides[(card["seed"],card["seat"],t)] = delta
    result = {}
    for flag in (0,1):
        part = [r for r in rows if sides[(r["seed"],r["seat"],r["t"])] == flag]
        methods = {"native":"native", "prior":"prior", "critic":"critic", "calibrated":str(weights["beta"])}
        result[str(flag)] = {"roots":len(part), "any_exact":sum(any(m["exact"] for m in r["metrics"]) for r in part),
                             "methods":{k:study.stats(part,mode,weights)["totals"] for k,mode in methods.items()}}
    study.save(study.HERE/"search_changes.json",result)
    for k,r in result.items():
        print(k,r["roots"],r["any_exact"],[(n,s["exact"],s["correct"],s["workers"],s["heavy_correct"],s["heavy"]) for n,s in r["methods"].items()])


if __name__=="__main__":
    main()

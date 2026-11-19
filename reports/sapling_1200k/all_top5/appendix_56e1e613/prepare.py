"""Rebuild the frozen 56e1e613 archive, dual-engine validations, and per-game evidence."""
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
sys.path.insert(0, str(TOP))

import fetch
from appendix_97de1ebc.collect import inventories
from appendix_batch import prepare
from appendix_f7109df6.report import build
from continue_appendix import check_archive, previous_manifests
from validation_delta import run as validate

REVISION = "56e1e6137fcff237aa212ea9423a29e8d486501d"
PREVIOUS = "b243ff82d04a4d2a6ffb56016ec857fba06903fc"
MODEL_FREEZE = "ca46c10803e1031e7bf39abe6c3acaee7d318110"


def restore_published_replays(manifest_path):
    if not manifest_path.is_file():
        return
    manifest = fetch.load(manifest_path)
    if manifest["revision"] != REVISION:
        raise ValueError("published revision changed")
    check_archive(manifest)
    keys = ("id", "revision", "source", "sha256", "canonical_sha256", "seed", "environment_uuid",
            "transitions", "terminal", "clean_terminal", "source_attempt_errors", "frame_verified")
    for entry in manifest["new_archives"]:
        with tarfile.open(fetch.ROOT / entry["archive"]) as archive:
            for game in manifest["games"]:
                if game["archive_location"]["archive"] != entry["archive"]:
                    continue
                fetch.atomic(fetch.CACHE / game["path"], archive.extractfile(game["path"]).read())
                fetch.dump(fetch.CACHE / "receipts" / (game["id"] + ".json"), {k: game[k] for k in keys})


def main():
    prior = previous_manifests(TOP / "appendix_b243ff82")
    old = fetch.load(prior[-1])
    if old["revision"] != PREVIOUS or old["accumulated_games"] != 1629:
        raise ValueError("unexpected predecessor")
    fetch.git("merge-base", "--is-ancestor", PREVIOUS, REVISION)
    names = [str(path.parent.relative_to(TOP)) for path in prior]
    delta = inventories(PREVIOUS, REVISION, names)
    if delta["unscored_new_ids"] or delta["new_count"] != 1 or delta["new_mutual"]:
        raise ValueError("frozen delta changed")
    fetch.git("merge-base", "--is-ancestor", PREVIOUS, MODEL_FREEZE)
    fetch.git("merge-base", "--is-ancestor", MODEL_FREEZE, REVISION)
    before = inventories(PREVIOUS, MODEL_FREEZE, names)
    after = inventories(MODEL_FREEZE, REVISION, names)
    if before["all_added_references_by_id"] or after["all_added_references_by_id"] != delta["all_added_references_by_id"]:
        raise ValueError("new score references differ across the Sapling model freeze")
    delta["model_freeze"] = {
        "revision": MODEL_FREEZE,
        "new_index_references_before_freeze": len(before["all_added_references_by_id"]),
        "new_references_after_freeze": after["new_reference_count"] - after["old_reference_count"],
        "new_scored_top_games_before_freeze": before["new_count"],
        "new_scored_top_games_after_freeze": after["new_count"],
    }
    fetch.dump(HERE / "score_delta.json", delta)
    faults = [p.parent / "unscored/manifest.json" for p in prior if (p.parent / "unscored/manifest.json").is_file()]
    manifest_path = HERE / "manifest.json"
    restore_published_replays(manifest_path)
    fault_cards = {}
    prepare(REVISION, prior, HERE, (delta["new_count"], delta["new_mutual"]), unscored=fault_cards)
    if set(fault_cards) != {g["id"] for p in faults for g in fetch.load(p)["games"]}:
        raise ValueError("historical fault roster changed")
    manifest = fetch.load(manifest_path)
    expected = {gid for gid, g in delta["all_added_references_by_id"].items() if g["class"] == "top"}
    if {g["id"] for g in manifest["games"]} != expected or manifest["accumulated_games"] != delta["union_games"]:
        raise ValueError("independent full catalog and frozen delta disagree")
    check_archive(manifest)
    ledgers = []
    for name, path in zip(names, prior, strict=True):
        ledger = path.parent / "validation_maps.jsonl.gz"
        blob = fetch.git("show", REVISION + ":" + str(ledger.relative_to(fetch.ROOT)))
        if ledger.read_bytes() != blob:
            raise ValueError("historical map registry changed: " + name)
        ledgers.append((name, fetch.sha(blob)))
    validate(fetch.CACHE, HERE, REVISION, PREVIOUS, ledgers, previous_fault_ledgers=faults)

    def run(script, *args):
        subprocess.run([sys.executable, str(TOP / script), *map(str, args)], check=True)

    index = TOP / "context_appendix_56e1e613_index.tsv.gz"
    receipt = TOP / "context_appendix_56e1e613_receipt.json"
    run("appendix_499886f1/cases.py", "--manifest", manifest_path, "--cache", fetch.CACHE)
    run("appendix_0137949d/cash.py", "--manifest", manifest_path, "--cache", fetch.CACHE)
    run("turn_action_windows.py", HERE, "--cache", fetch.CACHE)
    run("context_catalog.py", "--manifest", manifest_path, "--output", index)
    run("context_audit.py", "--index", index, "--validation", HERE / "validation_games.jsonl.gz", "--out", receipt, "--jobs", "1")
    mutual = sum(g["cohort"] == "mutual" for p in [*prior, manifest_path] for g in fetch.load(p)["games"])
    build(HERE, total_mutual=mutual)
    packed_historical = [fetch.CACHE / g["path"] for p in prior for g in fetch.load(p)["games"]]
    cache_args = ["--replay-cache", fetch.CACHE / "replays"] if all(p.is_file() for p in packed_historical) else []
    run("future_economy_check.py", *cache_args, "--expect-games",
        manifest["accumulated_games"], "--out", HERE / "future_economy_check.json")
    run("appendix_56e1e613/exposure.py")
    run("appendix_56e1e613/development_maps.py")
    page = HERE / "README.md"
    text = page.read_text().replace("终局现金（前五 / 对手）", "官方终局分（前五 / 对手）")
    page.write_text(text + "\n从 b243ff82 至模型冻结 ca46c108，天梯成绩引用无新增。该场出现在 ca46c108 之后、此次冻结的 56e1e613 之前，两队真实动作和官方终分均已逐项复验。本轮累计 1,630 场有双方官方分，另 1 场历史官方故障，独立考试排除 1,631 个真实起始地形。[逐步双方实际指令](turn_actions.json.gz) · [增量地形排除核对](exposure.json) · [分阶段经营核对](future_economy_check.json)\n\n新登记开发图 621758400 和 621758401 各自与完整 1,631 图的种子和 225 格 SHA 均无重合。新增比赛的名义种子所生成的本地棋盘与官方录像不同，两种身份均已进入排除依据。[新开发图与完整天梯排除清单](development_maps.json)\n")


if __name__ == "__main__":
    main()

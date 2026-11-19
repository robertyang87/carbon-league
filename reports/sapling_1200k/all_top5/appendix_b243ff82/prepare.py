"""Rebuild the frozen b243ff82 archive, dual-engine validations, and per-game evidence."""
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

REVISION = "b243ff82d04a4d2a6ffb56016ec857fba06903fc"
PREVIOUS = "bec6a010923622a2bca371cf55ac822cb7e350fe"


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
    prior = previous_manifests(TOP / "appendix_bec6a010")
    old = fetch.load(prior[-1])
    if old["revision"] != PREVIOUS or old["accumulated_games"] != 1627:
        raise ValueError("unexpected predecessor")
    fetch.git("merge-base", "--is-ancestor", PREVIOUS, REVISION)
    names = [str(path.parent.relative_to(TOP)) for path in prior]
    delta = inventories(PREVIOUS, REVISION, names)
    if delta["unscored_new_ids"] or delta["new_count"] != 2 or delta["new_mutual"]:
        raise ValueError("frozen delta changed")
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

    index = TOP / "context_appendix_b243ff82_index.tsv.gz"
    receipt = TOP / "context_appendix_b243ff82_receipt.json"
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
    run("appendix_b243ff82/exposure.py")
    page = HERE / "README.md"
    text = page.read_text().replace("终局现金（前五 / 对手）", "官方终局分（前五 / 对手）")
    details = "\nFable v7 的对手从第 64 步起持续领先，终场领先 665.74。Master5s 在第 105–115 步同样遇到对手集中回库，随后在第 251 步取得持续到终局的领先，最终赢 327.70。两局来自不同地图。[Fable v7 逐局账目](67a8bebd8c22430a7fc0905220bd7671-1.md) · [Master5s 逐局账目](bea5046421b817a05fc98012bb1e403e-1.md)\n\n两场均由冻结前五策略执物理席 0，对手均为 RyanX-AIPowercore-it920-evo31。另有 1 局官方异常，合计 1,630 个已见初盘。[初盘排除核对](exposure.json) · [逐步实发命令与结算](turn_actions.json.gz) · [十步与三十步收入口径](future_economy_check.json)\n"
    page.write_text(text + details + "\n在包含本批、截至本轮的归档检出中运行 `python reports/sapling_1200k/all_top5/appendix_b243ff82/prepare.py` 可重建证据。程序优先读取已发布的原始回放。\n")


if __name__ == "__main__":
    main()

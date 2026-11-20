"""Verify four independent matchups, with both seats on each of two registered development maps."""
import gzip
import importlib.util
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
import zipfile
import zlib

import matrix as m


ARMS = ("new_vs_v3", "new_vs_v2", "new_vs_original", "v2_vs_original")
REFERENCE_COMPARISON = ("new_vs_original", "v2_vs_original")


def load_module(name, source):
    spec = importlib.util.spec_from_file_location(name, source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def equal_maps(games, seeds, summarize):
    """Keep every map-seat; an unresolved seat cannot make its map disappear."""
    rows = []
    for seed in seeds:
        pair = sorted((g for g in games if g["seed"] == seed), key=lambda g: g["seat"])
        m.require([g["seat"] for g in pair] == [0, 1], "Different seats on one physical map")
        rows.append(dict(seed=seed, **summarize(pair)))
    win_rates = [r["observed_win_fraction_when_all_decided"] for r in rows]
    margins = [r["mean_official_margin_all"] for r in rows]
    return dict(by_physical_map=rows,
                number_of_maps=len(rows),
                win_fraction=mean(win_rates) if None not in win_rates else None,
                mean_official_margin=mean(margins) if None not in margins else None)


def settle(run, commit, archives, out, digest):
    """Bind ZIPs to hosted artifact digests, rerun the referee, then report matched arms."""
    m.require(tuple(m.ARMS) == ARMS, "Wrong direct and common-opponent contrasts")
    audit = load_module("four_arm_referee_settlement", m.BASE / "settle.py")
    views = load_module("four_arm_public_views", m.STRATEGY / "native_action_onpolicy_intake/observations.py")
    audit.m = m
    audit.CONTRASTS = (REFERENCE_COMPARISON,)
    original_block = audit.block
    original_aggregate = audit.aggregate

    def verified_block(archive, seed, seat, run_id, head, protocol):
        games, errors, sequence, exposure = original_block(archive, seed, seat, run_id, head, protocol)
        block_errors = [reason for name, reason in errors.items() if name not in ARMS]
        if block_errors:
            errors["block"] = "; ".join(block_errors)
            return {}, errors, sequence, exposure
        with TemporaryDirectory() as temp, zipfile.ZipFile(archive) as handle:
            base = Path(temp)
            audit.unpack(archive, base)
            for arm in ARMS:
                folder = base / arm
                if not (folder / "turns.jsonl").exists():
                    continue
                try:
                    saved = handle.read(arm + "/observations.jsonl.gz")
                    if gzip.decompress(saved) != gzip.decompress(views.recover(folder)):
                        raise ValueError("Public observation record differs from the verified actions")
                except (OSError, EOFError, ValueError, TypeError, KeyError, IndexError, AssertionError, zlib.error) as exc:
                    errors[arm] = f"Public views: {exc}"
                    games.pop(arm, None)
        return games, errors, sequence, exposure

    def aggregate(games, seeds):
        m.require(len(seeds) == len(set(seeds)) == 2, "Expected two distinct physical maps")
        results = original_aggregate(games, seeds)
        for arm in ARMS:
            selected = [g for g in games if g["arm"] == arm]
            results["arms"][arm]["map_weighted"] = equal_maps(selected, seeds, audit.outcomes)
        return results

    audit.block = verified_block
    audit.aggregate = aggregate
    return audit.merge(run, commit, list(archives), out, digest)

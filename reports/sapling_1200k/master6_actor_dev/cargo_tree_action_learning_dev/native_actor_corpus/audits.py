"""Verify the two physical examiner seats and settle all 58 recorded joint-action tapes."""
import hashlib
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.sched_setaffinity(0, {0})
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

from common import EXP, HERE, ROOT, FastCarbon, read, save, sha
import json

PANEL = EXP / "panel.json"
HELD = EXP / "learned_carbon_observation/decoder/held_summary.json"
CLEARANCE = EXP / "learned_carbon_observation/source_clearance.json"
RECENT = EXP / "territory_sequence/windowed/fresh_original/clearance.json"
EXAMINER = ROOT / "data/master3s2_source.py"
FASTSIM = ROOT / "code/fastsim.py"
OFFICIAL = ROOT / "code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.py"
FORMAL = (1_887_000_000, 1_888_000_000)
EXTERNAL = (621758300, 621758400)
PANEL_SHA = "fb490e72f2df1fac20a7c26310523e165956a5619fad7de46104604096611124"
EXAMINER_SHA = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
EXAMINER_CONFIG = "9d37b28c3bfc3cb28411c437d24d49c3dc8af7b813d9d7a16b0b0896aa9a37a6"


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def terrain(cells):
    if len(cells) != 225 or any(value != int(value) or not 0 <= value <= 255 for value in cells):
        raise ValueError("Invalid physical opening")
    return hashlib.sha256(bytes(map(int, cells))).hexdigest()


def equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f"{label}: {actual} != {expected}")


def official_row(card):
    receipt = read(ROOT / card["receipt_path"])
    if "official" in receipt:
        game = receipt["game"]
        equal(game["raw_sha256"], card["sha256"], "native receipt tape hash")
        return receipt["official"], game
    matches = [game for run in receipt["runs"].values() if run["pack_sha256"] == card["pack"]
               for game in run["games"] if game["transcript_sha256"] == card["sha256"]]
    equal(len(matches), 1, "matching official games in the batch receipt")
    return matches[0], None


def verify(card, examiner_md5):
    for field, key in (("path", "sha256"), ("receipt_path", "receipt_sha256")):
        equal(sha(ROOT / card[field]), card[key], field)
    tape = read(ROOT / card["path"])
    official, native = official_row(card)
    for key, actual in (("seed", tape["seed"]), ("seat", tape["candidate_seat"]), ("pack", tape["pack_sha256"]),
                        ("terrain", tape["initial_terrain_u8_sha256"])):
        equal(actual, card[key], key)
    for key, actual in (("seed", card["seed"]), ("seat", card["seat"]), ("map_sha256", card["terrain"]),
                        ("transcript_sha256", card["sha256"]), ("statuses", ["DONE", "DONE"])):
        equal(official[key], actual, key)
    equal(tape["examiner_md5"], examiner_md5, "examiner identity")
    equal(digest(tape["configuration"]), EXAMINER_CONFIG, "examiner search settings")
    equal(tape["audit_messages"] + tape["errors"], [], "archived agent or search failures")
    equal(terrain(tape["initial_full_carbon"]), card["terrain"], "archived initial terrain")
    cfg = tape["game_configuration"]
    sim = FastCarbon(cfg).reset(seed=card["seed"])
    equal(sim.carbon, tape["initial_full_carbon"], "225 opening cells")
    n = len(tape["actions"])
    equal(n, tape["steps"], "tape step count")
    equal(n, official["steps"], "official step count")
    equal([row[0] for row in tape["search_calls"]], list(range(n)), "examiner invocation steps")
    if not 0 < n < cfg["episodeSteps"] or "searched" not in {row[1] for row in tape["search_calls"]}:
        raise ValueError("Invalid tape length or examiner never ran its search")
    min_price = min(cfg["recCollectorCost"], cfg["recPlanterCost"])
    for t, joint in enumerate(tape["actions"]):
        equal(len(joint), 2, "physical joint-action arity")
        pre_cash = list(sim.cash)
        sim.step(*joint)
        extinct = [not sim.player_workers[s] and not sim.player_trees[s]
                   and (not sim.bases[s] or sim.cash[s] < min_price) for s in (0, 1)]
        if any(extinct) and t + 1 != n:
            raise ValueError("Archived tape continues after extinction")
    equal(sim.step_count, n, "replayed transitions")
    if n != cfg["episodeSteps"] - 1 and not any(extinct):
        raise ValueError("Unexplained early terminal")
    scores = [(n - cfg["episodeSteps"] - 1 if extinct[s] else pre_cash[s]) if any(extinct) else sim.cash[s]
              for s in (0, 1)]
    equal(extinct, tape["extinct_by_seat"], "extinction")
    equal(pre_cash, tape["last_pre_cash"], "terminal pre-action balances")
    equal(list(sim.cash), tape["last_post_cash"], "terminal post-action balances")
    equal(scores, tape["official_score_by_seat"], "reconstructed rewards")
    equal(scores, official["official_reward_by_seat"], "official rewards")
    actions_sha = digest(tape["actions"])
    if native is not None:
        equal(native["actions_sha256"], actions_sha, "native joint actions")
        equal(native["final_state_sha256"], digest(sim.get_state()), "native final state")
    return {
        "expert_seat": 1 - card["seat"], "steps": n, "official_score_by_seat": scores,
        "identity_audit": {
            "official_statuses": official["statuses"], "examiner_md5": examiner_md5,
            "actual_joint_actions_sha256": actions_sha, "extinct_by_seat": extinct,
            "final_pre_cash_by_seat": pre_cash, "final_post_cash_by_seat": sim.cash,
            "reconstructed_official_rewards_by_seat": scores, "full_opening_cells_matched": 225,
            "raw_and_official_receipt_sha_verified": True, "native_final_state_hash_checked": native is not None,
            "examiner_calls": dict(Counter(row[1] for row in tape["search_calls"])),
        },
    }


def main():
    equal(sha(PANEL), PANEL_SHA, "frozen parent panel")
    equal(sha(EXAMINER), EXAMINER_SHA, "original examiner source")
    examiner_md5 = hashlib.md5(EXAMINER.read_bytes()).hexdigest()
    panel = read(PANEL)
    decoder = {c["seed"]: c["terrain"] for c in read(HELD)["run"]["cards"]}
    equal(sorted(decoder), [621471120, 621476120, 621477121], "decoder-held terrains")
    cfg = read(ROOT / panel["cards"][0]["path"])["game_configuration"]
    clearance = read(RECENT)
    explicit = clearance["physical_boards"] | clearance["explicit_earlier_development_reservations"]
    registered = set(read(CLEARANCE)["excluded_first_win_controls_and_343_355"])
    reserved = {seed: terrain(FastCarbon(cfg).reset(seed=seed).carbon)
                for seed in sorted(registered | set(range(*EXTERNAL)))}
    for seed, physical in explicit.items():
        equal(reserved[int(seed)], physical, "registered physical terrain")
    roles = defaultdict(set)
    memberships = defaultdict(dict)
    cards = []
    for card in panel["cards"]:
        seed, physical = card["seed"], card["terrain"]
        evidence = verify(card, examiner_md5)
        reasons = []
        if seed in decoder or physical in decoder.values():
            reasons.append("decoder_held_terrain")
        if seed in reserved or physical in reserved.values():
            reasons.append("reserved_control_or_external_development")
        if FORMAL[0] <= seed < FORMAL[1]:
            reasons.append("reserved_formal_seed")
        original = panel["split"][physical]
        split = "excluded" if reasons else original
        cards.append({**card, **evidence, "prior_split": original, "split": split, "quarantine_reasons": reasons})
        roles[physical].add(evidence["expert_seat"])
        memberships[split][physical] = seed
    equal(len(cards), 58, "number of tapes")
    equal(len(roles), 29, "number of physical terrains")
    equal(len({c["sha256"] for c in cards}), len(cards), "distinct physical transcripts")
    equal(sum(len(v) for v in memberships.values()), len(roles), "terrain disjointness")
    if any(s != {0, 1} for s in roles.values()):
        raise ValueError("A terrain's two physical examiner seats do not form one pair")
    sources = [PANEL, HELD, CLEARANCE, RECENT, EXAMINER, FASTSIM, OFFICIAL]
    result = {
        "cards": cards, "split_by_terrain": {t: split for split, group in memberships.items() for t in group},
        "memberships": {s: [{"terrain": t, "seed": n, "expert_seats": [0, 1]} for t, n in sorted(v.items())]
                        for s, v in memberships.items()},
        "split_counts": {s: {"terrains": len(group), "tapes": sum(c["split"] == s for c in cards),
                             "examiner_turns": sum(c["steps"] for c in cards if c["split"] == s)}
                         for s, group in memberships.items()},
        "verified_tapes": len(cards), "verified_joint_transitions": sum(c["steps"] for c in cards),
        "actual_extinction_games": sum(any(c["identity_audit"]["extinct_by_seat"]) for c in cards),
        "decoder_held_physical_terrains": decoder, "registered_and_external_physical_terrains": reserved,
        "formal_seed_half_open": FORMAL, "external_development_seed_half_open": EXTERNAL,
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
        "examiner_config_sha256": EXAMINER_CONFIG,
        "model_input_scope": "One acting examiner seat per tape. Official terminal rewards, terrain identity, seed, opposite-seat commands and full opening are replay evidence. The learner receives only the causal public view and the examiner's recorded command label.",
    }
    save(HERE / "sources.json", result)
    print(result["split_counts"], flush=True)


if __name__ == "__main__":
    main()

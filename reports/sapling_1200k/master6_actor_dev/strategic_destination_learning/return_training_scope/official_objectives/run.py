"""Apply the two match requirements to the exact registered six-map binary-score experiment."""
from collections import defaultdict
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SCOPE = HERE.parent
sys.path.insert(0, str(SCOPE))
spec = importlib.util.spec_from_file_location("scope_851c_alignment", SCOPE / "alignment.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
o = a.o

U_SOURCE = "b498db5dc900fee62605ffe379591291cfe3713239da206ecc049dfe136b45f5"
TRAIN = "63daa51e127aafea36401ebdce6400de60faa772d3929a1803ce768e121be5b4"


def rewards(ep):
    s = ep["seat"]
    scores = ep["official_scores_by_seat"]
    diff = scores[s] - scores[1-s]
    win = int(diff > 0)
    o.require(abs(ep["margin"] - diff) < 1e-8 and win == int(ep["wdl"] == 1), "official score mismatch")
    return diff / 1000.0, win


def zero_direction(seeds, nparams):
    """A shared zero W reward annihilates every score term; a cosine has 0/0 denominator."""
    z = np.zeros((len(seeds), len(seeds)), float)
    return dict(physical_seeds=seeds, parameters=nparams, folds=[
        dict(seed=s, norm=0., held_dot_five=0., other_five_norm=0., cosine=None, dot_unit_normalized_five=None)
        for s in seeds], raw_gram=z.tolist(), cosine_matrix=[[None]*len(seeds) for _ in seeds],
        pairs=[dict(seed_left=s, seed_right=t, dot=0., cosine=None) for i, s in enumerate(seeds) for t in seeds[i+1:]],
        positive_folds=0, positive_of_15=0, distinct_pair_dot_sum=0., pass_screen=False)


def main():
    p = a.INTAKE / "hosted/training_manifest.json"
    ref_path = SCOPE / "alignment.json"
    ref = o.read(ref_path)
    o.require(o.sha(p) == ref["data_sha256"] == TRAIN and o.sha(ref_path) == U_SOURCE, "training cohort or U report changed")
    o.require(o.sha(SCOPE / "alignment.py") == ref["program_sha256"] and o.sha(a.learning.BEHAVIOR) == ref["parent_sha256"],
              "behavior or gradient implementation changed")
    o.require(o.sha(SCOPE / "alignment_protocol.json") == ref["protocol_sha256"] == a.PROTOCOL,
              "original criterion or normalization changed")
    plan = o.read(a.INTAKE / "learning.json")
    _, episodes = a.learning.training_manifest(plan, p)
    arms = {(e["seed"], e["seat"], e["arm"]): e for e in episodes}
    seeds = sorted({e["seed"] for e in episodes})
    prior = {(m["seed"], v["seat"], v["arm"]): v for m in ref["maps"] for v in m["episodes"]}
    o.require(len(arms) == len(episodes) == len(prior) == 24 and len(seeds) == 6, "wrong cohort")
    base = o.package(a.learning.BEHAVIOR)
    teacher = o.teacher_reader(base)
    models = [o.torch_model(base, i) for i in (0, 1)]
    for m in models:
        for name, tensor in m.named_parameters():
            tensor.requires_grad_(not name.startswith(o.VALUE))
    vectors = [[], []]
    map_rows = []
    used_tapes = set()
    n_steps = 0
    for seed in seeds:
        roots = []
        games = []
        for seat in (0, 1):
            for arm, sibling in (("draw_a", "draw_b"), ("draw_b", "draw_a")):
                key = (seed, seat, arm)
                ep = arms[key]
                other = arms[seed, seat, sibling]
                o.require(ep["partner_arm"] == sibling and ep["terrain"] == other["terrain"], "incorrect sibling")
                cash, win = rewards(ep)
                other_cash, other_win = rewards(other)
                official_adv = cash - other_cash
                o.require(win == other_win == 0, "W is no longer identically zero; compute its gradient explicitly")
                traj, digest = a.recover(ep, base, teacher, set(plan["scheduled_turns"]))
                o.require(digest == prior[key]["trajectory_orders_sha256"] and digest not in used_tapes
                          and len(traj) == prior[key]["supported"], "archived trajectory, root identity or duplicate")
                used_tapes.add(digest)
                for features, meta, public in traj:
                    o.require(abs(meta["advantage"] - (ep["u"] - other["u"])) < 1e-12, "incorrect U reference")
                    roots.append((features, dict(meta, advantage=official_adv), public))
                games.append(dict(seat=seat, arm=arm, sibling=sibling, official_scores_by_seat=ep["official_scores_by_seat"],
                                  score_difference=1000 * cash, score_difference_advantage_div_1000=official_adv,
                                  u=ep["u"], u_advantage=ep["advantage"], win=win, win_advantage=win-other_win,
                                  physical_score_roots=len(traj), tape_orders_sha256=digest))
                n_steps += ep["official_transitions"]
        grad, counts, errors = a.gradient(roots, models, base)
        for i, (name, vec) in enumerate(zip(("it50", "sapling"), grad)):
            o.require(ref["parameter_counts_by_tensor"][name] == counts[i], "trainable parameter set changed")
            vectors[i].append(vec)
        map_rows.append(dict(seed=seed, terrain=ep["terrain"], games=games, **errors))
        print(seed, len(roots), flush=True)
    o.require(n_steps == 7176 and sum(v["supported"] for v in map_rows) == 275, "schedule coverage changed")
    margin = {name: a.map_comparison(vs, seeds) for name, vs in zip(("it50", "sapling"), vectors)}
    win = {name: zero_direction(seeds, m["parameters"]) for name, m in margin.items()}
    all_results = {"previous_u": ref["towers"], "official_margin_div_1000": margin, "win_only": win}
    screens = {key: all(t["pass_screen"] for t in towers.values()) for key, towers in all_results.items()}
    o.save(HERE / "result.json", dict(
        program_sha256=o.sha(Path(__file__)), behavior_sha256=ref["parent_sha256"], manifest_sha256=TRAIN,
        registration_sha256=a.PROTOCOL, u_reference_sha256=U_SOURCE, replay_and_gradient_code_sha256=ref["program_sha256"],
        per_map_score_gradient_denominator=48, map_seat_pairs=12, games=24, new_games=0, wins=0, steps=n_steps,
        scheduled=288, supported=275, map_details=map_rows, towers_by_reward=all_results, screen_pass_by_reward=screens,
        joint_user_objectives_supported=screens["win_only"] and screens["official_margin_div_1000"], optimizer_steps=0,
        win_gradient_proof="For each of the twelve same-map same-seat sibling pairs both official scores yield W=0 in both draws; A_W=0 multiplies each score gradient exactly. Angle with zero is undefined.",
        assessment="Within these 24 all-loss training trajectories, an official margin improvement can be measured. A positive-vs-negative win signal and the joint test performance require separate evidence."
    ))
    print({k: [(n, x["positive_folds"], x["positive_of_15"], x["distinct_pair_dot_sum"]) for n, x in v.items()]
            for k,v in all_results.items()}, flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(621758460)
    main()

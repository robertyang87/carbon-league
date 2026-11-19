"""One parent-policy gradient update from paired official league returns.

The conditional choice is between two different physical joint orders. Prefix,
menu and safety branch are fixed within this single behavior-policy update.
Every supported log likelihood in a trajectory receives that trajectory's
paired return. Every scheduled but unsupported action has zero score gradient.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent))
from common import (BASE, EXP, PACK, PARENT_SHA, ROOT, FastCarbon, actual_features,
                    package, policy, read, save, sha, tensorbatch, torch_model)
from fit import VALUE, export, repack
from prepare_data import teacher_reader

LR = 1.e-4
LOGIT_TOL = .003
PROB_TOL = .00075
BATCH = 8


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def payoff(scores, seat):
    require(len(scores) == 2 and np.isfinite(scores).all(), "invalid official scores")
    margin = float(scores[seat] - scores[1-seat])
    wdl = float(margin > 0) + .5 * (margin == 0)
    return margin, .5 * wdl + .5 / (1 + math.exp(-float(np.clip(margin/1000, -50, 50))))


def decode_root(model, feat, x):
    enc = []
    for net in model._NETS:
        tok, ctx = net.encode(x, model._HYBRID["style"])
        enc.append((net, tok, ctx, net.lin(ctx, "h0")))
    base = model._return_decode(x, feat.aux, enc, False)
    alt = model._return_decode(x, feat.aux, enc, True)
    labels, reject = policy.guard(model, x, base, alt)
    return enc, labels, not reject


def capture(model, teacher, feat, x, event, cmd, meta):
    slot = event["slot"]
    opts = event["options"]
    labels = feat.labels(opts[0])
    moves = event["actions"]
    pick = event["choice"]
    require(0 <= slot < 10 and str(feat.own_uid[slot]) == event["uid"]
            and labels[:slot].tolist() == event["prefix"]
            and int(x["tok_kind"][slot]) == event["role"], "invalid worker or conditional prefix")
    require(len(opts) == len(moves) == len(set(moves)) == 2 and opts[0] != opts[1]
            and pick in (0, 1) and opts[pick] == event["issued"] == cmd, "joint command not sampled")
    enc, joint, route = decode_root(model, feat, x)
    require(route == event["route"] and feat.to_actions(joint) == opts[0], "safety branch or greedy action drift")
    for i, option in enumerate(opts):
        y = feat.labels(option)
        require(feat.to_actions(y) == option and y[slot] == moves[i]
                and y[:slot].tolist() == event["prefix"], "alternative prefix or physical action mismatch")

    row = actual_features(feat, x, opts[0], model)
    require(row.pop("cut") > slot and row["eligible"][slot] and row["legal"][slot].tolist() == event["legal"]
            and all(row["legal"][slot, moves]), "fitted focal action outside true legal support")
    fixed = np.zeros_like(row["residual"])
    parent = np.zeros_like(fixed)
    teacher(x, feat.aux, enc, route, labels, fixed, parent, slot + 1)
    logits = np.array(event["logits"], np.float64)
    q = np.array(event["probabilities"], np.float64)
    observed_gap = float(logits[0] - logits[1])
    native_gap = float(np.subtract(*parent[slot, moves].astype(np.float64)))
    checked_q = np.exp(logits-logits.max())
    checked_q /= checked_q.sum()
    require(np.isfinite(logits).all() and 0 <= observed_gap <= 1.000001
            and np.allclose(q, checked_q, atol=1e-12, rtol=0)
            and abs(math.log(q[pick])-event["logp"]) < 1e-12
            and int(event["uniform"] >= q[0]) == pick, "invalid logged choice probability")
    require(abs(native_gap-observed_gap) <= LOGIT_TOL, "independent native prefix does not reproduce observed odds")
    row["residual"] = fixed
    # Features originate before either command is run. Feat.observe replaces these
    # arrays on the next tick. A private copy makes readback insensitive to mutation.
    public_root = (copy.deepcopy(x), copy.deepcopy(feat.aux), list(feat.own_uid))
    receipt = dict(**meta, step=event["step"], slot=slot, moves=moves, pick=pick,
                   parent_order=opts[0], observed_gap=observed_gap, native_gap=native_gap,
                   observed_p0=q[0], old_logp=event["logp"], route=route)
    return row, receipt, public_root


def assemble(card, model, teacher, steps, advantage, registration, reg_sha):
    import collect as collector
    path = ROOT/card["path"]
    require(sha(path) == card["sha256"], "collection tape digest mismatch")
    g = read(path)
    seat, seed = card["seat"], card["seed"]
    require((g["seed"], g["candidate_seat"], g["initial_terrain_u8_sha256"]) == (seed, seat, card["terrain"])
            and len(g["actions"]) == g["steps"], "wrong game, terrain or length")
    actor_sha = registration["sampler_sha256"] if card["arm"] == "sampled" else registration["parent_sha256"]
    require(g["arm"] == card["arm"] and g["policy_sha256"] == actor_sha and g["registration_sha256"] == reg_sha
            and g["opponent_sha256"] == registration["opponents"][card["opponent"]]["sha256"], "policy or registration changed")
    env = collector.begin(seed, g["game_configuration"])
    require(hashlib.sha256(bytes(map(int, env.states[0].observation.full_carbon))).hexdigest() == card["terrain"],
            "replay opening terrain differs")
    sampled = card["arm"] == "sampled"
    events = {e["step"]: e for e in g["events"]} if sampled else {}
    if sampled:
        require(len(events) == len(g["events"]) and set(events) == {t for t in steps if t < g["steps"]},
                "missing or extra scheduled draw")
    F = sys.modules["cnet2_feat"]
    feat = F.Feat(seat) if sampled else None
    meta = dict(terrain=card["terrain"], seed=seed, seat=seat, opponent=card["opponent"], advantage=advantage)
    captured = []
    for t, cmds in enumerate(g["actions"]):
        require(not env.done, "transition after official terminal")
        if sampled:
            obs = collector.public(env, seat)
            require(obs["player"] == seat and obs["step"] == t and "full_carbon" not in obs, "non-public actor state")
            x = feat.observe(obs)
            if t in events:
                e = events[t]
                require(e["seat"] == seat and e["issued"] == cmds[seat] and e["public_sha256"] == digest(obs),
                        "observed decision or physical joint action differs from archive")
                if e["supported"]:
                    captured.append(capture(model, teacher, feat, x, e, cmds[seat], meta))
        env.step(cmds)
    require(env.done and [s.status for s in env.states] == ["DONE", "DONE"]
            and [s.reward for s in env.states] == g["official_scores_by_seat"], "official terminal replay mismatch")
    margin, u = payoff(g["official_scores_by_seat"], seat)
    require(margin == card["margin"] and abs(u - card["u"]) < 1e-12, "terminal utility differs from index")
    return captured


def evaluate(net, d, ids):
    b = tensorbatch(d, ids)
    slots = torch.as_tensor(d["slot"][ids])
    moves = torch.as_tensor(d["moves"][ids])
    index = torch.arange(len(ids))
    mixed = b["residual"] + .25*net(b)[0]
    projected = mixed[index, slots].gather(1, moves)
    return projected[:, 0] - projected[:, 1]


def errors(net, data, rows):
    with torch.inference_mode():
        gaps = np.concatenate([evaluate(net, data, ix).numpy().astype(np.float64)
                              for ix in groups(len(rows))])
    probs = 1/(1+np.exp(-gaps))
    observed_gaps = np.asarray([r["observed_gap"] for r in rows])
    observed_probs = np.asarray([r["observed_p0"] for r in rows])
    return gaps, probs, dict(roots=len(rows), max_signed_logit_difference_error=float(np.max(abs(gaps-observed_gaps))),
                             max_probability_error=float(np.max(abs(probs-observed_probs))),
                             max_native_logit_difference_error=max(abs(r["native_gap"]-r["observed_gap"]) for r in rows))


def groups(size):
    return [np.arange(i, min(i+BATCH, size), dtype=np.int64) for i in range(0, size, BATCH)]


def fit(net, data, rows, tickets, per_ticket):
    for name, param in net.named_parameters():
        param.requires_grad_(not name.startswith(VALUE))
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0, foreach=False)
    optimizer.zero_grad(set_to_none=True)
    normalizer = tickets*per_ticket
    objective = 0.
    for ids in groups(len(rows)):
        gaps = evaluate(net, data, ids)
        # Use the exact acting probability at theta0 and the differentiable
        # Torch Jacobian. Independent native-vs-Torch parity is checked before
        # this correction. All minibatch sums precede the lone optimizer step.
        native = torch.tensor([rows[i]["observed_gap"] for i in ids], dtype=torch.float64)
        mixed = native + (gaps.double() - gaps.detach().double())
        sign = torch.tensor([1. if rows[i]["pick"] == 0 else -1. for i in ids], dtype=mixed.dtype)
        advantage = torch.tensor([rows[i]["advantage"] for i in ids], dtype=mixed.dtype)
        loss = -(torch.nn.functional.logsigmoid(sign*mixed)*advantage).sum() / normalizer
        require(torch.isfinite(loss).item(), "non-finite likelihood objective")
        loss.backward()
        objective += float(loss.detach())
    groups_norm = {}
    for name, param in net.named_parameters():
        if param.grad is not None:
            family = name.split(".")[0]
            groups_norm[family] = groups_norm.get(family, 0.) + float(torch.sum(param.grad.double()**2))
    groups_norm = {k: math.sqrt(v) for k, v in groups_norm.items()}
    require(groups_norm.get("stem", 0) > 0 and groups_norm.get("gru", 0) > 0 and groups_norm.get("whead", 0) > 0,
            "terminal policy gradient did not reach trunk, decoder and worker readout")
    bias_gradient = net.whead.bias.grad.detach().tolist()
    preclip = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
    optimizer.step()
    return dict(updates=1, lr=LR, weight_decay=0, normalization_denominator=normalizer, objective=objective,
                unclipped_worker_bias_gradient=bias_gradient, norm_before_clip=preclip, norm_after_clip=min(1., preclip), gradient_norm_by_module=groups_norm)


def archive_game_cards(manifest):
    registry = read(FOLDER/"clearance.json")
    games = manifest["games"]
    require(len(games) == 72 and len({r["sha256"] for r in games}) == 72, "collection incomplete or duplicated")
    by_pair = {}
    for r in games:
        key = (r["seed"], r["seat"], r["opponent"])
        pair = by_pair.setdefault(key, {})
        require(r["arm"] in ("sampled","greedy") and r["arm"] not in pair, "duplicate or unknown arm")
        pair[r["arm"]] = r
    require(len(by_pair) == 36 and all(set(a) == {"sampled", "greedy"} for a in by_pair.values()), "unpaired game")
    require(len({r["terrain"] for r in games}) == 18, "unexpected unique terrain count")
    require(all(p["sampled"]["terrain"] == p["greedy"]["terrain"] for p in by_pair.values()), "unmatched pair terrain")
    train_seeds = registry["groups"]["train"]
    opponents = ("it20", "it40", "it50")
    expected = {(seed, seat, opponents[i % 3]) for i, seed in enumerate(train_seeds) for seat in (0, 1)}
    require(set(by_pair) == expected, "train map, seat or frozen opponent differs from preregistration")
    require(all(r["terrain"] == registry["by_seed"][str(r["seed"])] for r in games), "terrain differs from preregistration")
    return by_pair


def run():
    clock = time.monotonic()
    rule = read(FOLDER/"protocol.json")
    steps = rule["collection"]["sample_steps"]
    require(len(steps) == len(set(steps)) == 12 and rule["initial_actor"] == sha(PACK) == PARENT_SHA, "wrong base or schedule")
    import collect as collector
    manifest = read(FOLDER/"collection.json")
    reg_path = FOLDER/"collection_registration.json"
    registration = read(reg_path)
    reg_sha = sha(reg_path)
    require(manifest["status"] == "complete" and manifest["registration_sha256"] == reg_sha
            and manifest["protocol_sha256"] == sha(FOLDER/"protocol.json")
            and registration == json.loads(json.dumps(collector.registration(manifest["protocol_sha256"]))),
            "collector, input engine, rules or frozen actors changed")
    by_pair = archive_game_cards(manifest)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261119)
    torch.use_deterministic_algorithms(True)
    model = package()
    with np.load(BASE, allow_pickle=False) as parent_weights:
        require(repack.quantize(BASE, parent_weights) == repack.literal(PACK.read_text(), repack.STUDENT),
                "editable float source does not reproduce deployed starting student")
    teacher = teacher_reader(model)
    net = torch_model(model, 1)
    all_roots = []
    tickets = []
    for key in sorted(by_pair):
        arms = by_pair[key]
        sampled, base = arms["sampled"], arms["greedy"]
        a = sampled["u"] - base["u"]
        assemble(base, model, teacher, steps, 0., registration, reg_sha)
        captured = assemble(sampled, model, teacher, steps, a, registration, reg_sha)
        all_roots.extend(captured)
        tickets.append(dict(seed=key[0], seat=key[1], opponent=key[2], terrain=sampled["terrain"],
                            A=a, sampled_U=sampled["u"], baseline_U=base["u"], supported=len(captured)))
        print("paired", len(tickets), "/", len(by_pair), "supported", len(all_roots), flush=True)
    require(all_roots, "no stochastic support")
    features, rows, public = zip(*all_roots)
    d = {k:np.stack([r[k] for r in features]) for k in features[0]}
    d.update(slot=np.array([r["slot"] for r in rows]), moves=np.array([r["moves"] for r in rows]))
    _, old_probs, parity = errors(net, d, rows)
    require(parity["max_signed_logit_difference_error"] <= LOGIT_TOL and parity["max_probability_error"] <= PROB_TOL,
            f"failed pre-fit Torch likelihood parity {parity}")
    before = {k: p.detach().clone() for k, p in net.named_parameters()}
    trained = fit(net, d, rows, len(by_pair), len(steps))
    trained["updated_float_l2"] = math.sqrt(sum(float(torch.sum((p.detach()-before[k]).double()**2))
                                                 for k, p in net.named_parameters()))
    del before
    released = export(net, model, "candidate", FOLDER)
    child = package(FOLDER/"candidate.py")
    require(sha(FOLDER/"candidate.py") == released["package_sha256"] and sha(FOLDER/"candidate.npz") == released["weights_sha256"],
            "published actor or weights changed")
    require(released["package_bytes"] <= 10_000_000 and released["quantized_changed"] > 0, "deployment export failed")
    gap_child, prob_child, _ = errors(torch_model(child,1), d, rows)
    audit = []
    teacher_reader_child = teacher_reader(child)
    changes = [0, 0]
    for idx,(r,(x,aux,units)) in enumerate(zip(rows,public)):
        # Unconstrained native serving path on that exact, observed history.
        feat = sys.modules["cnet2_feat"].Feat(r["seat"])
        feat.aux, feat.own_uid = aux, units
        enc, labels, child_route = decode_root(child, feat, x)
        require(np.array_equal(labels, child._mixed_actions(x,aux)), "readback differs from native packaged decision")
        issued = feat.to_actions(labels)
        child_features = actual_features(feat, x, issued, child)
        require(child_features["cut"] == len(labels), "free child action cannot be encoded as legal")
        old_prefix = feat.labels(r["parent_order"])
        fixed = np.zeros_like(child_features["residual"])
        combined = np.zeros_like(fixed)
        # Density shift is a projection onto the old logged menu+route, even
        # when the new free policy changes prefix/consequent menu or guard.
        teacher_reader_child(x, aux, enc, r["route"], old_prefix, fixed, combined, r["slot"]+1)
        actual_gap = float(np.subtract(*combined[r["slot"],r["moves"]].astype(np.float64)))
        require(abs(actual_gap-gap_child[idx]) <= LOGIT_TOL, "quantized checkpoint projection differs from live pack")
        child_p = 1/(1+math.exp(-actual_gap))
        changed = issued != r["parent_order"]
        changes[r["seat"]] += int(changed)
        audit.append(dict(**r, child_free_order=issued, changed=changed, child_free_route=child_route,
                          parent_torch_p0=float(old_probs[idx]), projected_child_native_gap=actual_gap,
                          projected_child_native_p0=child_p, projected_child_torch_p0=float(prob_child[idx]),
                          projected_shift=child_p-r["observed_p0"]))
    audit_file = FOLDER/"decision_audit.json.gz"
    save(audit_file, audit)
    save(FOLDER/"fit.json",dict(operator="one_full_batch_paired_terminal_policy_gradient", protocol_sha256=sha(FOLDER/"protocol.json"),
         collection_sha256=sha(FOLDER/"collection.json"), optimizer_sha256=sha(Path(__file__)), parent_sha256=sha(PACK),
         package_sha256=released["package_sha256"], export=released, parity=parity, training=trained,
         decisions=len(rows), sampled_episodes=len(tickets), tickets=tickets, audit_sha256=sha(audit_file),
         parent_weights_sha256=sha(BASE), seed_manifest_sha256=sha(FOLDER/"clearance.json"),
         registration_sha256=reg_sha, torch_version=torch.__version__, numpy_version=np.__version__,
         probability_audit="Conditional on the original two-command menu and route; emitted free choices and route separately audited.",
         illegal_free_command_roots=0, fitted_decisions_with_distinct_physical_orders=len(rows),
         changed_free_joint_by_seat=changes, physical_gate=all(v>0 for v in changes),
         max_abs_projected_probability_shift=max(abs(r["projected_shift"]) for r in audit), train_seconds=time.monotonic()-clock))
    print("export", released["package_sha256"], "actual changes", changes, flush=True)


if __name__ == "__main__":
    run()

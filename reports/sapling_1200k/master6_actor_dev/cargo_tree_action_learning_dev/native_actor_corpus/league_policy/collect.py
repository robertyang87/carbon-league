"""Collect complete paired training games with signed public policies and the original interpreter."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
FIT = HERE.parent.parent
ROOT = next(folder for folder in HERE.parents if (folder / "code/env_src").is_dir())
ENGINE = ROOT / "code/env_src/carbon_challenge-master"
sys.path.insert(0, str(ENGINE))
from zerosum_env import make

SPEC = ENGINE / "zerosum_env/envs/carbon/carbon.json"
CFG = {k: v.get("default") if isinstance(v, dict) else v
       for k, v in json.loads(SPEC.read_bytes())["configuration"].items()}
MODS = ("cnet2_feat", "cnet2_np", "fastsim", "cnet2_search")
PARENT = FIT / "candidate.py"
SAMPLER = FIT / "onpolicy_native/training_policy.py"
STEPS = tuple(range(12, 277, 24))
OPP_ORDER = ("it20", "it40", "it50")


def canonical_bytes(doc):
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_gzip(path, raw):
    write_bytes(path, gzip.compress(canonical_bytes(raw), compresslevel=6, mtime=0))


def write_json(path, doc):
    write_bytes(path, (json.dumps(doc, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


@contextmanager
def attached(modules):
    """Both single-file exports may use the same sys.modules names at inference."""
    saved = {name: sys.modules.pop(name, None) for name in MODS}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name in MODS:
            sys.modules.pop(name, None)
        sys.modules.update({k: v for k, v in saved.items() if v is not None})


class Player:
    def __init__(self, path):
        spec = importlib.util.spec_from_file_location("league_player", path)
        self.mod = importlib.util.module_from_spec(spec)
        with attached({}):
            spec.loader.exec_module(self.mod)
            self.modules = {k: sys.modules[k] for k in MODS if k in sys.modules}
            self.modules.update(getattr(self.mod, "_P", {}))
        self.wall = []
        self.cpu = 0.

    def act(self, obs, cfg):
        wall, cpu = time.perf_counter(), time.process_time()
        with attached(self.modules):
            cmd = self.mod.agent(obs, cfg)
        self.wall.append(time.perf_counter() - wall)
        self.cpu += time.process_time() - cpu
        if not isinstance(cmd, dict):
            raise ValueError("Public policy returned a non-dictionary action")
        _, bases, workers, _ = obs["players"][obs["player"]]
        for uid, action in cmd.items():
            allowed = ("RECCOLLECTOR", "RECPLANTER", "NONE") if uid in bases else ("UP", "DOWN", "LEFT", "RIGHT", "NONE") if uid in workers else ()
            if action not in allowed:
                raise ValueError(f"Invalid public order {uid} {action}")
        return dict(cmd)

    def diagnostics(self):
        stats = getattr(getattr(self.mod, "_AG", None), "stats", {}) or {}
        errors = {k: int(v) for k, v in stats.items() if "error" in k and isinstance(v, (int, np.integer))}
        if any(errors.values()):
            raise ValueError(f"Runtime error reported by packaged policy: {errors}")
        return dict(calls=len(self.wall), wall_s=sum(self.wall), cpu_s=self.cpu,
                    max_wall_s=max(self.wall), p95_wall_s=float(np.percentile(self.wall, 95)),
                    errors=errors)


def begin(seed, game_configuration):
    np.random.seed(seed)
    random.seed(seed)
    return make("carbon", configuration={**game_configuration, "randomSeed": seed})


def public(env, seat):
    shared = env.states[0].observation
    private = env.states[seat].observation
    return dict(player=seat, step=shared.step, players=shared.players, trees=shared.trees,
                carbon=private.carbon, remainingOverageTime=private.remainingOverageTime)


def utility(margin):
    wdl = 1. if margin > 0 else .5 if margin == 0 else 0.
    exp_neg = math.exp(-abs(margin / 1000))
    sig = (1. if margin >= 0 else exp_neg) / (1. + exp_neg)
    return .5 * (wdl + sig)


def verify_event(event, obs, cmd, seed):
    if event["issued"] != cmd or (event["step"], event["seat"]) != (obs["step"], obs["player"]):
        raise ValueError("Sampler order/turn not executed")
    if event["supported"]:
        prob = np.exp(np.subtract(event["logits"], max(event["logits"])))
        prob /= prob.sum()
        draw = float(np.random.default_rng(np.random.SeedSequence(
            [9173951, seed, obs["player"], obs["step"]])).random())
        pick = int(draw >= prob[0])
        choices, options, legal = event["actions"], event["options"], event["legal"]
        if (not np.allclose(prob, event["probabilities"], rtol=0., atol=1e-12)
                or not math.isclose(math.log(prob[pick]), event["logp"], abs_tol=1e-12)
                or pick != event["choice"] or draw != event["uniform"]
                or len(set(choices)) != 2 or not all(legal[a] for a in choices)
                or len(options) != 2 or options[0] == options[1] or options[pick] != cmd):
            raise ValueError("Invalid behavior distribution, support or physical command")
    event["public_sha256"] = hashlib.sha256(canonical_bytes(obs)).hexdigest()


def play(seed, seat, terrain_sha, opponent_path, policy_path, arm, sample_steps=()):
    """One complete game using public observations, same global RNG initialization in both arms."""
    if seat not in (0, 1) or arm not in ("sampled", "greedy"):
        raise ValueError("Invalid policy role")
    os.environ["SAPLING_NATIVE_STREAM"] = str(seed)
    os.environ.pop("SAPLING_NATIVE_TRACE", None)
    cfg = {**CFG, "randomSeed": seed}
    env = begin(seed, cfg)
    initial = [int(k) for k in env.states[0].observation.full_carbon]
    if hashlib.sha256(bytes(initial)).hexdigest() != terrain_sha:
        raise ValueError(f"Unexpected initial terrain on {seed}")
    own, opp = Player(policy_path), Player(opponent_path)
    if arm == "sampled":
        own.mod._SAMPLE_STEPS = tuple(sample_steps)
    players = [own, opp] if seat == 0 else [opp, own]
    events, actions = [], []
    time0 = time.perf_counter()
    while not env.done:
        t = int(env.states[0].observation.step)
        if t != len(actions) or t >= cfg["episodeSteps"] - 1:
            raise ValueError(f"Unexpected live interpreter step {t}")
        both = [public(env, s) for s in (0, 1)]
        cmd = [model.act(both[s], cfg) for s, model in enumerate(players)]
        if arm == "sampled" and t in sample_steps:
            event = own.mod.last_training_event
            verify_event(event, both[seat], cmd[seat], seed)
            events.append(event)
        actions.append(cmd)
        env.step(cmd)
    status = [a.status for a in env.states]
    scores = [float(a.reward) for a in env.states]
    steps = int(env.states[0].observation.step)
    if status != ["DONE", "DONE"] or not all(map(math.isfinite, scores)) or steps != len(actions):
        raise ValueError(f"Incomplete official game {seed}/{seat}: {status}, {steps}")
    return dict(seed=seed, candidate_seat=seat, arm=arm, initial_terrain_u8_sha256=terrain_sha,
                initial_full_carbon=initial, game_configuration=cfg, actions=actions, events=events,
                official_score_by_seat=scores, official_scores_by_seat=scores, status=status, steps=steps,
                cash_by_seat=[p[0] for p in env.states[0].observation.players],
                wall_s=time.perf_counter()-time0, local_per_player_diagnostic=[p.diagnostics() for p in players],
                policy_sha256=sha_file(policy_path), opponent_sha256=sha_file(opponent_path))


def summarize(path, raw, opponent_id):
    scores, seat = raw["official_score_by_seat"], raw["candidate_seat"]
    margin = scores[seat] - scores[1-seat]
    return dict(seed=raw["seed"], seat=seat, candidate_seat=seat, terrain=raw["initial_terrain_u8_sha256"],
                opponent=opponent_id, arm=raw["arm"], path=str(path.relative_to(ROOT)), sha256=sha_file(path),
                official_score_by_seat=scores, actual_transitions=raw["steps"], u=utility(margin), margin=margin,
                result="win" if margin>0 else "loss" if margin<0 else "draw",
                supported=sum(r["supported"] for r in raw["events"]),
                alternate=sum(r.get("choice", 0) for r in raw["events"]))


def registration(protocol_sha):
    protocol, registry, maps = (HERE / k for k in ("protocol.json", "opponents.json", "clearance.json"))
    if sha_file(protocol) != protocol_sha:
        raise ValueError("Main protocol digest mismatch")
    plan, opponents, clearance = (json.loads(p.read_bytes()) for p in (protocol, registry, maps))
    if (tuple(plan["collection"]["sample_steps"]) != STEPS or clearance["source_snapshot"] != plan["starting_commit"]
            or clearance["seed_intersections"] or clearance["physical_intersections"] or sha_file(PARENT) != plan["initial_actor"]):
        raise ValueError("Protocol, maps or frozen policy mismatch")
    cards = {c["id"]:c for c in opponents["opponents"]}
    seeds = sorted(clearance["groups"]["train"])
    if len(set(seeds)) != plan["map_sets"]["train_count"] or len({clearance["by_seed"][str(s)] for s in seeds}) != len(seeds):
        raise ValueError("Training maps not unique")
    if any(sha_file(ROOT / cards[k]["path"]) != cards[k]["sha256"] for k in OPP_ORDER):
        raise ValueError("A frozen opponent changed")
    jobs = [dict(seed=s, seat=seat, arm=arm, opponent=OPP_ORDER[i%3], terrain=clearance["by_seed"][str(s)])
            for i,s in enumerate(seeds) for seat in (0, 1) for arm in ("greedy", "sampled")]
    if len(jobs) != 72:
        raise ValueError("Missing tickets")
    engine = {f:sha_file(ENGINE / f) for f in ("zerosum_env/core.py", "zerosum_env/envs/carbon/carbon.py",
                "zerosum_env/envs/carbon/helpers.py","zerosum_env/envs/carbon/idgen.py","zerosum_env/envs/carbon/carbon.json")}
    return dict(schema=1, protocol_sha256=protocol_sha, collector_sha256=sha_file(Path(__file__)),
                clearance_sha256=sha_file(maps), opponents_sha256=sha_file(registry),
                parent_sha256=sha_file(PARENT), sampler_sha256=sha_file(SAMPLER),
                sampler_source_sha256=sha_file(SAMPLER.with_name("sampler.py")), sample_steps=STEPS,
                engine_sha256=engine, opponents={k:{"path":v["path"],"sha256":v["sha256"]} for k,v in cards.items()},
                rng="numpy.random and random reset to map seed before official board creation in both arms; per-draw local stream keyed on seed, seat and turn. Both policies act in physical seat order. Tests compare both action tapes up to first trained alternate.", tickets=jobs)


def train_ticket(job):
    ticket, manifest, reg_digest = job
    seed, seat, arm = ticket["seed"], ticket["seat"], ticket["arm"]
    path = HERE / "train_games" / f"{seed}_{seat}_{arm}.json.gz"
    expected = manifest["sampler_sha256"] if arm == "sampled" else manifest["parent_sha256"]
    opp = manifest["opponents"][ticket["opponent"]]
    if path.is_file():
        raw = json.loads(gzip.decompress(path.read_bytes()))
    else:
        raw = play(seed, seat, ticket["terrain"], ROOT/opp["path"], SAMPLER if arm=="sampled" else PARENT, arm, STEPS)
        raw["registration_sha256"] = reg_digest
        write_gzip(path, raw)
    if ((raw["seed"],raw["candidate_seat"],raw["arm"],raw["initial_terrain_u8_sha256"]) !=
          (seed,seat,arm,ticket["terrain"]) or raw["registration_sha256"] != reg_digest or
          raw["policy_sha256"] != expected or raw["opponent_sha256"] != opp["sha256"]):
        raise ValueError(f"Ticket identity failure on {path}")
    return summarize(path, raw, ticket["opponent"])


def verify_pair(path_sampled, path_greedy):
    s,g = (json.loads(gzip.decompress((ROOT/p).read_bytes())) for p in (path_sampled, path_greedy))
    events = [e for e in s["events"] if e.get("choice")==1]
    end = events[0]["step"] if events else s["steps"]
    if s["actions"][:end] != g["actions"][:end]:
        raise ValueError(f"Control drift before first exploration {s['seed']}/{s['candidate_seat']}")
    if events:
        seat=s["candidate_seat"]
        if (g["actions"][end][seat] != events[0]["options"][0] or
            g["actions"][end][1-seat] != s["actions"][end][1-seat]):
            raise ValueError("First sampled change did not have the actual greedy command as control")
    elif s["steps"] != g["steps"]:
        raise ValueError("Outcome without exploration should replay exactly")
    return dict(seed=s["seed"],seat=s["candidate_seat"],verified_equal_steps=end,first_alternative_step=end if events else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze","run"))
    parser.add_argument("--protocol-sha", required=True)
    parser.add_argument("--registration-sha")
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    manifest=registration(args.protocol_sha)
    reg_path=HERE/"collection_registration.json"
    payload=(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    if args.mode=="freeze":
        if reg_path.exists() and reg_path.read_bytes()!=payload:
            raise ValueError("Refusing to overwrite a different registration")
        write_bytes(reg_path,payload)
        print(sha_file(reg_path),flush=True); return
    if reg_path.read_bytes()!=payload or sha_file(reg_path)!=args.registration_sha:
        raise ValueError("Unsigned or changed collection manifest")
    with (HERE/".collect.lock.tmp").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        tasks=[(j,manifest,args.registration_sha) for j in manifest["tickets"]]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            # Validate the very first paired run before advancing the other registered tickets.
            first=list(pool.map(train_ticket,tasks[:2]))
            verify_pair(first[1]["path"],first[0]["path"])
            rows=first+list(pool.map(train_ticket,tasks[2:]))
        pair_checks=[verify_pair(rows[i+1]["path"], rows[i]["path"]) for i in range(0,len(rows),2)]
        doc=dict(status="complete", protocol_sha256=args.protocol_sha,registration_sha256=args.registration_sha,
                 games=rows,paired_opening_checks=pair_checks,
                 results_by_arm={arm:dict(Counter(r["result"] for r in rows if r["arm"]==arm)) for arm in ("sampled","greedy")})
        write_json(HERE/"collection.json",doc)
        print(json.dumps(doc["results_by_arm"]),flush=True)


if __name__=="__main__":
    main()

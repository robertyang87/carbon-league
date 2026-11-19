"""Observe the original examiner at its public entry; never enter its search loop."""
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

SCHEMA = "original-master3s2-public-clock/1"
SOURCE_KEY = "reports/sapling_1200k/native_clock.py"
COUNTERS = ("searched", "single", "aborted", "changed")
THREAD_ENV = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
RETURNS = {"searched", "single", "depth_zero_return_base"}


def required(plan):
    """The signed source list determines coverage, even if a human-readable label is removed."""
    return plan is not None and SOURCE_KEY in plan["protocol_sha256"]


def assignments(plan, cpus):
    """Keep the same CPU after resuming; alternate cores for each physical seat on successive maps."""
    if not required(plan):
        return {}
    if len(cpus) != 2 or len(set(cpus)) != 2:
        raise ValueError("expected two distinct worker CPUs")
    return {(m["seed"], seat): cpus[(i + seat) % 2]
            for i, m in enumerate(plan["maps"]) for seat in (0, 1)}


def host_snapshot(master):
    try:
        from threadpoolctl import threadpool_info
        pools = [{key: p.get(key) for key in ("user_api", "internal_api", "num_threads", "version")}
                 for p in threadpool_info()]
        pool_error = None
    except Exception as exc:
        # This optional inventory must never decide whether the original examiner gets to act.
        pools = []
        pool_error = f"{type(exc).__name__}: {exc}"

    group = next(line[3:] for line in Path("/proc/self/cgroup").read_text().splitlines()
                 if line.startswith("0::"))
    root = Path("/sys/fs/cgroup")
    cpu = root / group.lstrip("/")
    if not (cpu / "cpu.max").is_file():
        cpu = root
    stat = dict(line.split() for line in (cpu / "cpu.stat").read_text().splitlines())
    model = next((s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines()
                  if s.startswith(("model name", "Hardware"))), platform.machine())
    masks = Counter()
    for thread in Path("/proc/self/task").iterdir():
        try:
            masks[tuple(sorted(os.sched_getaffinity(int(thread.name))))] += 1
        except ProcessLookupError:
            continue
    return dict(hostname=platform.node(), cpu_model=model, kernel=platform.release(),
                pid=os.getpid(), affinity=sorted(os.sched_getaffinity(0)),
                thread_affinity=[dict(cpus=list(mask), threads=count) for mask, count in sorted(masks.items())],
                torch_threads=master._torch.get_num_threads(),
                torch_interop_threads=master._torch.get_num_interop_threads(),
                thread_env={key: os.environ.get(key) for key in THREAD_ENV},
                pools=pools, pool_error=pool_error,
                cgroup=group, cpu_max=(cpu / "cpu.max").read_text().strip(),
                cpu_stat={key: int(value) for key, value in stat.items()},
                perf_counter=time.perf_counter(), process_cpu_seconds=time.process_time())


def counters(stats):
    return tuple(stats[k] for k in COUNTERS)


class Recorder:
    def __init__(self, master):
        self.master = master
        self.before = counters(master._S_STATS)
        self.start = host_snapshot(master)
        self.calls = []
        self.errors = []

    def record(self, step, wall, cpu):
        stats = self.master._S_STATS
        after = counters(stats)
        searched, single, aborted, changed = [n - p for n, p in zip(after, self.before)]
        self.before = after
        if self.master._S_REC is not None:
            raise ValueError("per-candidate recording was enabled")
        if (searched, single, aborted, changed) == (0, 1, 0, 0):
            kind, candidates, depth, rolls, native = "single", 1, 0, 0, None
        else:
            candidates, depth, rolls, native, _ = stats["last"]
            if (searched, single, aborted, changed) == (0, 0, 1, 0) and depth == 0:
                kind = "depth_zero_return_base"
            elif (searched, single) == (1, 0) and aborted in (0, 1) and changed in (0, 1) and depth > 0:
                kind = "searched"
            else:
                raise ValueError("unclassified original search return")
        self.calls.append(dict(step=step, kind=kind, candidates=candidates, max_depth=depth,
                               rolls=rolls, native_seconds=native, native_aborted=bool(aborted),
                               changed=bool(changed), public_wall_seconds=wall, process_cpu_seconds=cpu))

    def wrap(self, agent):
        def observed(obs, cfg):
            cpu = time.process_time()
            start = time.perf_counter()
            action = agent(obs, cfg)
            wall = time.perf_counter() - start
            cpu = time.process_time() - cpu
            try:
                self.record(obs["step"], wall, cpu)
            except Exception as exc:
                # Preserve the policy's actual action and finish the game. Verification rejects the receipt.
                self.errors.append([obs["step"], repr(exc)])
            return action
        return observed


def summarize(calls, budget, max_depth):
    # Include the single and zero-depth returns that are omitted from the examiner's searched-only histograms.
    return dict(candidate_count=dict(Counter(str(c["candidates"]) for c in calls)),
                max_completed_depth=dict(Counter(str(c["max_depth"]) for c in calls)),
                return_types=dict(Counter(c["kind"] for c in calls)),
                native_aborted=sum(c["native_aborted"] for c in calls),
                non_single_below_reachable_depth=sum(c["kind"] != "single" and
                    c["max_depth"] < min(max_depth, 299 - c["step"]) for c in calls),
                native_seconds_over_budget=sum(c["native_seconds"] is not None and
                    c["native_seconds"] > budget for c in calls),
                public_seconds_over_budget=sum(c["public_wall_seconds"] > budget for c in calls),
                native_budget_seconds=budget)


def clocked_play(job):
    import tree_audited_game as arena
    from compound_formal import D1
    from turning_point_ledger import write

    play = arena.play
    if job[-1] == D1:
        from d1_sapling_arena import play
    load = arena.prior.shared.module
    records = []

    def observed_load(path, name):
        master = load(path, name)
        if name == "master_original" and Path(path) == Path(arena.prior.SOURCE):
            record = Recorder(master)
            master.agent = record.wrap(master.agent)
            records.append(record)
        return master

    arena.prior.shared.module = observed_load
    try:
        row = play(job)
    finally:
        arena.prior.shared.module = load
    if len(records) != 1:
        raise ValueError("original examiner public entry did not have exactly one recorder")
    rec = records[0]
    end = host_snapshot(rec.master)
    filename = f"native_clock_{row['seed']}_{row['candidate_seat']}.json.gz"
    receipt = dict(schema=SCHEMA, seed=row["seed"], candidate_seat=row["candidate_seat"],
                   examiner_md5=arena.prior.EXPECTED_MD5, pack_sha256=job[-1], weight_identity=job[1],
                   raw_sha256=row["raw_sha256"], host_start=rec.start, host_end=end,
                   calls=rec.calls, errors=rec.errors,
                   summary=summarize(rec.calls, rec.master._S_ABORT, rec.master._S_D2))
    sha = write(Path(job[-2]) / filename, receipt, compressed=True)
    row["native_examiner_clock"] = dict(file=filename, sha256=sha)
    return row


def validate_host(host, affinity):
    if (not all(isinstance(host.get(k), str) and host[k] for k in
                ("hostname", "cpu_model", "kernel", "cgroup", "cpu_max"))
            or host.get("affinity") != affinity
            or not all(type(host.get(k)) is int and host[k] > 0
                       for k in ("pid", "torch_threads", "torch_interop_threads"))
            or not host.get("thread_affinity")
            or not all(isinstance(x["cpus"], list) and x["cpus"] and type(x["threads"]) is int
                       and x["threads"] > 0 for x in host["thread_affinity"])
            or set(host.get("thread_env", {})) != set(THREAD_ENV)
            or "pool_error" not in host or not isinstance(host.get("pools"), list)
            or (host["pool_error"] is not None and (host["pools"] or not isinstance(host["pool_error"], str)
                                                    or not host["pool_error"]))
            or any(type(p["num_threads"]) is not int or p["num_threads"] <= 0 for p in host["pools"])
            or not {"usage_usec", "nr_throttled", "throttled_usec"} <= (host.get("cpu_stat") or {}).keys()
            or not all(type(x) is int and x >= 0 for x in host["cpu_stat"].values())):
        raise ValueError("missing or inconsistent native clock host measurements")


def duration(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def verify(directory, row, raw):
    expected = f"native_clock_{row['seed']}_{row['candidate_seat']}.json.gz"
    link = row.get("native_examiner_clock") or {}
    if link.get("file") != expected:
        raise ValueError("frozen run requires an original examiner clock receipt")
    path = Path(directory) / expected
    if not path.is_file():
        raise ValueError("original examiner clock receipt is missing")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != link.get("sha256"):
        raise ValueError("original examiner clock receipt bytes changed")
    clock = json.loads(gzip.decompress(data))
    identity = (SCHEMA, raw["seed"], raw["candidate_seat"], raw["examiner_md5"],
                raw["pack_sha256"], raw["weight_identity"], row["raw_sha256"], [])
    if tuple(clock.get(k) for k in ("schema", "seed", "candidate_seat", "examiner_md5",
                                    "pack_sha256", "weight_identity", "raw_sha256", "errors")) != identity:
        raise ValueError("original examiner clock identity or collection failed")
    start, end = clock["host_start"], clock["host_end"]
    for snap in (start, end):
        validate_host(snap, row["cpu_affinity"])
    if (any(start[k] != end[k] for k in ("pid", "hostname", "cgroup", "cpu_model"))
            or any(not duration(snap[k]) for snap in (start, end) for k in ("perf_counter", "process_cpu_seconds"))
            or any(end[k] < start[k] for k in ("perf_counter", "process_cpu_seconds"))
            or any(end["cpu_stat"].get(k, -1) < n for k, n in start["cpu_stat"].items())):
        raise ValueError("inconsistent native host timestamps or counters")

    steps = row["actual_transitions"]
    calls, events = clock["calls"], raw["search_calls"]
    if len(calls) != steps or len(events) != steps:
        raise ValueError("incomplete original examiner clock observations")
    hist = {k: Counter() for k in ("cands", "depth", "rolls")}
    native_times = []
    cutoff_count = changes = 0
    public_total = cpu_total = 0.
    cfg = raw["configuration"]
    for t, (call, event) in enumerate(zip(calls, events)):
        kind = call["kind"]
        k, d, rolls = call["candidates"], call["max_depth"], call["rolls"]
        native, wall, cpu = call["native_seconds"], call["public_wall_seconds"], call["process_cpu_seconds"]
        aborted, changed = call["native_aborted"], call["changed"]
        if (type(call["step"]) is not int or call["step"] != t or event != [t, kind] or kind not in RETURNS
                or any(type(n) is not int for n in (k, d, rolls))
                or not 1 <= k <= cfg["_S_K"] or not 0 <= d <= min(cfg["_S_D2"], 299 - t)
                or type(aborted) is not bool or type(changed) is not bool
                or not duration(wall) or not duration(cpu)
                or (kind == "single" and (k, d, rolls, native, aborted, changed) != (1, 0, 0, None, False, False))
                or (kind != "single" and (k < 2 or not cfg["_S_M"] <= rolls <= cfg["_S_MMAX"]
                                          or not duration(native) or native > wall + 1e-6))
                or (kind == "depth_zero_return_base" and (d, aborted, changed) != (0, True, False))
                or (kind == "searched" and d <= 0)):
            raise ValueError(f"invalid native examiner clock observation at step {t}")
        if kind == "searched":
            for key, value in (("cands", k), ("depth", d), ("rolls", rolls)):
                hist[key][str(value)] += 1
            native_times.append(native)
        cutoff_count += aborted
        changes += changed
        public_total += wall
        cpu_total += cpu
    stats = row["search"]
    if (clock["summary"] != summarize(calls, cfg["_S_ABORT"], cfg["_S_D2"])
            or any(dict(h) != {str(k): v for k, v in stats[name].items()} for name, h in hist.items())
            or native_times != stats["time"] or cutoff_count != stats["aborted"] or changes != stats["changed"]
            or public_total > row["call_seconds_by_seat"][1 - raw["candidate_seat"]]["sum"] + 1e-6
            or public_total > end["perf_counter"] - start["perf_counter"] + 1e-6
            or cpu_total > end["process_cpu_seconds"] - start["process_cpu_seconds"] + 1e-6):
        raise ValueError("per-step native clock disagrees with the original examiner or outer runner")

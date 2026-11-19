"""Paired public-entry actions, encoder isolation and single-thread local timing on an archived trajectory."""
import argparse
from collections import Counter
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import selectors
import subprocess
import sys
import time
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PACKS = {"incumbent": HERE.parent / "candidate.py", "constant": HERE / "constant.py", "student": HERE / "candidate.py"}
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def compact(obj):
    return json.dumps(obj, separators=(",", ":"), allow_nan=False).encode()


def array_hash(values):
    import numpy as np

    h = hashlib.sha256()
    for key in sorted(values):
        a = np.asarray(values[key])
        assert a.dtype != object
        h.update(compact([str(key), a.dtype.str, a.shape]))
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def trace(module, current):
    import numpy as np

    policy_class = sys.modules["cnet2_np"].Policy
    feature_module = sys.modules["cnet2_feat"]
    original_encode = policy_class.encode
    original_prefix_init = feature_module.Prefix.__init__

    def encode(self, x, style):
        role = module._NETS.index(self)
        assert role == len(current["encode_order"]) and style == module._HYBRID["style"]
        current["encode_order"].append(role)
        maps = x["maps"]
        assert maps.shape == (33, 225) and maps.dtype == np.uint8
        if role == 0:
            current["partner_x"] = array_hash(x)
            current["reference_maps"] = maps.copy()
            current["other_input"] = array_hash({**x, "maps": maps[1:]})
        else:
            assert current["other_input"] == array_hash({**x, "maps": maps[1:]})
            old = current["reference_maps"]
            visible = old[3] > 0
            assert np.array_equal(maps[0, visible], old[0, visible])
            fixed = (old[14:18] > 0).any(axis=0)
            observed_history = old[2] > 0
            eligible = ~observed_history & ~fixed
            assert np.array_equal(maps[0, ~eligible], old[0, ~eligible])
            current["eligible_cells"] = int(eligible.sum())
            current["visible_cells"] = int(visible.sum())
            current["hidden_changed_cells"] = int((maps[0] != old[0]).sum())
            current["student_carbon"] = array_hash({"c": maps[0]})
        tokens, context = original_encode(self, x, style)
        if role == 0:
            current["partner_encoding"] = array_hash({"tokens": tokens, "context": context})
        return tokens, context

    def prefix_init(self, x, aux):
        assert current["partner_x"] == array_hash(x)
        digest = array_hash(aux)
        assert digest == current.setdefault("prefix_aux", digest)
        current["prefix_initializations"] += 1
        return original_prefix_init(self, x, aux)

    policy_class.encode = encode
    feature_module.Prefix.__init__ = prefix_init


def worker(pack, mode, cpu):
    os.sched_setaffinity(0, {cpu})
    assert all(os.environ[k] == "1" for k in THREADS)
    raw = pack.read_bytes()
    mod = types.ModuleType("public_agent")
    mod.__file__ = str(pack)
    sys.modules[mod.__name__] = mod
    exec(compile(raw, str(pack), "exec"), mod.__dict__)
    import numpy as np

    current = {}
    if mode == "audit":
        trace(mod, current)
    ready = dict(pack_sha256=sha(raw), bytes=len(raw), python=platform.python_version(), numpy=np.__version__,
                 affinity=sorted(os.sched_getaffinity(0)), thread_env={k: os.environ[k] for k in THREADS},
                 cgroup={k: Path("/sys/fs/cgroup", k).read_text().strip()
                         for k in ("cpu.max", "memory.max", "cpuset.cpus.effective")})
    print(json.dumps(ready), flush=True)
    for line in sys.stdin:
        obs, cfg = json.loads(line)
        assert set(obs) == {"remainingOverageTime", "step", "player", "carbon", "trees", "players"}
        assert cfg.get("randomSeed") is None
        current.clear()
        if mode == "audit":
            current.update(encode_order=[], prefix_initializations=0)
        cpu_started = time.process_time_ns()
        started = time.perf_counter_ns()
        actions = mod.agent(obs, cfg)
        elapsed = (time.perf_counter_ns() - started) / 1e6
        cpu_elapsed = (time.process_time_ns() - cpu_started) / 1e6
        r = dict(t=obs["step"], action=actions)
        if mode == "audit":
            del current["reference_maps"]
            assert current.pop("encode_order") == [0, 1]
            assert current["prefix_initializations"] == 2
            r.update(current)
            if not obs["step"]:
                r["frozen"] = dict(hybrid=mod._HYBRID, partner_weights=array_hash(mod._NETS[0].w),
                                   student_weights=array_hash(mod._NETS[1].w), shared_source=mod._SRC_SHA256,
                                   return_coef=array_hash({"coef": mod._RETURN_THETA}))
        else:
            r.update(entry_ms=elapsed, process_cpu_ms=cpu_elapsed, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        print(json.dumps(r, separators=(",", ":")), flush=True)


def receive(proc):
    with selectors.DefaultSelector() as handle:
        handle.register(proc.stdout, selectors.EVENT_READ)
        if not handle.select(30):
            raise TimeoutError("agent did not respond within 30 seconds")
        data = proc.stdout.readline()
    if not data:
        raise RuntimeError(f"agent worker EOF, status {proc.poll()}")
    return json.loads(data)


def run_pack(pack, mode, cpu, frames, cfg):
    args = [sys.executable, "-I", "-u", str(__file__), "worker", str(pack), mode, str(cpu)]
    start = time.perf_counter()
    rows = []
    env = {**os.environ, **{k: "1" for k in THREADS}, "PYTHONDONTWRITEBYTECODE": "1"}
    with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0, env=env) as child:
        try:
            ready = receive(child)
            init_ms = (time.perf_counter() - start) * 1000
            assert ready["pack_sha256"] == sha(pack.read_bytes())
            for obs in frames:
                start = time.perf_counter()
                child.stdin.write(compact([obs, cfg]) + b"\n")
                child.stdin.flush()
                row = receive(child)
                wall_ms = (time.perf_counter() - start) * 1000
                assert row.pop("t") == obs["step"]
                if mode == "timing":
                    row["external_ms"] = wall_ms
                rows.append(row)
            child.stdin.close()
            assert child.wait(timeout=30) == 0
        finally:
            if child.poll() is None:
                child.kill()
    return dict(ready=ready, init_ms=init_ms, rows=rows)


def compare_arms(by_arm, frames):
    comparisons = {}
    for left, right in itertools.combinations(PACKS, 2):
        counts = Counter()
        changed = Counter()
        examples = []
        for s in (0, 1):
            for t, obs in enumerate(frames[s]):
                a = by_arm[left][s]["timing"]["rows"][t]["action"]
                b = by_arm[right][s]["timing"]["rows"][t]["action"]
                crew = obs["players"][s]
                roles = {**dict.fromkeys(crew[1], "base"), **{u: w[2].lower() for u, w in crew[2].items()}}
                assert set(a) <= roles.keys() and set(b) <= roles.keys()
                counts["turns"] += 1
                counts["changed_turns"] += a != b
                for u, role in roles.items():
                    counts["slots"] += 1
                    if a.get(u) != b.get(u):
                        changed[role] += 1
                        if len(examples) < 8:
                            examples.append([s, t, u, role, a.get(u, "NONE"), b.get(u, "NONE")])
        comparisons[f"{left}_to_{right}"] = dict(**counts, changed_slots=sum(changed.values()),
                                                    by_role=dict(changed), first_command_differences=examples)
    return comparisons


def verify(protocol, card, raw, cpu):
    import numpy as np

    sys.path.insert(0, str(ROOT / "code"))
    from fastsim import FastCarbon

    cfg = raw["game_configuration"]
    assert cfg.get("randomSeed") is None
    sim = FastCarbon(cfg).reset(full_carbon=raw["initial_full_carbon"])
    terrain = sha(bytes(map(int, sim.carbon)))
    assert terrain == raw["initial_terrain_u8_sha256"] == card["terrain"]
    frames = [[], []]
    for t, pair in enumerate(raw["actions"]):
        for s in (0, 1):
            frames[s].append(sim.obs(s))
            assert frames[s][-1]["step"] == t
        sim.step(*pair)
    assert t == 298 and np.allclose(sim.cash, raw["last_post_cash"], rtol=0, atol=.011)
    by_arm = {name: [{}, {}] for name in PACKS}
    # Reverse order across physical seats to reduce package-order confounding on the shared host.
    for mode in ("timing", "audit"):
        for s, order in enumerate([list(PACKS), list(reversed(PACKS))]):
            for name in order:
                print(f"public entry {mode} seat {s} {name}", flush=True)
                by_arm[name][s][mode] = run_pack(PACKS[name], mode, cpu, frames[s], cfg)

    invariant_rows = 0
    performance = {}
    maps = {}
    for name in PACKS:
        wall = []
        entry_wall = []
        entry_cpu = []
        rss = []
        cold = []
        changed_cells = []
        for s in (0, 1):
            measured = by_arm[name][s]["timing"]
            cold.append(measured["init_ms"])
            audited = by_arm[name][s]["audit"]
            reference = by_arm["incumbent"][s]["audit"]["rows"]
            assert measured["ready"] == audited["ready"]
            for t, (m, a, original) in enumerate(zip(measured["rows"], audited["rows"], reference, strict=True)):
                assert m["action"] == a["action"]
                if t == 0:
                    assert original["frozen"] == a["frozen"]
                for key in ("partner_x", "partner_encoding", "prefix_aux", "prefix_initializations", "other_input", "visible_cells", "eligible_cells"):
                    assert a[key] == original[key], (name, s, t, key)
                if name == "incumbent":
                    assert a["hidden_changed_cells"] == 0
                wall.append(m["external_ms"])
                entry_wall.append(m["entry_ms"])
                entry_cpu.append(m["process_cpu_ms"])
                rss.append(m["peak_rss_kib"])
                changed_cells.append(a["hidden_changed_cells"])
                invariant_rows += 1

        def ms(seq):
            return {**{f"p{p}": round(float(np.percentile(seq, p)), 3) for p in (50, 95, 99)},
                    "max": round(max(seq), 3), "mean": round(float(np.mean(seq)), 3)}

        performance[name] = dict(calls=len(wall), init_external_ms=cold,
                                 entry_ms=ms(entry_wall), process_cpu_ms=ms(entry_cpu), external_ms=ms(wall), external_gt_1s=sum(v > 1000 for v in wall),
                                 peak_rss_kib=max(rss), **by_arm[name][0]["timing"]["ready"])
        maps[name] = dict(changed_hidden_cells=sum(changed_cells), turns_with_changed_input=sum(v > 0 for v in changed_cells))
    differences = {}
    for name in ("constant", "student"):
        deltas = {}
        for metric in ("entry_ms", "external_ms", "process_cpu_ms"):
            deltas[metric] = ms([a[metric] - b[metric] for seat in (0, 1) for a, b in
                                 zip(by_arm[name][seat]["timing"]["rows"],
                                     by_arm["incumbent"][seat]["timing"]["rows"], strict=True)])
        differences[name + "_minus_incumbent"] = deltas

    result = dict(terrain=terrain, seed=raw["seed"], card=card, terminal_archived_cash=sim.cash,
                  protocol_sha256=sha(protocol.read_bytes()), runner_sha256=sha(Path(__file__).read_bytes()),
                  timing=performance, paired_step_timing_difference_ms=differences, input_checks=maps, invariant_rows=invariant_rows,
                  comparison=compare_arms(by_arm, frames),
                  method="Replay archived actions irrespective of public-entry output. Timing uses fresh isolated Python processes on one core with one numerical thread, serial JSONL from send through receive including serialization. A second independent run instruments every encoder and Prefix initializer and must reproduce each uninstrumented action. Cross-package checks compare partner inputs and encoded outputs including raw bytes, Sapling non-carbon inputs, all already observed or publicly fixed carbon map bytes, shared Prefix inputs and its initial aux arrays. Per-slot Prefix reserves can diverge after different decisions.",
                  environment=platform.platform(), host_cpu=next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')))
    return by_arm, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=int, default=4)
    args = parser.parse_args()
    protocol = HERE / "protocol.json"
    metadata = json.loads(protocol.read_text())
    card = next(c for c in metadata["cards"] if c["split"] == "val")
    source = (ROOT / card["path"]).read_bytes()
    assert sha(source) == card["sha256"]
    assert sha(PACKS["incumbent"].read_bytes()) == metadata["incumbent"]
    rows, summary = verify(protocol, card, json.loads(gzip.decompress(source)), args.cpu)
    detail = HERE / "normal_entry.json.gz"
    detail.write_bytes(gzip.compress(compact(rows), mtime=0))
    summary["detail_sha256"] = sha(detail.read_bytes())
    (HERE / "normal_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    if sys.argv[1:2] == ["worker"]:
        worker(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
    else:
        main()

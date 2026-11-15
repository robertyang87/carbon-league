import sys, os, time, json, argparse
sys.path.insert(0, "/root/carbon/k0pkg/code")
os.chdir("/root/carbon/k0pkg/code")
import k0_annotate as A

ap = argparse.ArgumentParser()
ap.add_argument("--weights"); ap.add_argument("--replays"); ap.add_argument("--games")
ap.add_argument("--out", required=True)
ap.add_argument("--window-lo", type=int, default=200)
ap.add_argument("--window-hi", type=int, default=299)
ap.add_argument("--belief", default="sample")
ap.add_argument("--teacher-style", type=int, default=2)
ap.add_argument("--K", type=int, default=16)
ap.add_argument("--M", type=int, default=8)
ap.add_argument("--H", type=int, default=2)
ap.add_argument("--seed-teacher", type=int, default=7000)
ap.add_argument("--train-temp", type=float, default=1.0)
ap.add_argument("--behavior-temp", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=12345)
ap.add_argument("--pi0", type=float, default=1.0)
ap.add_argument("--pi1", type=float, default=1.0)
ap.add_argument("--isolation-file", action="append", default=[])
ap.add_argument("--workers", type=int, default=1)
ap.add_argument("--accept-unbound-identity", action="store_true")
ap.add_argument("--verify", action="store_true")
a = ap.parse_args([
    "--weights", "/root/carbon/k0pkg/weights/cnet2_m3s2r1_heads_v2.npz",
    "--replays", "/root/carbon/k0pkg/replays",
    "--games", "/root/carbon/k0pkg/games_c2_timing.json",
    "--out", "/root/carbon/data/k0_annot_c2",
    "--isolation-file", "/root/carbon/k0pkg/iso_shard0.json",
    "--isolation-file", "/root/carbon/k0pkg/iso_test323.json",
])
plan = A.build_plan(a)
cfg = plan["config"]
A._init_worker(plan["weights_bytes"])
pg = plan["per_game"][0]
print("game", pg["game"], "n_roots", len(pg["roots"]), flush=True)
picks = [pg["roots"][0], pg["roots"][99], pg["roots"][100], pg["roots"][-1]]
for (step, seat) in picks:
    spec = {"game": pg["game"], "path": pg["path"], "step": step,
            "me": seat, "ft": pg["ft"]}
    t0 = time.time()
    r = A._eval_root(spec, cfg)
    dt = time.time() - t0
    print(f"root step={step} seat={seat} wall={dt:.2f}s arms={len(r['arms'])}", flush=True)

"""Stage 0 verification 1: bare-net self-play throughput on DGX (m3s2r1 vs itself).

Uses the EXISTING PPO lockstep rollout path (rl2_ppo_rollout.play_batch),
both sides = m3s2r1, NumPy trunk (CPU workers, no GPU inference server).
Run from code/ with:  env -u CNET2_TEMP_KIND CNET2_TEACHER=1 python3 stage0_measure_throughput.py <npz> [games=48] [lockstep=16]

Prints one JSON line per task + a THROUGHPUT summary line. Seeds 995000+
(fresh segment, disjoint from calibration 996000/999000/997000/999200/999600).
"""
import os
import sys
import time
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import rl2_ppo_rollout as RO  # noqa: E402


def main():
    npz = sys.argv[1]
    n_games = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    lockstep = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    seed0 = 995000
    t_all = time.time()
    tot_games = 0
    tot_steps = 0
    t_pol = t_opp = t_sim = 0.0
    ndec = 0
    wins = 0
    for off in range(0, n_games, lockstep):
        n = min(lockstep, n_games - off)
        games = [("SELF", seed0 + off + i, (off + i) % 2) for i in range(n)]
        task = {"adapter": "cnet2", "wkey": npz, "games": games,
                "temp": 1.0, "greedy": False, "record": False, "rng_seed": 12345 + off}
        r = RO.play_batch(task)
        recs = r["games"]
        uniq = [g for g in recs if not g.get("mirror")]
        tot_games += len(uniq)
        tot_steps += sum(g["steps"] for g in uniq)
        t_pol += r["t_pol"]
        t_opp += r["t_opp"]
        t_sim += r["t_sim"]
        ndec += r["ndec"]
        wins += sum(1 for g in uniq if g["margin"] > 0)
        print(json.dumps({"task": off // lockstep, "games": len(uniq),
                          "wall": round(r["wall"], 1),
                          "steps_mean": round(sum(g["steps"] for g in uniq) / max(len(uniq), 1), 1),
                          "t_pol": round(r["t_pol"], 1), "t_opp": round(r["t_opp"], 1),
                          "t_sim": round(r["t_sim"], 1)}), flush=True)
    wall = time.time() - t_all
    print("THROUGHPUT", json.dumps({
        "games": tot_games, "wall_s": round(wall, 1),
        "games_per_min": round(tot_games / wall * 60, 1),
        "steps_per_game": round(tot_steps / max(tot_games, 1), 1),
        "decisions": ndec,
        "t_pol_s": round(t_pol, 1), "t_opp_s": round(t_opp, 1), "t_sim_s": round(t_sim, 1),
        "side0_win_share": round(wins / max(tot_games, 1), 3)}), flush=True)


if __name__ == "__main__":
    main()

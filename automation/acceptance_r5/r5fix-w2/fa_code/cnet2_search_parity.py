"""cnet2_search_parity -- packed vs unpacked game-for-game: plays the same seeds with both files vs an
opponent and compares every action of the tested side plus the final state.
python3 cnet2_search_parity.py PACKED.py UNPACKED.py OPP.py SEED0 N"""
import os, sys, hashlib, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_v53
from fastsim import FastCarbon


def play(a_file, b_file, seed, side):
    fa = eval_v53.load_agent_fn(a_file); fb = eval_v53.load_agent_fn(b_file)
    sim = FastCarbon(eval_v53.CFG).reset(seed=seed); h = hashlib.sha1()
    for step in range(300):
        o = [sim.obs(0), sim.obs(1)]
        aa = fa(o[side], eval_v53.CFG); ab = fb(o[1 - side], eval_v53.CFG)
        h.update(json.dumps(aa, sort_keys=True).encode())
        sim.step(*((aa, ab) if side == 0 else (ab, aa)))
    return h.hexdigest(), [round(c, 2) for c in sim.cash]


if __name__ == "__main__":
    pk, up, opp, s0, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    ok = True
    for s in range(s0, s0 + n):
        r1 = play(pk, opp, s, s % 2); r2 = play(up, opp, s, s % 2)
        print(s, "packed", r1, "unpacked", r2, "SAME" if r1 == r2 else "DIFF", flush=True)
        ok &= r1 == r2
    print("ALL SAME" if ok else "MISMATCH")

"""Build from our frozen original-derived parent. The examiner stays byte-identical."""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = "7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Source anchor must occur exactly once: {before[:85]}")
    return text.replace(before, after)


def build():
    original = (HERE.parent / "cold_start/candidate.py").read_bytes()
    if sha(original) != PARENT:
        raise ValueError("Different source parent")
    src = original.decode().replace("\r\n", "\n")
    start = src.index("def _sample(")
    end = src.index("def _deviations(", start)
    sample = src[start:end]
    sample = once(sample, "step, fixed=None):", "step, fixed=None, choose=None):")
    sample = once(sample, "a = rng.choice(3, p=p)", "a = rng.choice(3, p=p) if choose is None else choose(bid, p)")
    sample = once(sample, "a = rng.choice(5, p=p)", "a = rng.choice(5, p=p) if choose is None else choose(wid, p)")
    src = src[:start] + (HERE / "rollout.py").read_text() + "\n\n" + sample + src[end:]
    play = src.index("    def play(live, pol, t):")
    src = src[:play] + "    coupled = {}\n\n" + src[play:]
    before = """            r1 = np.random.RandomState(seed0 + 7919 * t + 31 * m + 1); r2 = np.random.RandomState(seed0 + 7919 * t + 31 * m + 2)
            am = _sample(o['players'], me, lwm, lbm, r1, o['step'])
            ao = _sample(o['players'], op, lwp, lbp, r2, o['step'])"""
    after = """            key = (t, m)
            if key not in coupled:
                base_seed = seed0 + 7919 * t + 31 * m
                coupled[key] = (_identity_uniforms(base_seed + 1, players[me]),
                                _identity_uniforms(base_seed + 2, players[op]))
            own_draw, enemy_draw = coupled[key]
            am = _sample(o['players'], me, lwm, lbm, None, o['step'], choose=own_draw)
            ao = _sample(o['players'], op, lwp, lbp, None, o['step'], choose=enemy_draw)"""
    return once(src, before, after).replace("\n", "\r\n").encode()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = build()
    dest = HERE / "candidate.py"
    if args.verify:
        if dest.read_bytes() != output:
            raise ValueError("Pack differs from its builder")
    else:
        dest.write_bytes(output)
    print(sha(output), len(output))

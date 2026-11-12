"""Call the real actor decoder with isolated fixed logits; no game or weights needed."""
import json
import os
import sys
import pathlib

os.environ['CNET2_TEMP_KIND'] = '0.15,0.3,0.3'
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'code'))
import numpy as np
import rl2_ppo_cnet2 as a


class FixedPolicy:
    w = {'value.weightT': np.zeros((1, 1)), 'value.bias': np.zeros(1),
         'aemb.weight': np.zeros((9, 1))}

    def lin(self, x, name):
        if name == 'whead':
            return np.array([0, .1, .2, .3, .4])
        if name == 'V':
            return np.zeros((5, 1))
        return np.zeros(1)

    def gru(self, x, h):
        return np.zeros(1)


class Prefix:
    def __init__(self, *args):
        pass

    def feats(self, s):
        return np.zeros(a.F.UF), np.zeros((5, a.F.MF)), np.ones(5, bool)

    def advance(self, s, v):
        pass


class CaptureRNG:
    def choice(self, n, p):
        self.p = p.copy()
        return 0


actor = a.Cnet2Actor.__new__(a.Cnet2Actor)
actor.pol = FixedPolicy()
a.F.Prefix = Prefix
x = {'tok_mask': np.arange(a.F.OWN) == 0, 'tok_kind': np.zeros(a.F.OWN, int)}
probs = []
for temp in [.5, .7, 1.0]:
    rng = CaptureRNG()
    actor._decode(x, {}, rng, temp, False,
                  enc=(np.zeros((a.F.OWN, 1)), np.zeros(1)))
    probs.append(rng.p)
    print(json.dumps({'passed_opponent_temperature': temp,
                      'actual_action_probabilities': rng.p.round(6).tolist()}))
assert all(np.array_equal(probs[0], p) for p in probs)
print(json.dumps({'confirmed': 'snapshot temperature is overridden by global learner TEMP_KIND'}))

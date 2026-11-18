"""Architecture and native NPZ compatibility of both Sapling presets."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from cnet2_model import CNet2
from init_sapling import (DEFAULT_PRESET, EXPECTED_PARAMS, SAPLING_CONFIG,
                          SAPLING_PRESETS, initialize, resolve_preset_metadata)


class SaplingPresetTest(unittest.TestCase):
    def test_registered_parameter_counts(self):
        self.assertEqual(EXPECTED_PARAMS, 551187)
        for name, spec in SAPLING_PRESETS.items():
            with self.subTest(preset=name):
                net = CNet2(**spec["cfg"])
                self.assertEqual(sum(p.numel() for p in net.parameters()), spec["params"])
        self.assertEqual(SAPLING_PRESETS["sapling-1.2m"]["cfg"]["G"], 3)
        self.assertEqual(SAPLING_PRESETS["sapling-1.2m"]["cfg"]["C"], 93)

    def test_default_tensor_and_rng_compatibility(self):
        seed = 1234
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            expected = CNet2(**SAPLING_CONFIG).state_dict()
        rng = torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "default.npz"
            receipt = initialize(path, seed)  # original two-argument API
            self.assertEqual(receipt["preset"], DEFAULT_PRESET)
            with np.load(path, allow_pickle=False) as data:
                for name, tensor in expected.items():
                    np.testing.assert_array_equal(data[name], tensor.numpy())
            with self.assertRaises(FileExistsError):
                initialize(path, seed)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_large_export_strictly_roundtrips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "large.npz"
            initialize(path, 1234, "sapling-1.2m")
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data["meta"]))
                preset, cfg, count = resolve_preset_metadata(meta)
                net = CNet2(**cfg)
                net.load_state_dict({k: torch.from_numpy(data[k].copy()) for k in data.files if k != "meta"}, strict=True)
            self.assertEqual((preset, count), ("sapling-1.2m", 1206476))

    def test_legacy_metadata_and_invalid_identity(self):
        legacy = dict(cfg=dict(SAPLING_CONFIG), params=EXPECTED_PARAMS)
        legacy["cfg"].pop("TA")
        self.assertEqual(resolve_preset_metadata(legacy)[0], DEFAULT_PRESET)
        for patch in (dict(preset="sapling-1.2m"), dict(params=1206476),
                      dict(cfg=dict(SAPLING_CONFIG, C=88, NB=6)),
                      dict(cfg=dict(SAPLING_CONFIG, C=96, NB=6)),
                      dict(cfg=dict(SAPLING_CONFIG, NB=True)),
                      dict(cfg=dict(SAPLING_CONFIG, unknown=0))):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                resolve_preset_metadata(dict(legacy, **patch))

    def test_large_numpy_torch_forward_parity_both_seats(self):
        import cnet2_feat as F
        import cnet2_train as trainer
        from cnet2_np import Policy
        from eval_v53 import CFG
        from fastsim import FastCarbon

        torch.set_num_threads(2)
        trainer.DEV = torch.device("cpu")
        net = CNet2(**SAPLING_PRESETS["sapling-1.2m"]["cfg"]).eval()
        weights = {k: v.detach().numpy() for k, v in net.state_dict().items()}
        policy = Policy(dict(weights, meta={"cfg": net.cfg}))
        sim = FastCarbon(CFG).reset(seed=621000000)
        features = [F.Feat(p) for p in (0, 1)]
        actions = []
        for p in (0, 1):
            obs = sim.obs(p)
            features[p].observe(obs)
            actions.append({bid: ("RECCOLLECTOR" if i == 0 else "RECPLANTER")
                            for i, bid in enumerate(obs["players"][p][1])})
        sim.step(*actions)
        max_logit_error = max_value_error = 0.0
        for p in (0, 1):
            feat = features[p]
            x = feat.observe(sim.obs(p))
            y = feat.labels({})
            prefix = F.Prefix(x, feat.aux)
            uf = np.zeros((F.OWN, F.UF), np.float32)
            mf = np.zeros((F.OWN, 5, F.MF), np.float32)
            legal = np.zeros((F.OWN, 5), bool)
            for s in range(F.OWN):
                if x["tok_mask"][s]:
                    uf[s], mf[s], legal[s] = prefix.feats(s)
                    prefix.advance(s, int(y[s]))
            b = {k: np.asarray(x[k])[None] for k in
                 ("maps", "scal", "tok_pos", "tok_kind", "tok_mask", "tok_cargo", "hist_cell", "hist_cargo", "hist_move")}
            b.update(uf=uf[None], mf=mf[None], legal=legal[None], y=y[None],
                     teacher=np.array([2]), carbon=np.zeros((1, F.N), np.uint8), value=np.zeros(1, np.float32))
            with torch.no_grad():
                logits, value, _ = net(trainer.to_dev(b, aug=False))
            _, np_logits, np_value = policy.act(x, feat.aux, teacher=2, forced=y)
            mask = x["tok_mask"][:F.OWN, None] & legal
            max_logit_error = max(max_logit_error, float(np.abs(logits.numpy()[0][mask] - np_logits[mask]).max()))
            max_value_error = max(max_value_error, abs(float(value[0]) - np_value))
        self.assertLess(max_logit_error, 2e-5)
        self.assertLess(max_value_error, 2e-5)
        print(f"C93/G3 both-seat NumPy/Torch: logits max abs={max_logit_error:.3g}, value={max_value_error:.3g}")


if __name__ == "__main__":
    unittest.main()

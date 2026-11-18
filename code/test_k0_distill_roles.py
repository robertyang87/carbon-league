"""Role-loading regressions: independent CNet2 configs and bound provenance."""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("BC_DEVICE", "cpu")

import numpy as np
import torch

import k0_distill_u as D
from cnet2_model import CNet2


SMALL = dict(C=16, NB=1, D=16, NH=2, L=1, E=8, G=4)
LARGE = dict(C=24, NB=2, D=24, NH=3, L=1, E=8, G=4, TA=1, TNH=4)


class RoleModels(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def export(self, name, cfg, *, metadata=True, arrays_change=None, meta_raw=None):
        net = CNet2(**cfg)
        arrays = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        if arrays_change:
            arrays_change(arrays)
        if meta_raw is not None:
            arrays["meta"] = np.array(meta_raw)
        elif metadata:
            arrays["meta"] = np.array(json.dumps({"cfg": net.cfg}))
        path = self.root / name
        np.savez(path, **arrays)
        return path

    def read(self, path, role="--student-init"):
        return D._read_weights_file(str(path), role)

    def test_independent_architectures_and_identity(self):
        sp = self.export("student.npz", SMALL)
        ap = self.export("anchor.npz", LARGE)
        sw, aw = self.read(sp), self.read(ap, "--kl-reference")
        student, _, sr = D._build_role_model(sw, "--student-init", "cpu")
        anchor, _, ar = D._build_role_model(
            aw, "--kl-reference", "cpu", legacy_cfg=student.cfg)
        self.assertEqual(student.cfg["D"], 16)
        self.assertEqual(anchor.cfg["D"], 24)
        self.assertEqual(ar["cfg_source"], "meta.cfg")
        self.assertNotEqual(sr["cfg_sha256"], ar["cfg_sha256"])
        self.assertEqual(aw["sha256"], hashlib.sha256(ap.read_bytes()).hexdigest())
        for key, value in anchor.state_dict().items():
            np.testing.assert_array_equal(value.cpu().numpy(), aw["arrays"][key])

    def test_same_file_constructs_independent_models(self):
        w = self.read(self.export("same.npz", SMALL))
        student, _, sr = D._build_role_model(w, "--student-init", "cpu")
        anchor, _, ar = D._build_role_model(
            w, "--kl-reference", "cpu", legacy_cfg=student.cfg)
        self.assertEqual(sr["cfg_sha256"], ar["cfg_sha256"])
        before = anchor.value.weight.detach().clone()
        with torch.no_grad():
            student.value.weight.add_(1)
        self.assertTrue(torch.equal(before, anchor.value.weight))

    def test_legacy_cfg_call_and_anchor_compatibility_are_recorded(self):
        w = self.read(self.export("legacy.npz", SMALL, metadata=False))
        student, _, sr = D._build_role_model(
            w, "--student-init", "cpu", cfg_override=json.dumps(SMALL))
        anchor, _, ar = D._build_role_model(
            w, "--kl-reference", "cpu", legacy_cfg=student.cfg)
        self.assertEqual(sr["cfg_source"], "cli_cfg")
        self.assertEqual(ar["cfg_source"], "legacy_student_cfg_compatibility")
        self.assertEqual(student.cfg, anchor.cfg)

    def test_student_override_does_not_override_anchor_metadata(self):
        w = self.read(self.export("with_meta.npz", SMALL))
        override = dict(SMALL, TA=1, TNH=4)
        student, migration, sr = D._build_role_model(
            w, "--student-init", "cpu", cfg_override=json.dumps(override))
        anchor, _, ar = D._build_role_model(
            w, "--kl-reference", "cpu", legacy_cfg=student.cfg)
        self.assertEqual(student.cfg["TA"], 1)
        self.assertEqual(anchor.cfg["TA"], 0)
        self.assertTrue(migration["migrated_missing"])
        self.assertEqual(sr["cfg_source"], "cli_cfg")
        self.assertEqual(ar["cfg_source"], "meta.cfg")

    def test_missing_metadata_never_hides_shape_mismatch(self):
        w = self.read(self.export("legacy_large.npz", LARGE, metadata=False))
        with self.assertRaisesRegex(SystemExit, "--kl-reference.*do not fit"):
            D._build_role_model(w, "--kl-reference", "cpu", legacy_cfg=SMALL)

    def test_invalid_metadata_refused_instead_of_defaulting(self):
        for idx, meta in enumerate(("{broken", "[]", '{"cfg": []}')):
            with self.subTest(meta=meta):
                p = self.export(f"bad_meta{idx}.npz", SMALL, meta_raw=meta)
                with self.assertRaisesRegex(SystemExit, "--kl-reference.*meta"):
                    self.read(p, "--kl-reference")

    def test_bad_configuration_refused(self):
        w = self.read(self.export("normal.npz", SMALL))
        for cfg in (dict(SMALL, NH=3), [], "{bad"):
            with self.subTest(cfg=cfg), self.assertRaises(SystemExit):
                D._build_role_model(w, "--student-init", "cpu", cfg_override=cfg)

    def test_missing_policy_key_refused(self):
        p = self.export("missing.npz", SMALL,
                        arrays_change=lambda a: a.pop("stem.weight"))
        with self.assertRaisesRegex(SystemExit, "--student-init.*missing"):
            D._build_role_model(self.read(p), "--student-init", "cpu")

    def test_finite_source_overflow_during_load_refused(self):
        def poison(a):
            a["value.bias"] = np.full(a["value.bias"].shape, 1e300, np.float64)
        p = self.export("overflow.npz", SMALL, arrays_change=poison)
        with self.assertRaisesRegex(SystemExit, "--kl-reference.*non-finite"):
            D._build_role_model(self.read(p), "--kl-reference", "cpu")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

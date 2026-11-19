"""Confirm training-to-pack bindings without running any game or agent inference."""
import json
from pathlib import Path
import tempfile
import unittest

import attribution as check


class FrozenActorTests(unittest.TestCase):
    def test_real_files_and_substitution_controls(self):
        parent=check.build.PARENT
        single=check.TRAINED_AGENT
        hybrid=check.UPDATED_HYBRID
        original=json.loads(check.HYBRID_MANIFEST.read_bytes())
        single_header={"pack_sha256":check.build.digest(single),"weight_sha256":check.build.digest(check.TRAINED_NPZ)}
        hybrid_header={"pack_sha256":check.build.digest(hybrid),"weight_sha256":check.build.digest(check.HYBRID_MANIFEST)}
        check.prove_artifacts("single",single,single_header)
        self.assertEqual(set(check.prove_artifacts("hybrid",hybrid,hybrid_header)["student_payload_sha256"]),check.build.PAYLOAD_NAMES)
        with self.assertRaisesRegex(ValueError,"single-game"):
            check.prove_artifacts("single",parent,dict(single_header,pack_sha256=check.build.digest(parent)))
        with self.assertRaisesRegex(ValueError,"build manifest"):
            check.prove_artifacts("hybrid",hybrid,dict(hybrid_header,weight_sha256="0"*64))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            wrong_npz=path/check.TRAINED_NPZ.name
            wrong_npz.write_bytes((check.build.WEIGHTS/"cash_v3.npz").read_bytes())
            with self.assertRaisesRegex(ValueError,"actual FP32"):
                check.prove_artifacts("single",single,single_header,npz=wrong_npz)
            lines,fields=check.build.payloads(hybrid.read_bytes().decode())
            for field in sorted(check.build.PAYLOAD_NAMES):
                a,b=fields[field]
                fake=path/hybrid.name
                fake.write_text("".join(lines[:a]+[f"{field} = []\n"]+lines[b:]))
                changed=json.loads(json.dumps(original))
                changed["input_and_output_sha256"][str(hybrid.relative_to(check.build.ROOT))]=check.build.digest(fake)
                manifest=path/"forged.json";manifest.write_text(json.dumps(changed))
                forged_header=dict(pack_sha256=check.build.digest(fake),weight_sha256=check.build.digest(manifest))
                with self.subTest(field=field), self.assertRaisesRegex(ValueError,"student segment "+field):
                    check.prove_artifacts("hybrid",fake,forged_header,manifest=manifest)
            changed=json.loads(json.dumps(original))
            del changed["input_and_output_sha256"][str(check.TRAINED_NPZ.relative_to(check.build.ROOT))]
            manifest=path/"omitted.json";manifest.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,"does not bind"):
                check.prove_artifacts("hybrid",hybrid,dict(hybrid_header,weight_sha256=check.build.digest(manifest)),manifest=manifest)


if __name__=="__main__":
    unittest.main()

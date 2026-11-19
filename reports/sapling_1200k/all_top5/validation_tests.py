"""Exercise verification with deliberately corrupted and abnormal public-replay fixtures."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import validation as audit


class ValidationControls(unittest.TestCase):
    cache = Path("/workspace/sapling-all-top5-20261119")

    @classmethod
    def setUpClass(cls):
        cls.row = json.loads((audit.HERE / "manifest.json").read_bytes())["games"][0]
        with gzip.open(cls.cache / cls.row["path"]) as f:
            cls.env = json.load(f)

    def replay(self, env, meta=None):
        row = copy.deepcopy(meta or self.row)
        data = audit.encoded(env)
        packed = gzip.compress(data,mtime=0)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / row["path"]
            target.parent.mkdir(parents=True)
            target.write_bytes(packed)
            receipt = dict(id=row["id"], canonical_sha256=audit.sha(data),sha256=audit.sha(packed),seed=env["configuration"]["randomSeed"],
                           terminal=env["statuses"],environment_uuid=env["id"],transitions=len(env["steps"])-1)
            return audit.check_one(row,Path(folder),receipt)

    def test_unchanged(self):
        self.assertEqual(self.replay(self.env)["verdict"], "consistent")

    def test_changed_visibility_step_and_card(self):
        e=copy.deepcopy(self.env)
        obs=e["steps"][5][1]["observation"]
        obs["carbon"][0] += 1
        e["steps"][9][0]["observation"]["step"] += 1
        e["steps"][-1][0]["reward"] += 200
        e["rewards"][0] += 200
        result=self.replay(e)
        self.assertEqual(result["verdict"], "discrepancy")
        for label in ("visible_carbon_s1","step_sequence","terminal_card_score","official_reward_rule"):
            self.assertIn(label,result["issues"])

    def test_changing_authoritative_early_actions_is_detected(self):
        e=copy.deepcopy(self.env)
        e["steps"][1][0]["action"] = {}
        result=self.replay(e)
        self.assertIn("engine_seat_assets",result["issues"])
        self.assertEqual(result["issues"]["engine_seat_assets"]["first_frame"],1)

    def test_platform_timeout_separate_from_extinction(self):
        e=copy.deepcopy(self.env)
        row=copy.deepcopy(self.row)
        e["steps"]=e["steps"][:3]
        pair=e["steps"][-1]
        p=pair[0]["observation"]["players"][0]
        p[0],p[2],p[3] = 0,{},{}
        e["statuses"]=["TIMEOUT","DONE"]
        e["rewards"]=[0,e["steps"][-2][1]["reward"]]
        for seat in (0,1):
            pair[seat]["status"]=e["statuses"][seat]
            pair[seat]["reward"]=e["rewards"][seat]
        row["score_by_seat"]=e["rewards"]
        row["card"]["scores"]=dict(zip(row["card"]["order"],e["rewards"]))
        row["card"]["winner"]=row["card"]["order"][1]
        out=self.replay(e,row)
        self.assertEqual(out["verdict"],"consistent",out["issues"])
        self.assertEqual(out["terminal_kind"],"timeout")
        self.assertEqual(out["fault_seats"],[0])
        self.assertEqual(out["extinct_seats"],[])


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache",type=Path,default=ValidationControls.cache)
    args=parser.parse_args()
    ValidationControls.cache=args.cache
    unittest.main(argv=[__file__])

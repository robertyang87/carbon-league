"""Mechanical tests with artificial terminal effects; these are not performance measurements."""
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import policy
import train


def fixture():
    rows=[]
    for b in range(8):
        for j,x in enumerate((.1,.3,.7,.9)):
            rows.append(dict(board="made-up-"+str(b), seed=b, seat=j%2,t=60+j,uid=str(j),
                             batch="synthetic",root_hash=f"{b}-{j}",x=[x,.1,.7,.15,.05],
                             y=3000*(x-.5),filled=True,pay_is_v1=(j in (0,2)),defer_cmd="LEFT",
                             pay_debit=20,defer_debit=0))
    return rows


class TrainingTests(unittest.TestCase):
    def test_exact_eligibility(self):
        rows=fixture()
        self.assertTrue(train.support(rows)["sufficient"])
        rows[0]=dict(rows[0], filled=False)
        gap=train.support(rows)
        self.assertEqual(gap["missing"]["paid_pairs"],1)
        self.assertEqual(gap["failed_pay_intents"],1)
        self.assertFalse(gap["sufficient"])
        with self.assertRaises(ValueError):
            train.leave_one_map_out(rows)

    def test_sign_support_needs_distinct_maps(self):
        rows=fixture()
        for r in rows:
            if r["seed"]:r["y"]=abs(r["y"])
        self.assertEqual(train.support(rows)["missing"]["paid_negative_terrains"],1)
        self.assertEqual(train.support(rows)["counts"]["paid_negative_terrains"],1)

    def test_zero_and_parent_direction(self):
        r=dict(filled=True,y=-30,pay_is_v1=True)
        self.assertEqual(train.signed_gain(r,"defer"),30)
        self.assertEqual(train.signed_gain(r,"pay"),0)
        self.assertEqual(train.signed_gain(dict(r,pay_is_v1=False),"pay"),-30)
        self.assertEqual(train.signed_gain(dict(r,y=0),"defer"),0)

    def test_equal_map_regression(self):
        rows=fixture()
        a,b=train.ridge(rows)
        duplicated=rows[:4]*20+rows
        c,d=train.ridge(duplicated)
        np.testing.assert_allclose(a["beta"],c["beta"],atol=1e-10,rtol=1e-10)
        self.assertAlmostEqual(b["beta"][0],d["beta"][0])
        self.assertGreater(np.linalg.norm(a["beta"][1:]),0)

    def test_holdout_cannot_change_its_own_weights(self):
        rows=fixture()
        first=train.leave_one_map_out(rows)
        hold=first["folds"][0]["board"]
        noisy=[dict(r,y=r["y"]*17) if r["board"]==hold else r for r in rows]
        second=train.leave_one_map_out(noisy)
        f1,f2=first["folds"][0],second["folds"][0]
        self.assertNotIn(hold, f1["fold_weights"]["training_terrains"])
        self.assertEqual(f1["fold_weights"],f2["fold_weights"])
        self.assertEqual([r["choice"] for r in f1["roots"]],[r["choice"] for r in f2["roots"]])
        self.assertNotEqual(f1["gain"],f2["gain"])
        for r in f1["roots"]:
            self.assertEqual(r["issued"],"NONE" if r["choice"]=="pay" else "LEFT")

    def test_failed_pay_intents_are_heldout_outcomes(self):
        rows=fixture()
        base=train.leave_one_map_out(rows)
        root=next(r for r in base["folds"][0]["roots"] if r["changed"])
        row=next(r for r in rows if r["root_hash"]==root["root_hash"])
        bad=dict(row,root_hash="unfilled-extra",y=-row["y"]*10,filled=False,pay_debit=0)
        with_bad=train.leave_one_map_out(rows+[bad])
        old,new=base["folds"][0],with_bad["folds"][0]
        self.assertEqual(old["fold_weights"],new["fold_weights"])
        self.assertLess(new["gain"],old["gain"])
        self.assertGreater(new["paid_gain"],new["unfilled_gain"])
        self.assertEqual(32,train.support(rows+[bad])["counts"]["paid_pairs"])

    def test_off_support_stays_with_v1(self):
        model,_=train.ridge(fixture())
        for v in (-5,math.nan):
            for pay in (True,False):
                self.assertEqual(policy.choose(model,[v,.1,.7,.15,.05],pay),"pay" if pay else "defer")

    def test_stale_receipt_cannot_release(self):
        status={"stage":"insufficient_realized_support","eligible_for_development":False}
        with patch.object(train,"calculate",return_value=(status,[],None,None)):
            with self.assertRaisesRegex(ValueError,"no eligible release"):
                train.rebuild_release()

    def test_overlay_strict_release_requires_matching_artifact(self):
        status={"stage":"ready_for_mechanism_verification","eligible_for_development":True}
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            (d/"protocol.json").write_bytes(b"{}")
            (d/"status.json").write_bytes(train.json_bytes(status))
            (d/"evaluation.json").write_bytes(b'{"forged":true}')
            with patch.object(train,"HERE",d),patch.object(train,"calculate",return_value=(status,[],{},{})):
                with self.assertRaisesRegex(ValueError,"evaluation.json"):
                    train.rebuild_release()


if __name__=="__main__":
    unittest.main()

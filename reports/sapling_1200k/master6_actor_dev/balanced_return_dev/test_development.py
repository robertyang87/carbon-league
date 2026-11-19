"""Test eight-ticket completeness, full public history, cash accounting and safe resumption, without starting any games."""
from contextlib import ExitStack
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import development as dev


class ToyWorld:
    def __init__(self, seed, purpose=None):
        if purpose != "development":
            raise ValueError("wrong split")
        self.seed, self.step, self.terminal = seed, 0, None
        self.terrain_sha256 = str(seed)
        self.commands = []

    def obs(self, seat):
        return {"step":self.step,"player":seat}

    def advance(self, joint):
        self.commands.append(joint)
        self.step += 1
        if self.step == 12:
            self.terminal = official(self.seed,self.commands,"development")


def official(seed, commands, purpose):
    if len(commands) != 12 or purpose != "development":
        raise ValueError("incomplete simulated case")
    cash=[sum(1 for joint in commands if joint[p].get("worker") == "UP") for p in (0,1)]
    score=cash[0]-cash[1]
    return dict(scores=cash,margin_by_seat=[score,-score],u_by_seat=[.5+score/30,.5-score/30],
                result_by_seat=["win" if x>0 else "draw" if x==0 else "loss" for x in (score,-score)],
                extinct_by_seat=[False,False],termination="official_episode_limit",terrain_sha256=str(seed),steps=12)


def accounting_module():
    m=ModuleType("return_accounts")
    def account(w, commands, o, seat):
        if official(w.seed,commands,"development") != o:
            raise ValueError("terminal cash differs")
        return dict(score=o["scores"][seat],enemy_score=o["scores"][1-seat],official_margin=o["margin_by_seat"][seat])
    m.account=account
    m.contrast=lambda p,c:dict(official_margin=c["official_margin"]-p["official_margin"])
    return m


class ToyPack:
    all_instances = 0

    def __init__(self, path,student=True):
        self.label=path
        self.student=student
        self.step=0
        ToyPack.all_instances += 1

    def observe(self,obs,issue=True):
        if obs["step"] != self.step:
            raise ValueError("missing public history")
        self.step += 1
        if not issue:
            return None
        return {"worker":"UP" if self.label == "challenger" and self.student else "LEFT"}


class DevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.ctx=ExitStack()
        self.addCleanup(self.ctx.close)
        self.tmp=Path(self.ctx.enter_context(TemporaryDirectory()))
        self.ctx.enter_context(patch.object(dev,"OUT",self.tmp))
        self.ctx.enter_context(patch.object(dev,"PublicPack",ToyPack))
        self.ctx.enter_context(patch.object(dev.common,"assert_cpu",lambda:None))
        self.ctx.enter_context(patch.object(dev.worlds,"World",ToyWorld))
        self.ctx.enter_context(patch.object(dev.worlds,"replay_official",official))
        self.module=accounting_module()
        self.ctx.enter_context(patch.dict(sys.modules,{"return_accounts":self.module}))
        self.paths=dict(candidate="challenger",parent="incumbent",opponent="rival")
        self.identity={"screen":"toy"}
        self.ctx.enter_context(patch.object(dev,"configuration",lambda binding:({"development":[22,33]},self.paths,self.identity)))

    def test_eight_full_games_and_shadows_are_not_games(self):
        report=dev.run()
        self.assertEqual([(r["seed"],r["seat"]) for r in report["pairs"]],[(22,0),(22,1),(33,0),(33,1)])
        self.assertEqual((report["by_arm"]["parent"]["draws"],report["by_arm"]["candidate"]["wins"]),(4,4))
        self.assertEqual((report["formal_games"],report["original_master3s2_games"]),(0,0))
        for attr in report["attribution_by_arm"].values():
            self.assertEqual((attr["all_roots"],attr["parent_candidate_different"],attr["student_ablation_roots"],
                              attr["candidate_differs_from_no_student"]),(48,48,8,8))
        self.assertTrue(report["retain_for_next_study"])
        count=ToyPack.all_instances
        self.assertEqual(dev.run(),report)
        self.assertEqual(ToyPack.all_instances,count)

    def test_checkpoint_continues_and_bad_prior_ticket_is_rejected(self):
        scratch=self.tmp/"progress_candidate_22_1.json.gz"
        real=dev.checkpoint
        def stopped(path,data):
            real(path,data)
            if len(data["actions"])==3:
                raise KeyboardInterrupt
        with patch.object(dev,"checkpoint",stopped),self.assertRaises(KeyboardInterrupt):
            dev.play(22,1,"candidate",self.paths,self.identity,self.module)
        preserved=dev.read(scratch)
        with self.assertRaisesRegex(ValueError,"another policy"):
            dev.play(22,1,"candidate",self.paths,{"wrong":"ticket"},self.module)
        result=dev.play(22,1,"candidate",self.paths,self.identity,self.module)
        self.assertEqual(result["actions"][:3],preserved["actions"])
        self.assertFalse(scratch.exists())
        self.assertEqual(result["official"]["result_by_seat"][1],"win")

    def test_tampered_case_is_not_accepted_on_resume(self):
        leaf=dev.play(33,0,"parent",self.paths,self.identity,self.module)
        leaf["shadow"][1]["student_ablation_roots"] = 1
        leaf["shadow"][1]["candidate_without_sapling"] = {}
        with self.assertRaisesRegex(ValueError,"unregistered student ablation"):
            dev.validate_case(leaf,leaf["ticket"],self.module)
        del leaf["shadow"][1]["candidate_without_sapling"]
        leaf["actions"][0][0] = {"worker":"UP"}
        with self.assertRaisesRegex(ValueError,"different actions"):
            dev.validate_case(leaf,leaf["ticket"],self.module)


if __name__=="__main__":
    unittest.main()

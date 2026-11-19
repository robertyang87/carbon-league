"""Resource handoff regression tests do not generate new games or touch the actual reservation."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import resource_slot as slot


class SlotTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.here=Path(self.tmp.name)
        self.res=self.here/"resources"
        self.dev=self.here/"native"
        self.before=self.here/"before"
        self.after=self.here/"after"
        for path in (self.res,self.dev,self.before,self.after):path.mkdir()
        for name,value in {"ROOT":self.here,"HERE":self.here,"SLOT":self.res,"DEV":self.dev,
                           "BEFORE":self.before,"AFTER":self.after}.items():
            setter=patch.object(slot,name,value);setter.start();self.addCleanup(setter.stop)
        r={"owner":"paired_hybrid_parent","pause_before":"after","training_predecessor":"before","arms":["parent","candidate"],"cpus":[0,1],
           "physical_seats":[0,1],"examiner_md5":"registered-examiner"}
        self.write(self.res/"reservation.json",r)
        setter=patch.object(slot,"reservation",return_value=r);setter.start();self.addCleanup(setter.stop)
        self.write(self.dev/"failure.json",{"result":"did_not_pass"})

    def write(self,path,obj):
        path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(obj))

    def failure(self,owner=None):
        values={"decision":"do_not_run_native","started_native_games":0,"evidence_files_sha256":{"failure.json":slot.sha(self.dev/"failure.json")}}
        if owner is not None:values.update(decision="owner_withdrawn",owner=owner)
        self.write(self.dev/"gate.json",values)
        return self.dev/"gate.json"

    def test_cancel_before_play_is_evidence_bound_and_idempotent(self):
        gate=self.failure("paired_hybrid_parent");slot.cancel(gate,"translation parity did not pass")
        slot.cancel(gate,"translation parity did not pass")
        slot.wait_before_source(self.after,{"intake_name":"after"})
        self.write(self.dev/"failure.json",{})
        with self.assertRaisesRegex(ValueError,"evidence file changed"):
            slot.wait_before_source(self.after,{"intake_name":"after"})

    def test_cancellation_rejects_source_owner_or_outcomes_and_fake_receipts(self):
        with self.assertRaisesRegex(ValueError,"unknown withdrawing"):slot.cancel(self.failure("outsider"),"cancel request")
        gate=self.failure()
        with slot.lock(".native-owner.lock"),self.assertRaises(BlockingIOError):slot.cancel(gate,"cancel request")
        self.write(self.dev/"parent/results.json",{"games":[],"status":"running"})
        with self.assertRaisesRegex(ValueError,"development began"):slot.cancel(gate,"cancel request")
        self.write(self.res/"cancelled.json",{})
        with self.assertRaises(ValueError):slot.wait_before_source(self.after,{"intake_name":"after"})
        self.assertFalse((self.res/"release.json").exists())

    def test_fabricated_negative_gate_without_actual_trajectories_is_rejected(self):
        with self.assertRaises((FileNotFoundError,ValueError)):
            slot.cancel(self.failure(),"unsubstantiated failure")
        self.assertFalse((self.res/"cancelled.json").exists())

    def test_both_cohorts_and_all_seat_hashes_required_for_real_threshold(self):
        std={"dev_protocol_sha256":"protocol","candidate_pack_sha256":"candidate"}
        files={};g={"transfer_by_batch":{}}
        for b,seeds in (("03",[11,12]),("04",[13,14])):
            td=self.here/"batches"/("intake_"+b)
            for name,val in (("protocol.json",{"train_maps":seeds}),("source_receipt.json",{}),("opportunities.jsonl.gz",{})):
                self.write(td/name,val)
            witness={}
            for seed in seeds:
                for seat in (0,1):
                    name=f"transfer{b}/{seed}_{seat}.json.gz"
                    self.write(self.dev/name,[seed,seat])
                    witness[name]=slot.sha(self.dev/name)
                    files[str((self.dev/name).relative_to(self.here))]=witness[name]
            by_map={str(seed):{"saved":2,"harmed":int(b=="04"),"net_saved":2-int(b=="04")} for seed in seeds}
            passed=b=="03"
            g["transfer_by_batch"][b]={"maps":by_map,"pass":passed}
            report={"batch":b,"protocol_sha256":"protocol","candidate_sha256":"candidate","per_seat_files":witness,"maps":by_map,
                    "identity":{"pack_sha256":"candidate","opportunity_sha256":slot.sha(td/"opportunities.jsonl.gz"),"source_receipt_sha256":slot.sha(td/"source_receipt.json")},
                    "passed":passed,"parity_passed":True,"all_actions_matched":True,"all_opportunity_predictions_matched":True,"all_original_v1_scores_matched":True}
            path=self.dev/f"transfer{b}.json";self.write(path,report);files[str(path.relative_to(self.here))]=slot.sha(path)
        slot.training_transfer(g,files,std,False)
        with self.assertRaisesRegex(ValueError,"observed prerequisite"):slot.training_transfer(g,files,std,True)
        files.pop("native/transfer04/14_1.json.gz")
        with self.assertRaisesRegex(ValueError,"unbound seat transfer"):slot.training_transfer(g,files,std,False)

    def test_grant_requires_boundary_and_no_opened_training(self):
        import continue_intakes
        with patch("batch_cli.other_native",return_value=[]),patch.object(continue_intakes,"active_lock",return_value=False),patch.object(continue_intakes,"verify_sealed"),patch.object(slot,"approval",return_value={"seeds":[1,2]}):
            self.write(self.dev/"slot_request.json",{})
            self.assertFalse(slot.grant())
            self.write(self.before/"training_labels.json",{})
            self.write(self.before/"aggregate.json",{"source_intakes":[{"dir":"before","labels_sha256":slot.sha(self.before/"training_labels.json")}]})
            self.write(self.after/"source/results.json",{"status":"running","games":[]})
            with self.assertRaisesRegex(ValueError,"already started"):slot.grant()
            (self.after/"source/results.json").unlink()
            self.assertTrue(slot.grant());self.assertTrue(slot.grant())
            self.assertEqual(slot.read(self.res/"granted.json")["predecessor_aggregate_sha256"],slot.sha(self.before/"aggregate.json"))
            with patch.object(continue_intakes,"active_lock",return_value=True):self.assertFalse(slot.grant())

    def test_missing_arms_and_forged_release_do_not_allow_training(self):
        self.write(self.res/"granted.json",{"seeds":[1,2],"boards":{"1":"a","2":"b"},"packs":{"parent":{},"candidate":{}}})
        self.write(self.res/"release.json",{})
        with patch.object(slot,"grant",return_value=True),self.assertRaises((FileNotFoundError, ValueError)):
            slot.wait_before_source(self.after,{"intake_name":"after"})
        with patch.object(slot,"grant",side_effect=AssertionError):slot.wait_before_source(self.here/"other",{})

    def test_release_requires_eight_seats_two_maps_and_truthful_official_scores(self):
        from source_games import gate
        import official_game_replay as official
        keys=[(s,p) for s in [1,2] for p in [0,1]]
        sig={"seeds":[1,2],"boards":{"1":"a","2":"b"},"packs":{arm:{"pack_sha256":arm,"weight_sha256":"w"} for arm in ("parent","candidate")}}
        self.write(self.res/"granted.json",sig)
        self.write(self.here/"source/results.json",{"header":{"configuration":{"size":15}}})
        self.write(self.here/"source/official_replay.json",{"engine_source_sha256":{"engine.py":"123"}})
        for arm in ("parent","candidate"):
            d=self.dev/arm;d.mkdir();r=[];o=[]
            for s,p in keys:
                name=f"raw_{s}_{p}.json.gz";raw=gzip.compress(json.dumps({"configuration":{"TEMP":1},"audit_messages":[]}).encode());(d/name).write_bytes(raw)
                digest=slot.sha(d/name)
                row={"raw_file":name,"raw_sha256":digest,"actual_transitions":2,"official_score_by_seat":[2,3],"result":("win" if p else "loss"),"seed":s,"candidate_seat":p};r.append(row)
                o.append({"map_sha256":sig["boards"][str(s)],"transcript_sha256":digest,"steps":2,"official_reward_by_seat":[2,3],"result":row["result"],"statuses":["DONE","DONE"],"seed":s,"seat":p})
            self.write(d/"results.json",{"header":{"cpus":[0,1],"plan_sha256":None,"configuration":{"size":15},"source_sha256":{"arena.py":"a"*64},"seeds":[1,2],"examiner_md5":"registered-examiner",**sig["packs"][arm]},"games":r,"status":"complete"})
            self.write(d/"official_replay.json",{"engine_source_sha256":{"engine.py":"123"},"runs":{arm:{"games":o,"pack_sha256":arm}}})
        self.write(self.dev/"gate.json",{})
        self.write(self.dev/"release_manifest.json",{})
        pairs=[]
        for i,(s,p) in enumerate(keys):
            pp={"seed":s,"seat":p}
            for a in ("parent","candidate"):
                pp[a]=slot.read(self.dev/a/"official_replay.json")["runs"][a]["games"][i]
            pairs.append(pp)
        self.write(self.dev/"comparison.json",{"games":8,"games_per_arm":4,"pairs":pairs,"grant_sha256":slot.sha(self.res/"granted.json"),
                   "gate_sha256":slot.sha(self.dev/"gate.json"),"release_manifest_sha256":slot.sha(self.dev/"release_manifest.json")})
        count=[]
        def replay(d,r):
            count.append((d.name,r["seed"],r["candidate_seat"]))
            return next(x for x in slot.read(d/"official_replay.json")["runs"][d.name]["games"] if (x["seed"],x["seat"])==(r["seed"],r["candidate_seat"]))
        def check_config(c):
            self.assertEqual(c,{"TEMP":1})
        with (patch.object(slot,"grant",return_value=True),patch.object(gate,"validate_game"),patch.object(gate,"verify_raw"),
              patch.object(gate,"validate_examiner_configuration",side_effect=check_config),patch.object(official,"check",side_effect=replay)):
            result=slot.release();self.assertEqual(result["games"],8);self.assertEqual(len(set(count)),8)
            slot.wait_before_source(self.after,{"intake_name":"after"})
            path=self.dev/"candidate/official_replay.json";x=slot.read(path);x["runs"]["candidate"]["games"][1]["official_reward_by_seat"]=[333,444];self.write(path,x)
            with self.assertRaisesRegex(ValueError,"unbound official replay"):slot.wait_before_source(self.after,{"intake_name":"after"})


if __name__=="__main__":
    unittest.main()

import unittest

import numpy as np
import audit


class AuditTests(unittest.TestCase):
    def test_support_uses_actual_worker_and_keeps_empty_strata_empty(self):
        table=audit.support([{"t":205,"choice":{"context":[40.,3.,0],"desired_loaded":False}},
                             {"t":205,"choice":{"context":[71.,5.,0],"desired_loaded":False}},
                             {"t":271,"choice":None}])
        self.assertEqual(table,{(205,1,0):(40.,71.,3.,5.,2)})
        x={"tok_mask":np.array([1,1,1,1,1,1,1,0,0,0],bool),"tok_kind":np.array([0,0,0,0,0,0,1,0,0,0]),"tok_pos":np.arange(10),
           "tok_cargo":np.array([40.,71.,39.,72.,40.,40.,40.,50.,0.,0.])}
        aux={"dob":np.array([3,5,4,4,6,4,4,4,4,4]),"en_min":np.array([40,71,99,99,99,39,99,99,99,99])}
        self.assertEqual(audit.eligible(x,aux,205,table),[0,1])
        self.assertEqual(audit.eligible(x,aux,204,table),[])
        self.assertEqual(audit.eligible(x,aux,271,table),[])

    def test_entropy_masks_illegal_and_resolves_ties_on_low_slot(self):
        mask=np.array([True,False,True,False,False])
        tr={i:{"legal":mask.copy(),"combined":np.array([0.,99.,0.,30.,3.],np.float32)} for i in (0,1,2)}
        tr[2]["combined"][0]=20.
        self.assertEqual(audit.highest_entropy([2,1,0],tr),0)
        tr[0]["legal"][2]=False
        self.assertEqual(audit.highest_entropy([0,2,1],tr),1)
        self.assertEqual(audit.highest_entropy([0],tr),-1)

    def test_positive_phase_control_uses_genuine_tensors_on_a_public_neighbor(self):
        parent=audit.common.policy.load(audit.common.PARENT)
        candidate=audit.common.policy.load(audit.HERE/"candidate.py")
        world,features,_,_=audit.common.open_state(621471121,1,206)
        feat=features[1];x=feat.observe(world.obs(1))
        def changed(at_phase):
            z=dict(x,left=299-at_phase)
            old=feat.to_actions(parent._mixed_actions(z,feat.aux));new=feat.to_actions(candidate._mixed_actions(z,feat.aux))
            return {k:(old.get(k),new.get(k)) for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
        self.assertEqual(changed(206),{})
        # This single-scalar decoder probe is counterfactual; the observed game action at 206 is unchanged.
        self.assertEqual(changed(205),{"player-1-worker-12":("RIGHT","LEFT")})

    def test_rejects_stale_prefix_even_if_argmax_unchanged(self):
        steps={0:{"previous":8,"action":1,"legal":np.array([True]*5),"combined":np.arange(5,dtype=np.float32),
                  "unit":np.array([1.,2.]),"move":np.ones((5,3))}}
        audit.same_trace(steps,steps)
        poisoned={0:{**steps[0],"move":steps[0]["move"].copy()}}
        poisoned[0]["move"][2,1]=15
        with self.assertRaisesRegex(ValueError,"different move"):audit.same_trace(steps,poisoned)
        quantized={0:{**steps[0],"combined":np.nextafter(steps[0]["combined"],np.inf)}}
        with self.assertRaisesRegex(ValueError,"different combined"):audit.same_trace(steps,quantized)
        with self.assertRaisesRegex(ValueError,"active physical slots"):audit.same_trace(steps,{})


if __name__=="__main__":unittest.main()

import unittest
from run import ARMS, compare

class TestThreeArms(unittest.TestCase):
    maps=(1,2)
    def group(self, parent,control,candidate):
        terms={arm:{} for arm in ARMS}
        for i,(seed,seat) in enumerate(( (s,a) for s in self.maps for a in (0,1) )):
            for arm,values in zip(ARMS,(parent,control,candidate)):
                v=values[i]
                terms[arm][seed,seat]={"margin":v,"U":0.5+v*.001,"outcome":"win" if v>0 else "loss"}
        return compare(terms,{"maps":self.maps})

    def test_each_of_two_comparators_required(self):
        r=self.group([-2]*4,[3]*4,[2]*4)
        self.assertEqual(r["verdict"],"retain_parent")
        self.assertTrue(all(r["candidate_vs"]["parent"]["criteria"].values()))
        self.assertFalse(r["candidate_vs"]["control"]["criteria"]["both_maps_positive"])

    def test_root_count_and_map_direction_independent(self):
        r=self.group([-3]*4,[-3]*4,[-4,1,-4,1])
        self.assertFalse(r["candidate_vs"]["control"]["criteria"]["three_of_four_U"])
        self.assertTrue(r["candidate_vs"]["control"]["criteria"]["both_maps_positive"])
        self.assertEqual(r["verdict"],"retain_parent")

    def test_positive_controls(self):
        r=self.group([-2]*4,[-1]*4,[1]*4)
        self.assertEqual(r["verdict"],"exploratory_followup_signal")
        self.assertEqual(r["rollup"]["candidate"]["win"],4)
        self.assertFalse(r["formal"])

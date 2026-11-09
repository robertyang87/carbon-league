"""Dataset m2fam: the RyanX M2-family bots named in M2FAM (comma list, default M2,M3,M4) as teacher slot 2;
with CNET2_STEP_MAX=100 opening rows only (opening stand-in for the exam). M4-only build + concat doubles M4.
+ our own ladder losses (v14p/v15/v16) with the WINNING opponent as teacher, only when that opponent maps to a slot.
slots: 0 Master, 1 v9, 2 v8-preview, 3 Godslayer-P2C, 4 RyanX-V11, 5 Master2, 6 v10, 7 v11-preview. WB/hy4 never teachers.
  CNET2_WEIGHTED=1 CNET2_UW=1 python3 build_v8.py OUT DIR [DIR...]"""
import sys
import cnet2_feat as F

import os
FAMILY = {"AgyCarbon-P2C-Godslayer-" + v: "AgyCarbon-P2C-Godslayer" for v in ("XC", "XCI", "XCII", "XCIII", "XCIV")}
FAMILY.update({"RyanX-AIPowercore-" + v: "RyanX-AIPowercore-M2" for v in os.environ.get("M2FAM", "M2,M3,M4").split(",")})
# 无始 is a HELD-OUT exam family (stand-in built from its replays): never a teacher. Its games still teach
# through the winners of those games (Master2s excluded: it is an exam too).


class _Teachers(list):
    def _m(self, n):
        return FAMILY.get(n, n)

    def __contains__(self, n):
        return list.__contains__(self, self._m(n))

    def __getitem__(self, i):
        r = list.__getitem__(self, i)
        return _Teachers(r) if isinstance(i, slice) else r

    def index(self, n, *a):
        return list.index(self, self._m(n), *a)


T = _Teachers(["_slot0", "_slot1", "RyanX-AIPowercore-M2", "_slot3",          # only M2's own sides become rows;
               "_slot4", "_slot5", "_slot6", "_slot7", "other"])           # the other slots keep their positions
F.TEACHERS = T
import cnet2_build as B  # noqa: E402
B.F.TEACHERS = T
B.OURS_MIN_OPP = 1e9          # our-loss games only teach when the winner has a real slot (never 'other')
if __name__ == "__main__":
    B.main(sys.argv[1], sys.argv[2:])

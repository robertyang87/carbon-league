"""Dataset v8 = v7 sources + the 无始 family (v6.0/v7.0 only, ~2610) in (exam only)
+ our own ladder losses (v14p/v15/v16) with the WINNING opponent as teacher, only when that opponent maps to a slot.
slots: 0 Master, 1 v9, 2 v8-preview, 3 Godslayer-P2C, 4 RyanX-V11, 5 Master2, 6 v10, 7 v11-preview. WB/hy4 never teachers.
  CNET2_WEIGHTED=1 CNET2_UW=1 python3 build_v8.py OUT DIR [DIR...]"""
import sys
import cnet2_feat as F

FAMILY = {"AgyCarbon-P2C-Godslayer-" + v: "AgyCarbon-P2C-Godslayer" for v in ("XC", "XCI", "XCII", "XCIII", "XCIV")}
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


T = _Teachers(["_slot0", "_slot1", "_slot2", "_slot3",
               "_slot4", "claude-AlphaCarbon-Master2", "_slot6", "_slot7", "other"])
F.TEACHERS = T
import cnet2_build as B  # noqa: E402
B.F.TEACHERS = T
B.OURS_MIN_OPP = 1e9          # our-loss games only teach when the winner has a real slot (never 'other')
if __name__ == "__main__":
    B.main(sys.argv[1], sys.argv[2:])

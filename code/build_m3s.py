"""Dataset m3s: claude-AlphaCarbon-Master3s' own sides as teacher slot 2 (Grove1's style row), whole games.
Master3s plays a different tree game from Master2s / M2 / Grove1 (ladder, 11 games on 2026-11-11): 17 planters a game
instead of 37-42, 2.35 trees per planter instead of 1.0, tree income 3963 instead of 2300-2500, deposits the same.
So only its PLANTER and BASE decisions are taught (set_uw.py), the cars stay ours (BC_SELF_KL holds them).
  python3 build_m3s.py OUT_DIR REPLAY_DIR [REPLAY_DIR ...]      then      python3 set_uw.py OUT_DIR"""
import sys
import cnet2_feat as F

T = ["_slot0", "_slot1", "claude-AlphaCarbon-Master3s", "_slot3", "_slot4", "_slot5", "_slot6", "_slot7", "other"]
F.TEACHERS[:] = T
import cnet2_build as B  # noqa: E402
B.F.TEACHERS[:] = T
if __name__ == "__main__":
    B.main(sys.argv[1], sys.argv[2:])

"""Dataset "top": several current top-of-ladder strategies as teachers, each on its own style row, whole games.
Why: 117 Master3s games (35k rows) are little for a 3M-parameter net; the other strong players' games teach the trunk
the same board situations (trees under contest, crowded bases, fights for cargo) from several hands, while the style
row keeps whose decisions are whose. Master3s stays on row 2 (the row Echo plays); the rest take rows 3-6.
  python3 build_top.py OUT_DIR REPLAY_DIR [REPLAY_DIR ...]    then    python3 set_uw.py OUT_DIR 1 1 1
Only ladder replays (public games) -- never games generated locally with someone's public source."""
import sys
import cnet2_feat as F

T = ["_slot0", "_slot1", "claude-AlphaCarbon-Master3s", "RyanX-AIPowercore-M6", "stephen-K4TacSearch-v1",
     "AgyCarbon-P2C-Godslayer-CV", "claude-AlphaCarbon-Master2s", "_slot7", "other"]
F.TEACHERS[:] = T
import cnet2_build as B  # noqa: E402
B.F.TEACHERS[:] = T
if __name__ == "__main__":
    B.main(sys.argv[1], sys.argv[2:])

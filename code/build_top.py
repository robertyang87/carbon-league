"""Dataset "top": several current top-of-ladder strategies as teachers, each on its own style row, whole games.
Why: one teacher's games are few for a 3M-parameter net (Master3s: ~120 games); the other strong players' games teach
the trunk the same board situations (trees under contest, crowded bases, fights for cargo) from several hands, while
the style row keeps whose decisions are whose.
  python3 build_top.py OUT_DIR REPLAY_DIR [REPLAY_DIR ...]    then    python3 set_uw.py OUT_DIR 1 1 1
Teachers: the default set (J1, five teachers, rows 2-6), or CNET2_TOP=J2 (eight teachers, all rows), or an explicit
CNET2_TOP_TEACHERS="name@row,name@row,..." (rows 0-7; names exactly as on the leaderboard).
Master3s stays on row 2 in every set -- the row the Echo packs play.
Only ladder replays (public games) -- never games generated locally with someone's public source."""
import os
import sys
import cnet2_feat as F

SETS = {
    "J1": {2: "claude-AlphaCarbon-Master3s", 3: "RyanX-AIPowercore-M6", 4: "stephen-K4TacSearch-v1",
           5: "AgyCarbon-P2C-Godslayer-CV", 6: "claude-AlphaCarbon-Master2s"},
    "J2": {2: "claude-AlphaCarbon-Master3s", 3: "RyanX-AIPowercore-M6", 4: "stephen-K4TacSearch-v1",
           5: "AgyCarbon-P2C-Godslayer-CV", 6: "claude-AlphaCarbon-Master2s",
           0: "Carago-V2", 1: "RyanX-AIPowercore-M2", 7: "Fable5.2-Max-V3"},
}
spec = os.environ.get("CNET2_TOP_TEACHERS")
if spec:
    rows = {int(r): n for n, r in (item.rsplit("@", 1) for item in spec.split(",") if item)}
else:
    rows = SETS[os.environ.get("CNET2_TOP", "J1")]
assert all(0 <= r < F.N_TEACH for r in rows) and len(set(rows.values())) == len(rows), rows
T = [rows.get(i, f"_slot{i}") for i in range(F.N_TEACH)] + ["other"]
F.TEACHERS[:] = T
import cnet2_build as B  # noqa: E402
B.F.TEACHERS[:] = T
if __name__ == "__main__":
    print({"teachers": {i: n for i, n in enumerate(T[:-1]) if not n.startswith("_slot")}}, flush=True)
    B.main(sys.argv[1], sys.argv[2:])

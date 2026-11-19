"""Two named controls with the same rival and seat for Master5s and Master6, on distinct maps."""
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fetch
from defense import DEST, MOVE


def main():
    names = {"claude-AlphaCarbon-Master5s", "claude-AlphaCarbon-Master6"}
    manifest = fetch.load(HERE / "manifest.json")
    cash = {g["id"]:g["turns"] for g in json.loads(gzip.decompress((HERE/"turn_cash.json.gz").read_bytes()))["games"]}
    cases = {g["id"]:g for g in json.loads(gzip.decompress((HERE/"cases.json.gz").read_bytes()))["rows"]}
    groups = defaultdict(list)
    for game in manifest["games"]:
        if len(names.intersection(game["seat_names"])) == 1:
            seat = next(s for s in (0, 1) if game["seat_names"][s] in names)
            groups[game["seat_names"][1-seat], seat].append(game)
    records = []
    for (rival, seat), games in sorted(groups.items()):
        if set(g["seat_names"][seat] for g in games) != names:
            continue
        for g in games:
            gid = g["id"]
            raw = (fetch.CACHE / g["path"]).read_bytes()
            if fetch.sha(raw) != g["sha256"]:
                raise ValueError(gid)
            game = json.loads(gzip.decompress(raw))
            frames, config = game["steps"], game["configuration"]
            side = 1-seat
            rival_units = [frame[0]["observation"]["players"][side][2] for frame in frames]
            has_collector = [any(w[2] == "COLLECTOR" for w in roster.values()) for roster in rival_units]
            last_loss = None
            if any(has_collector) and not has_collector[-1]:
                end = max(t for t, has in enumerate(has_collector) if has) + 1
                pre = frames[end-1][0]["observation"]["players"]
                commands = [f["action"] or {} for f in frames[end]]
                actions = []
                for uid, (pos,load,role) in rival_units[end-1].items():
                    if role != "COLLECTOR":
                        continue
                    cell = DEST[pos][MOVE[commands[side].get(uid)]]
                    for s, player in enumerate(pre):
                        for wid,w in player[2].items():
                            move = commands[s].get(wid)
                            if DEST[w[0]][MOVE[move]] == cell:
                                actions.append({"seat":s,"id":wid,"role":w[2],"before":w[0],"cargo":w[1],
                                                "action":move or "NONE","landing":cell,
                                                "survived":wid in frames[end][0]["observation"]["players"][s][2]})
                last_loss={"first_terminal_collectorless_t":end,"actual_contact":actions,"cash_after":
                           frames[end][0]["observation"]["players"][side][0],
                           "recruit_price_after":config["recCollectorCost"] + config["recCollectorIncreaseCost"] *
                               sum(w[2]=="COLLECTOR" for z in frames[end][0]["observation"]["players"] for w in z[2].values())}
            flows=[]
            for cutoff in (30,60,100,len(cash[gid])):
                delta=cash[gid][:cutoff]
                flows.append({"through_t":cutoff,"own_rival_bank":[round(sum(x[2][s][0] for x in delta),2) for s in (seat, side)],
                              "own_rival_tree":[round(sum(x[2][s][2] for x in delta),2) for s in (seat, side)]})
            defend = cases[gid]["defense"]["seat_totals"][seat]
            receipts = [row[0] for row in cash[gid] if row[2][side][0]>0]
            records.append({"id":gid,"seed":g["seed"],"opponent":rival,"our_seat":seat,"strategy":g["seat_names"][seat],
                            "own_rival_official":[g["score_by_seat"][s] for s in (seat,side)],
                            "rival_first_last_own_delivery":[receipts[0],receipts[-1]] if receipts else None,
                            "defense_chose_safe":defend.get("chose_geometrically_safe_move",0),
                            "defense_safe_available":defend.get("loaded_stay_under_threat_with_safe_move",0),
                            "enemy_cargo_acquired":defend.get("enemy_cargo_acquired",0),"cumulative_flows":flows,
                            "opponent_final_collector_loss":last_loss})
    assert len(records) == 4 and len({r["opponent"] for r in records}) == 2
    fetch.dump(HERE/"matched.json", {"inclusion":"every common named opponent in the 19-map cutoff for which Master5s and Master6 appear in the same physical seat, no conditioning on score","games":records})


if __name__ == "__main__":
    main()

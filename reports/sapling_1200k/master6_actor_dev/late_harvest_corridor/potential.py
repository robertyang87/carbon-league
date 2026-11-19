"""Public geometric opportunity ceilings for directly mined cargo; excludes all collision and tree revenues."""
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean, median

from data import public_roots
from policy import Context, adjacent, choose, dest, home_distance


HERE=Path(__file__).resolve().parent


def limits(ctx):
    cap=ctx.cfg["maxCellCarbon"]
    rates=[]
    for uid,(pos,load,kind) in ctx.me[2].items():
        if kind!="COLLECTOR" or load<=0 or pos in ctx.homes: continue
        parent=ctx.own.get(uid) or "NONE"
        if home_distance(dest(pos,parent,ctx.n),ctx.homes,ctx.n)>=home_distance(pos,ctx.homes,ctx.n): continue
        for cell in (pos,*adjacent(pos,ctx.n).values()):
            if cell not in ctx.cells or cell in ctx.homes or cell in ctx.enemyhomes: continue
            approach=int(cell!=pos)
            ticks=min(3,ctx.h-1-approach-home_distance(cell,ctx.homes,ctx.n))
            if ticks<1: continue
            estimate,staleness=ctx.cells[cell]
            if estimate<=0: continue
            # To bound rather than forecast, assign the full cap at the first mining turn
            # to stale or adjacent cells. Current collector's own cell is visible now.
            available=estimate if cell==pos and staleness==0 else cap
            rate=ctx.cfg["initialCollectRate"]
            rates.append({"uid":uid,"cell":cell,"mining_ticks":ticks,"unconditional_cap":cap*rate*ticks,
                          "stationary_open_patch":available*(1-(1-rate)**ticks),"source_age":staleness})
    if not rates: return {"stationary_open_patch":0.,"unconditional_cap":0.}
    return {"stationary_open_patch":max(x["stationary_open_patch"] for x in rates),
            "unconditional_cap":max(x["unconditional_cap"] for x in rates)}


def main():
    evidence=[]
    for split in ("dev","eval"):
        for (seed, seat, t),obs,past,base,cfg in public_roots(split):
            if not 5<=cfg["episodeSteps"]-1-t<=12:continue
            ctx=Context(obs,past,base,cfg)
            ans,_,meta=choose(obs,past,base,cfg)
            row={"split":split,"seed":seed,"seat":seat,"t":t,"geometric":limits(ctx),"change":ans!=base}
            if row["change"]:
                own=meta["target_worker"]; cand=meta["route"]
                here=ctx.me[2][own][0]
                row.update(uid=own,initial_cargo=ctx.me[2][own][1],current_cell=here,
                           patch=cand["target"],physical_base=cand["home"],mining_ticks=cand["holds"],
                           projected_mining=cand["harvest"],projected_bank=cand["delivery"],
                           enemy_tree_tax=cand["enemy_tax"],R_margin_score=meta["predicted_gain"],
                           first_command=meta["command"],parent_command=meta["parent"])
            evidence.append(row)
    totals={}
    for group in ("dev","eval"):
        rows=[r for r in evidence if r["split"]==group]; issued=[r for r in rows if r["change"]]
        games=sorted({(r["seed"],r["seat"]) for r in rows})
        def items(k):return [r for r in rows if (r["seed"],r["seat"])==k]
        def first(k):return next((r for r in items(k) if r["change"]),None)
        live=[r for k in games if (r:=first(k))]
        mean_geom=mean(max(r["geometric"]["stationary_open_patch"] for r in items(k)) for k in games)
        totals[group]={"games":len(games),"eligible_times":len(rows),"static_edits":len(issued),
                       "games_with_first_edit":len(live),"games_with_approach_detour":sum(r["current_cell"]!=r["patch"] for r in live),
                       "average_best_geometric_nearby_patch_per_game":mean_geom,
                       "average_predicted_gross_mining_at_actual_first_over_all_games":sum(r["projected_mining"] for r in live)/len(games),
                       "max_possible_per_mission_without_new_terrain_conditions":max(r["geometric"]["unconditional_cap"] for r in rows),
                       "first_edits":live}
    record={"policy_sha256":hashlib.sha256((HERE/"policy.py").read_bytes()).hexdigest(),"rows":evidence,"group":totals}
    (HERE/"potential.json.gz").write_bytes(gzip.compress(json.dumps(record,separators=(",",":"),allow_nan=False).encode(),mtime=0))
    for split,group in totals.items(): print(split,{k:v for k,v in group.items() if k!="first_edits"})

if __name__=="__main__":main()

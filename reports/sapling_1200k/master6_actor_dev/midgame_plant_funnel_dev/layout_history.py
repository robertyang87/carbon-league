"""Track which public positions precede each successful safe fertile planting opportunity."""
from collections import Counter, defaultdict
from functools import partial
from statistics import mean
from funnel import HERE, ROOT, load, save, FastCarbon


def distance(a, b, size):
    dx, dy = abs(a%size-b%size), abs(a//size-b//size)
    return min(dx,size-dx) + min(dy,size-dy)


def snapshots(sim):
    board=[]
    for seat in (0,1):
        obs=sim.obs(seat)
        mine,them=obs["players"][seat],obs["players"][1-seat]
        dist=partial(distance,size=sim.size)
        own_trees={p for p,_ in mine[3].values()}
        enemy_trees={p for p,_ in them[3].values()}
        team={}
        for uid,(p,_,role) in mine[2].items():
            if role!="PLANTER":
                continue
            home=min(dist(p,b) for b in mine[1].values())
            away=min(dist(p,b) for b in them[1].values())
            enemy=min((dist(p,q) for q,_,_ in them[2].values()), default=99)
            team[uid]=(p,home,away,enemy,int(p in own_trees),int(p in enemy_trees))
        board.append(team)
    return board


def process(game):
    raw=load(ROOT/game["path"])
    sim=FastCarbon(raw["game_configuration"]).reset(seed=game["seed"])
    history=[]
    for t,joint in enumerate(raw["actions"][:200]):
        history.append(snapshots(sim))
        sim.step(*joint)
    out=defaultdict(Counter)
    for row in game["roots"]:
        t,s,uid=row["t"],row["s"],row["uid"]
        current=history[t][s][uid]
        key=(s^game["candidate_seat"],)
        classes=["all"]
        if current[0] in row["public"]:
            classes.append("reached_safe_rich")
            if row["settled"]=="plant":
                classes.append("planted_safe_rich")
        for label in classes:
            for delay in (0,3,5,10):
                count=out[key+(label,delay)]
                count["opportunities"]+=1
                before=history[t-delay][s].get(uid)
                if before is None:
                    continue
                count["same_worker"]+=1
                p,home,away,enemy,owned_tree,enemy_tree=before
                count["enemy_half"]+=away<home
                count["enemy_one_step"]+=enemy<=1
                count["own_tree"]+=owned_tree
                count["enemy_tree"]+=enemy_tree
                count["distance_moved"]+=distance(p,current[0],sim.size)
    return [{"side":key[0],"class":key[1],"lookback":key[2],**v} for key,v in sorted(out.items())]


if __name__=="__main__":
    games=load(HERE/"opportunities.json.gz")
    raw=[]
    for game in games:
        raw.append({"source":game["path"],"seed":game["seed"],"candidate_seat":game["candidate_seat"],"counts":process(game)})
    groups=defaultdict(Counter)
    for game in raw:
        for row in game["counts"]:
            key=(row["class"],row["lookback"],row["side"],game["seed"])
            groups[key].update({k:v for k,v in row.items() if k not in ("class","lookback","side")})
    summary=[]
    for label in ("all","reached_safe_rich","planted_safe_rich"):
        for delay in (0,3,5,10):
            for side in (0,1):
                rows=[r for k,r in groups.items() if k[:3]==(label,delay,side) and r["same_worker"]]
                metrics={k:mean(r[k]/r["same_worker"] for r in rows) for k in
                         ("enemy_half","enemy_one_step","own_tree","enemy_tree","distance_moved")}
                summary.append(dict(label=label,lookback=delay,side=side,maps=len(rows),**metrics))
    save(HERE/"layout_history.json",dict(map_equal=summary,per_game=raw))

"""Recover official outcomes independently from cash, including the extinction case in the Top5 sample."""
from collections import Counter
from pathlib import Path
from dataset import HERE,SEQ,terr,read,write,sha


def main():
    sources=read(SEQ/"sources.json")
    rows=read(HERE/"rows.json.gz")
    num=Counter((r["game"],r["seat"])for r in rows)
    official=[];corrected=[]
    for c in sources:
        s=c["seat"]
        f=(terr.CACHE if c["source"]=="Top5" else terr.ROOT)/c["path"]
        assert sha(f)==c["sha256"]
        r=read(f)
        if c["source"]=="Top5":
            end=r["steps"][-1]
            cash=[p[0] for p in end[0]["observation"]["players"]]
            score=[x["reward"] for x in end]
            assert score==r["rewards"] and [x["status"] for x in end]==["DONE","DONE"]
        else:
            score=r["official_score_by_seat"]
            cash=r["last_post_cash"]
        entry=dict(game=c["sha256"],seed=c["seed"],seat=s,source=c["source"],split=c["split"],terrain=c["terrain"],
                   official_score=score,physical_cash=cash,official_margin=score[s]-score[1-s],rows=num[c["sha256"],s])
        if c["source"]=="Top5" and score!=cash:corrected.append(entry)
        official.append(entry)
    by={(c["game"],c["seat"]):c for c in official}
    sign_changes=sum((by[r["game"],r["seat"]]["official_margin"]<0)!=(r["eventual_margin"]<0) for r in rows)
    write(HERE/"official_scores.json",dict(score_authority="official terminal rewards; Top5 recorded post-game cash differs on one extinction",effects="The original rows.json.gz descriptive eventual_margin uses physical cash for ladder rows. All likelihood inputs and 24-turn labels are independent of terminal rewards; join by game and physical seat for the official terminal margin.",sign_changes=sign_changes,changed_top5=corrected,maps=official))
    assert sign_changes==0


if __name__=="__main__":main()

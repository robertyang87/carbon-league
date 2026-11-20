"""Preserve exact zero contrasts even on maps containing an unpriced, unchosen rival divergence."""
import json
from pathlib import Path
from statistics import mean
from learning import sha, save


HERE=Path(__file__).resolve().parent


def effects(rows):
    metrics={}
    for board in rows:
        maps=metrics.setdefault(str(board["seed"]),{})
        first=board["first"]
        single=[int(first==s) for s in (256,280)]
        both=[int(r["elite"])/2 for r in board["windows"]]
        coefficients={"single_vs_bc":single,"single_vs_elite_256":[single[0]-1,single[1]],
                      "single_vs_elite_280":[single[0],single[1]-1],"both_roots_vs_bc":both,
                      "both_roots_vs_elite":[both[0]-.5,both[1]-.5]}
        for k,weights in coefficients.items():
            out={}
            for measure in ("margin","win","raw_margin","raw_win"):
                terms=[(w,r[measure]) for w,r in zip(weights,board["results"]) if w!=0]
                out[measure]=None if any(v is None for w,v in terms) else sum(w*v for w,v in terms)
            maps[k]=out
    return metrics


def summarize(maps):
    result={}
    for k in next(iter(maps.values())):
        stats={}
        for metric in ("margin","win","raw_margin","raw_win"):
            values=[m[k][metric] for m in maps.values()]
            present=[q for q in values if q is not None]
            stats[metric]=dict(known=len(present),total=len(values),all_map_mean=mean(present) if len(present)==len(values) else None,
                               known_only_mean=mean(present) if present else None)
        result[k]=stats
    return result


if __name__=="__main__":
    p=HERE/"assessment.json"
    assessment=json.loads(p.read_text())
    maps=effects(assessment["maps"])
    save(HERE/"partial_effects.json",dict(assessment_sha256=sha(p),deployed_eligible=assessment["development_eligible"],per_terrain=maps,
            summary=summarize(maps),method="A structurally matched choice contributes zero even when the unchosen competing arm has a contaminated effect. Preserve unidentified terms whenever their contrast coefficient is nonzero. Existing strict preregistered all-root advance gate does not change."))

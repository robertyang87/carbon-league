"""Bind a tree_value_labels-format paired terminal trace only after confirming its physical single-worker intervention."""
from math import isfinite
from order import canonical


def bind_terminal_returns(anchor, ticket, protocol_sha, parent_run, alternative_run):
    seat=anchor.seat
    for run in (parent_run,alternative_run):
        if (run["protocol"]!=protocol_sha or run["seed"]!=ticket["seed"] or run["root"]!=anchor.observation["step"]
                or run["seat"]!=seat or run["status"]!=["DONE","DONE"] or run["hard_calls"]
                or run["steps"]!=len(run["rows"]) or run["final_t"]!=run["root"]+run["steps"]):
            raise ValueError("Run identity or terminal status differs from the source contract")
        if not run["rows"] or run["rows"][0]["public_sha256"]!=ticket["public_root_hashes"]:
            raise ValueError("Wrong root observation")
        if run["rows"][0]["actions"][1-seat]!=run["root_enemy"]:
            raise ValueError("The first executed opponent action differs")
        if not all(isfinite(x) for x in run["scores"]) or round(run["scores"][seat]-run["scores"][1-seat],2)!=round(run["official_margin"],2):
            raise ValueError("Inconsistent official scores")
    if parent_run["root_enemy"]!=alternative_run["root_enemy"]:
        raise ValueError("Branches do not share a simultaneous root opponent response")
    if canonical(parent_run["rows"][0]["actions"][seat],anchor.own)!=anchor.base:
        raise ValueError("Reference run did not execute the actual parent order")
    audit=anchor.audit(alternative_run["rows"][0]["actions"][seat])
    if not audit["accepted"]:
        raise ValueError("Not an admissible single-worker intervention: "+str(audit["reasons"]))
    return dict(anchor=anchor.anchor_id,training_map=ticket["terrain"],focal=anchor.focal,first_action_change=audit["changed"][0],
                parent_final_scores=parent_run["scores"],alternative_final_scores=alternative_run["scores"],
                alternative_minus_parent=round(alternative_run["official_margin"]-parent_run["official_margin"],2),
                common_enemy_order=parent_run["root_enemy"],terminal_protocol_sha256=protocol_sha)

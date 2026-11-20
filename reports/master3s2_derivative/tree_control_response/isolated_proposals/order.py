"""Pure offline candidate construction around an already issued whole public order.

No model invocation, sampled proposal, hypothetical enemy order, or outcome enters this API.
"""
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json

MOVE = {"NONE": (0, 0), "UP": (-1, 0), "RIGHT": (0, 1), "DOWN": (1, 0), "LEFT": (0, -1)}
BASE = {"NONE", "RECCOLLECTOR", "RECPLANTER"}


def digest(obj):
    return sha256(json.dumps(obj, separators=(",",":"), sort_keys=True, allow_nan=False).encode()).hexdigest()


def destination(cell, action, size):
    r,c=divmod(cell,size);dr,dc=MOVE[action]
    return ((r+dr)%size)*size+(c+dc)%size


def canonical(action, own):
    if not isinstance(action,dict):
        raise ValueError("The issued whole order must be an entity-to-action mapping")
    out={}
    for uid,a in action.items():
        valid=BASE if uid in own[1] else MOVE if uid in own[2] else ()
        if a not in valid:raise ValueError(f"Unknown entity or action: {uid}={a}")
        if a!="NONE":out[uid]=a
    return out


@dataclass
class Anchor:
    observation: dict
    order: dict
    focal: str
    config: dict

    def __post_init__(self):
        self.seat=self.observation["player"]
        if self.seat not in (0,1):raise ValueError("Expected a two-seat public observation")
        self.players=self.observation["players"];self.own=self.players[self.seat]
        w=self.own[2].get(self.focal)
        if w is None or w[2]!="PLANTER":raise ValueError("The named focal planter must be live on the sending seat")
        self.source=w[0];self.base=canonical(self.order,self.own);self.size=self.config["size"]
        self.other_destinations=defaultdict(list)
        for uid,row in self.own[2].items():
            if uid!=self.focal:self.other_destinations[destination(row[0],self.base.get(uid,"NONE"),self.size)].append(uid)
        c=sum(w[2]=="COLLECTOR" for p in self.players for w in p[2].values())
        n=sum(len(p[2]) for p in self.players)
        costs={"RECCOLLECTOR":self.config["recCollectorCost"]+c*self.config["recCollectorIncreaseCost"],
               "RECPLANTER":self.config["recPlanterCost"]+(n-c)*self.config["recPlanterIncreaseCost"]}
        cash=self.own[0];count=len(self.own[2]);self.reservations={}
        # The referee takes recruit commands in the incoming base order before worker movement.
        for uid,pos in self.own[1].items():
            act=self.base.get(uid)
            if act and count<self.config["workerLimit"] and cash>=costs[act]:
                cash-=costs[act];count+=1;self.reservations[pos]=uid
        self.cash_after_recruits=cash
        self.anchor_id=digest([self.observation,self.base,self.focal])

    def audit(self, proposed):
        try:alt=canonical(proposed,self.own)
        except ValueError as e:return dict(accepted=False,reasons=["invalid_command"],known_error=str(e))
        entities=sorted(self.base.keys()|alt.keys())
        changed=[dict(id=u,base=self.base.get(u,"NONE"),alternative=alt.get(u,"NONE")) for u in entities if self.base.get(u)!=alt.get(u)]
        reasons=[]
        if not any(r["id"]==self.focal for r in changed):reasons.append("focal_unchanged")
        if any(r["id"]!=self.focal for r in changed):reasons.append("other_entity_changed")
        dest=destination(self.source,alt.get(self.focal,"NONE"),self.size)
        blocked=self.other_destinations.get(dest,[])
        if blocked:reasons.append("allied_destination")
        if dest in self.reservations:reasons.append("active_recruit_destination")
        return dict(anchor=self.anchor_id,accepted=not reasons,reasons=reasons,changed=changed,
                    destination=dest,allied_occupants=sorted(blocked),recruiting_base=self.reservations.get(dest),
                    parent_order=self.base,candidate_order=alt)

    def candidates(self):
        choices={}
        orig=self.base.get(self.focal,"NONE")
        for action in MOVE:
            if action==orig:continue
            proposed=self.base.copy()
            if action=="NONE":proposed.pop(self.focal,None)
            else:proposed[self.focal]=action
            choices[action]=self.audit(proposed)
        return choices

    def receipt(self):
        rosters={str(k):v for k,v in sorted(self.other_destinations.items())}
        return dict(anchor=self.anchor_id,public_sha256=digest(self.observation),focal=self.focal,cell=self.source,
                    step=self.observation["step"],seat=self.seat,parent_order=self.base,parent_order_sha256=digest(self.base),
                    fixed_other_worker_destinations=rosters,active_recruit_cells=self.reservations,
                    available_cash_after_recruitment=self.cash_after_recruits, alternatives=self.candidates())

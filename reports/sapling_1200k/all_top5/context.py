"""Read a seat's causal decision background from the local, lossless Carbon League archive."""
import argparse
import bisect
import copy
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RULE_FILE = ROOT / "code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json"
PUBLIC_RULES = frozenset(json.loads(RULE_FILE.read_bytes())["configuration"]) - {"randomSeed"}
TIME_LIMITS = ("step_seconds", "overage_seconds", "hard_step_seconds", "max_slow_steps",
               "init_seconds", "initialization_seconds", "run_seconds")
OBS_FIELDS = frozenset(("step","player","remainingOverageTime","carbon","players","trees"))
LABEL_GROUPS = ("actions","next","outcome","full_carbon","opponent_private","runtime")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git_blob(root, spec):
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0")
    return subprocess.check_output(["git","-C",str(root),"-c","protocol.allow=never","cat-file","blob",spec],env=env)


def registered_context_receipts():
    for receipt in sorted(HERE.glob("context_*receipt.json")):
        batch = receipt.stem.removeprefix("context").removesuffix("_receipt").lstrip("_")
        yield receipt, HERE / batch / "manifest.json"
    for receipt in sorted(HERE.glob("appendix_*/context_receipt.json")):
        yield receipt, receipt.parent / "manifest.json"


class Corpus:
    """Resolve a decision by tuple or dense ordinal. Each opened game is independently hash checked."""

    def __init__(self, root=ROOT, index=None, *, manifest=None):
        self.root=Path(root)
        self._manifest=Path(manifest) if manifest is not None else HERE/"manifest.json"
        self._cards=None
        if manifest is not None and index is not None:raise ValueError("select only one catalog source")
        if manifest is not None:
            if __package__:
                from .context_catalog import catalog
            else:
                from context_catalog import catalog
            rows=catalog(manifest,self.root)
        else:
            data=Path(index or HERE/"context_index.tsv.gz").read_bytes()
            if index is not None:
                # Each published receipt binds the index bytes to its batch; copies can be renamed.
                fingerprint=digest(data)
                matches=[]
                for receipt, batch_manifest in registered_context_receipts():
                    if json.loads(receipt.read_bytes())["index_sha256"]==fingerprint:
                        matches.append(batch_manifest)
                if len(matches)>1:
                    raise ValueError("decision index belongs to multiple registered batches")
                if matches:
                    self._manifest=matches[0]
            with io.StringIO(gzip.decompress(data).decode("utf-8")) as stream:
                rows=list(csv.DictReader(stream,delimiter="\t"))
        self.rows={r["id"]:r for r in rows}
        self.order=[r["id"] for r in rows]
        self.starts=[int(r["ordinal_begin"]) for r in rows]
        self.size=int(rows[-1]["ordinal_end_exclusive"])

    def locate(self, ordinal):
        if type(ordinal) is not int or not 0<=ordinal<self.size: raise IndexError(ordinal)
        i=bisect.bisect_right(self.starts,ordinal)-1
        t,seat=divmod(ordinal-self.starts[i],2)
        return self.order[i],t,seat

    def ordinal(self, gid, turn, seat):
        r=self.rows[gid]
        if type(turn) is not int or not 0<=turn<int(r["transitions"]): raise IndexError(turn)
        if type(seat) is not int or seat not in (0,1):raise IndexError(seat)
        return int(r["ordinal_begin"])+turn*2+seat

    def provenance(self, gid):
        """Optional scorecard identity and date. Scores and winners are accessible via labels."""
        if gid not in self.rows:raise KeyError(gid)
        if self._cards is None:
            manifest=json.loads(self._manifest.read_bytes())
            ranks={p["id"]:p["rank"] for p in manifest["top5"]}
            self._cards={r["id"]:{"match_game_id":r["id"], "created_unix":r["card"].get("created"),
                         "seat_names":r["seat_names"],"seat_strategy_ids":r["seat_strategy_ids"],
                         "seat_ranks_at_snapshot":[ranks.get(k) for k in r["seat_strategy_ids"]],
                         "ranking_as_of_utc":manifest.get("ranking_as_of_utc"),"cohort":r["cohort"],
                         "source_receipt":r.get("source")} for r in manifest["games"]}
        return copy.deepcopy(self._cards[gid])

    def open(self, gid):
        r=self.rows[gid]
        src=self.root/r["source"]
        if r["git_blob"]:
            blob=src.read_bytes() if src.is_file() else git_blob(self.root,r["git_blob"])
            if digest(blob)!=r["replay_sha256"]:
                blob=git_blob(self.root,r["git_blob"])
        elif src.is_file() and r["tar_byte_offset"]:
            with src.open("rb") as f:
                f.seek(int(r["tar_byte_offset"]))
                blob=f.read(int(r["packed_bytes"]))
        else:
            contents=None if src.is_file() else io.BytesIO(git_blob(self.root,"HEAD:"+r["source"]))
            with tarfile.open(src if contents is None else None,mode="r:*",fileobj=contents) as archive:
                entry=archive.getmember(r["member"])
                if not entry.isfile():raise ValueError(f"unexpected archive member {gid}")
                blob=archive.extractfile(entry).read()
        if digest(blob)!=r["replay_sha256"]:raise ValueError(f"replay digest mismatch for {gid}")
        try:
            from pydantic_core import from_json
        except ImportError:
            from_json=json.loads
        game=Game(from_json(gzip.decompress(blob)))
        if game.turns!=int(r["transitions"]):raise ValueError(f"length differs from index for {gid}")
        return game


def visibility(workers, size):
    """Official rule: collectors see their own cell; planters see their four toroidal neighbours."""
    seen=[False]*(size*size)
    for cell,_,kind in workers.values():
        if kind=="COLLECTOR":
            seen[cell]=True
        elif kind=="PLANTER":
            y,x=divmod(cell,size)
            for pos in (((y+1)%size)*size+x,((y-1)%size)*size+x,
                        y*size+(x+1)%size,y*size+(x-1)%size):
                seen[pos]=True
        else:
            raise ValueError(f"unknown worker role {kind}")
    return seen


class Game:
    def __init__(self, replay):
        self._data=replay
        self._frames=replay["steps"]
        self._schema=replay["specification"]["observation"]
        self.turns=len(self._frames)-1
        self.size=int(replay["configuration"]["size"])
        # Unexpected observation schemas need review before their fields can enter a policy.
        public={k for k,s in self._schema.items() if not s.get("hidden",False)}
        if public!=OBS_FIELDS or not self._schema.get("full_carbon",{}).get("hidden"):
            raise ValueError(f"unrecognized public observation schema {public}")

    def _frame(self, t, seat, terminal=False):
        max_t=self.turns if terminal else self.turns-1
        if type(t) is not int or not 0<=t<=max_t:raise IndexError(f"turn {t}, last valid {max_t}")
        if type(seat) is not int or seat not in (0,1):raise IndexError(f"physical seat {seat}")
        return self._frames[t]

    def _obs(self, t, seat, terminal=False):
        frame=self._frame(t,seat,terminal)
        # Mirrors core.py's shared-property projection. Neither opposite-seat private fields nor
        # hidden shared-root fields are copied. All public properties here are top-level.
        own=frame[seat]["observation"]
        shared=frame[0]["observation"]
        return {key:(shared if prop.get("shared",False) else own)[key]
                for key,prop in self._schema.items() if key in OBS_FIELDS and not prop.get("hidden",False)}

    def _assets(self, obs):
        result=[]
        for seat,(cash,bases,workers,trees) in enumerate(obs["players"]):
            result.append({"seat":seat,"cash":cash,"bases":dict(bases),
                "workers":{wid:{"position":v[0],"cargo":v[1],"role":v[2]} for wid,v in workers.items()},
                "trees":{tid:{"position":v[0],"age":v[1],"owner":seat,
                              "planter_id_or_null":obs["trees"][tid][0],
                              "previous_transition_income":obs["trees"][tid][1]} for tid,v in trees.items()}})
        return result

    def snapshot(self, t, seat):
        """The pre-action observation and explicit visibility for this physical seat."""
        obs=self._obs(t,seat)
        mask=visibility(obs["players"][seat][2],self.size)
        return {"observation":copy.deepcopy(obs),"visibility_mask":mask,
                "carbon_known_or_null":[n if seen else None for n,seen in zip(obs["carbon"],mask,strict=True)],
                "assets_by_physical_seat":self._assets(obs),
                "own_previous_command":copy.deepcopy(self._frames[t][seat]["action"]) if t else None}

    def memory(self, t, seat):
        self._frame(t,seat)
        cells=self.size**2
        carbon=[None]*cells;seen_at=[None]*cells
        for k in range(t+1):
            obs=self._obs(k,seat)
            for pos,seen in enumerate(visibility(obs["players"][seat][2],self.size)):
                if seen:
                    carbon[pos]=obs["carbon"][pos];seen_at[pos]=k
        return {"last_observed_carbon":carbon,"last_observed_at":seen_at}

    def entity(self, t, seat, uid):
        """Entity trail through t, including position changes and own earlier commands when observable."""
        self._frame(t,seat)
        result=[]
        for k in range(t+1):
            obs=self._obs(k,seat)
            for s,(cash,bases,workers,trees) in enumerate(obs["players"]):
                if uid in workers:
                    pos,cargo,kind=workers[uid];record={"role":kind,"position":pos,"cargo":cargo}
                elif uid in trees:
                    pos,age=trees[uid];record={"role":"TREE","position":pos,"age":age,
                                               "previous_transition_income":obs["trees"][uid][1]}
                elif uid in bases:
                    record={"role":"BASE","position":bases[uid]}
                else:
                    continue
                record.update(turn=k,owner=s)
                if k<t and s==seat and record["role"]!="TREE":
                    record["own_recorded_command_from_here"]=self._frames[k+1][seat]["action"].get(uid)
                result.append(record)
        return result

    def decision(self, t, seat, lookbacks=(3,5,10), *, history=False, memory=False):
        self._frame(t,seat)
        if any(type(x)is not int or x<1 for x in lookbacks):raise ValueError("lookbacks must be positive integers")
        cfg=self._data["configuration"]
        rules={k:copy.deepcopy(v) for k,v in cfg.items() if k in PUBLIC_RULES}
        platform=self._data.get("info",{}).get("platform_time_control",{})
        limits={k:platform[k] for k in TIME_LIMITS if k in platform}
        out={"at":self.snapshot(t,seat),"game_rules":rules,"declared_platform_time_limits":limits}
        # At t<lookback there is no such preceding state. Do not silently substitute frame zero.
        out["lookbacks"]={str(n):self.snapshot(t-n,seat) if n<=t else None for n in lookbacks}
        if history:out["prior_states"]=[self.snapshot(k,seat) for k in range(t)]
        if memory:out["visibility_memory"]=self.memory(t,seat)
        return out

    def labels(self,t,seat,groups):
        """Explicit offline targets. Never called by decision()."""
        self._frame(t,seat)
        groups=set(groups)
        if not groups or not groups<=set(LABEL_GROUPS):raise ValueError("choose at least one named label group")
        out={}
        if "actions" in groups:
            out["joint_commands_from_t"]=copy.deepcopy([a["action"] for a in self._frames[t+1]])
        if "next" in groups:
            out["next_own_observation"]=copy.deepcopy(self._obs(t+1,seat,terminal=True))
        if "outcome" in groups:
            rewards=self._data["rewards"]
            out["final_reward_by_physical_seat"]=list(rewards);out["final_statuses"]=list(self._data["statuses"])
            out["final_margin_for_seat"]=rewards[seat]-rewards[1-seat]
        if "full_carbon" in groups:out["hidden_full_carbon_at_t"]=list(self._frames[t][0]["observation"]["full_carbon"])
        if "runtime" in groups:
            out["recorded_post_action_runtime_info_by_seat"]=[copy.deepcopy(x.get("info",{})) for x in self._frames[t+1]]
        if "opponent_private" in groups:
            opp=self._obs(t,1-seat)
            out["opponent_private_at_t"]={"carbon":list(opp["carbon"]),"remainingOverageTime":opp["remainingOverageTime"],
                                          "visibility_mask":visibility(opp["players"][1-seat][2],self.size)}
        return out

    def metadata(self):
        """Raw provenance and specification. Kept outside the policy row."""
        return copy.deepcopy({k:self._data[k] for k in ("id","name","title","description","version","schema_version",
                              "configuration","specification","info") if k in self._data})

    def training_rows(self, seat, *, lookbacks=()):
        for t in range(self.turns):
            yield self.decision(t,seat,lookbacks=lookbacks)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("address",nargs="*",help="ID T SEAT, or one global ordinal")
    p.add_argument("--manifest",type=Path,help="select a sibling appendix manifest with a separate ordinal space")
    p.add_argument("--lookbacks",nargs="*",type=int,default=[3,5,10])
    p.add_argument("--memory",action="store_true");p.add_argument("--history",action="store_true")
    p.add_argument("--entity",help="return this observed entity's history through the requested turn")
    p.add_argument("--labels",nargs="+",choices=LABEL_GROUPS)
    p.add_argument("--metadata",action="store_true",help="include raw provenance outside the decision")
    args=p.parse_args();c=Corpus(manifest=args.manifest)
    if len(args.address)==1: gid,t,seat=c.locate(int(args.address[0]))
    elif len(args.address)==3:gid=args.address[0];t,seat=map(int,args.address[1:])
    else:p.error("supply ID T SEAT, or a global ordinal")
    game=c.open(gid)
    payload={"decision":game.decision(t,seat,args.lookbacks,history=args.history,memory=args.memory)}
    if args.entity:payload["entity_history"]=game.entity(t,seat,args.entity)
    if args.labels:payload["labels"]=game.labels(t,seat,args.labels)
    if args.metadata:
        payload["metadata"]=game.metadata()
        payload["metadata"]["match_provenance"]=c.provenance(gid)
    print(json.dumps(payload,ensure_ascii=False,separators=(",",":"),allow_nan=False))


if __name__=="__main__":main()

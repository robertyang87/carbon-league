"""Build a byte-reversible, two-finalist cash-projection overlay on the single-file V2 agent."""
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
LEARN=HERE.parent
PARENT=LEARN.parent/"cold_start/candidate.py"
PARENT_SHA="7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed"
WEIGHTS_SHA="ceed9c721d2ff70bcd2a47bb47dacb5ca88b51073e8a7fb9beefe3598e679cd6"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def runtime():
    w=json.loads((LEARN/"weights.json").read_bytes())
    source=(LEARN/"runtime.py").read_bytes()
    if sha(source)!=w["runtime_sha256"] or sha((LEARN/"weights.json").read_bytes())!=WEIGHTS_SHA:
        raise ValueError("Changed trained projector")
    text=source.decode()
    # The original already builds this exact toroidal distance/neighbor table at import.
    # Reuse its table to avoid an additional cold start or lazy first inference allocation.
    start=text.index("AGGREGATE = ")
    end=text.index("\n\n@lru_cache",start)
    features=text[start:end]+"\n\n"+text[text.index("def _side("):]
    features=features.replace("distances,adj = lattice(boardsize)","distances,adj = DIST,NB")
    replace={"AGGREGATE":"_CASH_AGGREGATE","BASE":"_CASH_BASE","PHYSICAL":"_CASH_PHYSICAL","FULL":"_CASH_FULL",
             "_side":"_cash_side","features":"_cash_features","predict_residual":"_cash_residual","math":"_cash_math"}
    features=re.sub(r"\b("+ "|".join(replace) +r")\b",lambda m:replace[m.group()],features)
    model={k:w["enhanced"][k] for k in ("weights","scale")}
    return "import math as _cash_math\n_CASH_HEAD = "+repr(model)+"\n"+features+"\n"+(HERE/"selector.py").read_text()+"\n\n"


def edits():
    return [
      # Purely observational roster references, only on the two evidence windows. The
      # original turn engine, all pruning and the original native evaluation stay intact.
      ("def _search(obs, configuration, est, act0, lw0, lb0, prev, t0, home=()):",runtime(),True),
      ("    players = obs['players']\n    full = [1.0] * N\n","    cash_window = LAST_TURN + 1 - step in (14, 20)\n",False),
      ("                out.append([c, m, _turn(root, configuration, acts), players])\n",
       "                if cash_window:\n                    out[-1].append([out[-1][2]['players']])\n",False),
      ("            b[3] = o['players']; b[2] = _turn(o, configuration, [am, ao] if me == 0 else [ao, am])\n",
       "            if cash_window:\n                b[4].append(b[2]['players'])\n",False),
      ("    if _S_REC is not None:\n        _S_REC.append((step, me, cands,",
       "    if cash_window:\n        best, cash_reason, cash_pair, cash_scores = _cash_pick(best, alive, mean, dep, run, cands, depth, rolls, cut, step, me, configuration, t0)\n",True),
    ]


def convert(s):
    return s.replace("\n","\r\n").encode()


def build():
    parent=PARENT.read_bytes();out=parent
    if sha(parent)!=PARENT_SHA:raise ValueError("Changed V2 parent")
    marks=[]
    for i,(anchor,extra,before) in enumerate(edits()):
        a=convert(anchor);v=convert(extra)
        if out.count(a)!=1 or v in out:raise ValueError("Nonunique build edit "+str(i))
        out=out.replace(a,v+a if before else a+v);marks.append(v)
    round_trip=out
    for mark in reversed(marks):
        if round_trip.count(mark)!=1:raise ValueError("Nonunique removal")
        round_trip=round_trip.replace(mark,b"",1)
    if round_trip!=parent or len(out)>10_000_000:raise ValueError("Source recovery or file size")
    return out,dict(sha256=sha(out),bytes=len(out),parent_sha256=sha(parent),stripped_sha256=sha(round_trip),weights_sha256=WEIGHTS_SHA,
                    runtime_sha256=sha((LEARN/"runtime.py").read_bytes()),selector_sha256=sha((HERE/"selector.py").read_bytes()),
                    evidence_protocol="Parent V2, single common roll, native fully completed H8, actual native winner plus actual runner-up among survivors, remaining 14 or 20. On a branch that already terminated or any late/nonfinite/exceptional case, keep native order.",
                    extra_network_forwards=0,active_root_steps=[279,285],size_cap=10_000_000)


if __name__=="__main__":
    out,receipt=build();(HERE/"candidate.py").write_bytes(out)
    (HERE/"manifest.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(receipt["sha256"],len(out))

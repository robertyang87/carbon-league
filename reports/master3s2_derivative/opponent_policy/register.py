"""Admit only internally run games with identity-checked original-opponent command traces."""
import hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
HIST=ROOT/"reports/sapling_1200k/master6_actor_dev/strategic_destination_learning/sources.json"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    historical=json.loads(HIST.read_bytes())
    games=[{k:g[k] for k in ("path","seed","terrain","split","candidate_seat","expert_seat","sha256","actions_sha256","transitions","official_cash_path_sha256")}
           for g in historical["cards"] if g["split"] in ("train","val")]
    need=lambda ok:ok or (_ for _ in ()).throw(ValueError("Historical panel failed"))
    need(len(games)==56 and len({(g["terrain"],g["candidate_seat"]) for g in games})==56)
    terrain={k:{g["terrain"] for g in games if g["split"]==k} for k in ("train","val")}
    need(len(terrain["train"])==24 and len(terrain["val"])==4 and not terrain["train"]&terrain["val"])
    need(all(g["candidate_seat"]==1-g["expert_seat"] and g["seed"] not in range(621758580,621758584) for g in games))
    protocol=dict(source_master_sha256="65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5",source_manifest=sha(HIST),
        games=games,steps=list(range(7,299,13)),opponent_hash="57f9a594cce4f2ec9e3573d8d9d26ca1",
        operation="Only the one original root opponent policy inference in _search is calibrated, after net(xo,_SID), before _sample. Preserve original unmodified own action prior, both seats' rollout policy, critic and raw 0.58 deadline. Explicit one-line call and delimited source addition. Base and worker heads use original borrowed-free predictions from this frozen source. No third-party elite replay or weights. All game inputs are self-generated archive histories. Supervise exclusively the original examiner's actual simultaneous commands.",
        x="Every previous turn updates original BeliefTracker with our own seat's visible carbon + public players and our actual prior issued order. At selected root, original opponent view xo uses OUR belief est, all-ones knowledge argument (as in original search), opponent perspective, previous public players, original _quant rounding. Read only original deployed backbone's logits and already-available xo at enemy unit and neighbours. Adjacent published planes contain physical role/trees/cargo/base/carbon. Both opposite commands and final scores excluded from features. Forecast each physical arrival (per-unit modal destination and summed complete joint response after native collision masking).",
        fitting="Map-equal role CE: frozen upstream real root logits plus a 2-layer elementwise tanh fixed analytical embedding of own publicly estimated xo to a learned linear worker/base delta (no action rules). 50 L-BFGS updates max. KL(old||new) coefficient 1, L2=.01 on normalized readout weights including bias; fp16 checkpoint tensors cast to float32 at runtime. Multiply output by fixed 0.5 steps until fitting-set mean KL<=0.03, P99(abs(logit_delta))<=1.25. Source map/steps chosen before loading raw labels, zero-delta restores original physical logits, every held seat contributes even if acc decreases.",
        checks="Keep held four source physical maps and both corresponding applicant seats. Both heads and both seats nonworsening in map mean CE, overall mean CE decrease, >=3/4 physical maps improved; evaluate accurate worker arrivals and realized same-cell collisions (using actual own/other commands only as postdecision labels), counting every held root including no encounters. Leave terminal win-rate to same-machine new-map comparison against original; local complete original-search root tests require real 0.58 timing, mother/mother negative controls, original-vs-variant same prior and explicit changed returned command. No high-score selection.")
    dst=HERE/"protocol.json";new=json.dumps(protocol,sort_keys=True,indent=2)+"\n"
    if dst.exists() and dst.read_text()!=new:raise ValueError("Changed recorded panel")
    dst.write_text(new)
    print(sha(dst))
if __name__=="__main__":main()

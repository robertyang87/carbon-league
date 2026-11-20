"""Freeze the conditional forecast, role decoder and untouched held maps before fitting."""
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
D=H.parent
ROOT=D.parents[1]
def sha(x):return hashlib.sha256(x).hexdigest()
def read(p):return json.loads(p.read_bytes())
def dump(path,obj):
    raw=(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2,sort_keys=True)+'\n').encode()
    if path.exists() and path.read_bytes()!=raw:raise ValueError(path)
    path.write_bytes(raw);return sha(raw)
def main():
    protocol=D/'ladder/bilateral_delivery_learning/protocol.json'
    assert sha(protocol.read_bytes())=='820bce03de67871bcefcea3227ae3b234a08af3e5649301301dea6ae7b4919f0'
    b=read(protocol)
    state=read(D/'v4_replication_dev/hosted/settlement.json')
    tickets=[]
    for g in state['games']:
        a=next(a for a in state['archives'] if a['seed']==g['seed'] and a['seat']==g['seat'])
        file=D/'v4_replication_dev/hosted'/f"{a['id']}.zip"
        tickets.append(dict(domain='h2h',source=str(file.relative_to(ROOT)),sha256=a['sha256'],candidate=g['seat'],
            seed=g['seed'],terrain=g['terrain'],repeat=g['arm'],seats=[g['seat']],name='held'))
    assert {c['terrain'] for c in tickets}==set(b['held'].values())
    train=[dict(c,seats=[c['candidate']] if c['domain']=='h2h' else c['seats']) for c in b['train']]
    ac=D/'ladder/v4_alphacarbon_live_response/protocol.json'
    doc=dict(version='rollout-evasion-v1',parent='cd230d40a08f7dea07e261202b865f951b88b3b030febc9828627aa5f47c7d2a',train=train,held=tickets,
       cards='Only old 63 actual games, 40 terrain groups incl 35 vs complete original. From each actual candidate perspective learn to forecast the opposing side. Four development maps 610-613, V4 8 primary and V2 8 secondary; V4 ladder Alpha t267-298 two CRN streams times two arms are exposed diagnostics only.',alpha_sha=sha(ac.read_bytes()),
       alignment='At real observed t-1 reconstruct candidate-seat belief using own real history and private visible carbon; propagate both logged previous orders to construct rollout successor. Never use current own move or current enemy command to construct logits. Match observed worker by same role/position; actual enemy move at t is supervision only. Steps t=3,7,11,...,295; keep failed projections explicitly.',
       population='Heavily loaded enemy collector cargo>=100, has a lighter visible opposing collector at distance<=3, outside its banks, at least one strictly homeward neighbor lies within one cell of a lighter opposing collector and at least one action outside all such one-hop threats. One physical worker row per eligible reached sampled root; all other actors get exactly old logits.',
       target='Next physical enemy worker direction among NONE/UP/RIGHT/DOWN/LEFT. Predefine evade as a safer destination at distance>1 from all currently lighter opposing collectors, whose minimum distance to own bank does not decrease (incl stay). Condition on opportunity before seeing outcome; report actual safe homeward moves separately if applicable.',
       models='Frozen native unadjusted rollout logits; one learned inverse temperature in [.05,2] ever applied only inside support; same temperature plus shared 6-unit tanh direction readout from public mechanics/motion. L-BFGS max 80, seed 624739, residual bound=2.0 per option with L2=.002 on unit-normalized weights, map-equal NLL. No selection/refit from held.',
       evidence='Report worker NLL, categorical Brier, detour-event Brier (physical native team ordering via 24 common RNG draws at each eligible root). Require V4 held at least 3 maps and 50 eligible workers, physically decoded detour Brier below parent AND the same gated temperature control, 3/4 maps non-worse on this criterion relative to temperature, categorical Brier no worse than parent. On complete ordinary roots no actor can change (include reference calls on first 24 per map/seat/arm); report ordinary workers in mixed eligible roots separately. Keep all reversed metrics and behavior. Only if passing are funded V4 root/search probes warranted. No new entire match.')
    print(dump(H/'protocol.json',doc))
if __name__=='__main__':main()

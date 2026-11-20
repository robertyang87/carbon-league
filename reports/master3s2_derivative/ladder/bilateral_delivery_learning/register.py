"""Forecast the next ten actual self-deliveries on both sides of a registered historical match."""
import hashlib
import json
from pathlib import Path

H=Path(__file__).resolve().parent
DER=H.parents[1]
ROOT=DER.parents[1]
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(p):return json.loads(p.read_bytes())
def dump(p,o):
    raw=(json.dumps(o,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()
    if p.exists() and p.read_bytes()!=raw:raise ValueError(f'Frozen output differs {p}')
    p.write_bytes(raw)
    return sha(raw)

def main():
    old=DER/'native_critic_update/protocol.json'
    assert sha(old.read_bytes())=='e162d26dfbbcc2118d02450046fe330796212f7ed22aec826bfe2467ba147b3a'
    source=read(old)
    cards=[{k:v for k,v in c.items() if k!='split'} for c in source['tickets']]
    held=read(DER/'v4_replication_dev/protocol.json')['physical_boards']
    assert len(held)==4 and not {c['terrain'] for c in cards}&set(held.values())
    assert not set(source['excluded_terrain'])&{c['terrain'] for c in cards}
    p=dict(schema='bilateral-10step-delivery-v1',train=cards,held=held,other_excluded=source['excluded_terrain'],
      mask='Only physical players (cash/trees are public), time, acting seat, proposed complete own joint command. No actual simultaneous enemy command/hidden board/future variables. Both labels use REAL own_delivery cash writes on subsequent ten joint transitions. No cash gain from enemy walking into our base, tree income or cargo ownership counts unless credited as own_delivery.',
      selection='Train t=6,12,...288 when all ten transitions exist. Both seats in 35 archived original-facing games and our physical seat in 28 ladder games. Fit all 40 old maps equally, games then perspectives equally. All 21 true-pair physical maps and 610-613 excluded. Independent main=V4 on 610–613 both physical seats 8 games; secondary=V2. All are previously public outcomes, but this model has not accessed ten-step decomposition or held labels before freeze. No use of examples for training.',
      model='One two-output MLP 24 tanh units, positive softplus output in 500 carbon units. Shared state features include both sides of carrying/route lengths, nearby lighter attackers and defenders and simultaneous inbound multi-carrier conditions. Additional features see proposed own joint directions/recruitment only. Comparator identical 24-unit state-only model; numeric comparator affine ridge with each side current carried total. Map-equal Huber(1.0) to both real total deposits plus .0003 squared weight norm per parameter (sum across all weights), fixed L-BFGS 90 iterations seed 621034 applied to both neural heads. Training-internal normalizing moments; no tuned stop.',
      validation='By physical map, mean absolute error own, enemy and own-minus-enemy deliveries. Primary V4 net MAE must improve on both baselines; show pooled and per-map, multi-carrier and late 240+; zero/signed over/under counts. For old true double-response contrasts report predicted and actual differences in both receipts, net 10-step differential and independent official terminal margin; equal predictions stay on A, zero benefit if unchanged. Report exact commands A/B and counterexamples; no causal interpretation from historic single-action rows. All pairs with ten verified transitions; shorter horizons reported apart. Late ladder example not a test.',
      deployment='offline; no policy update, rank submission or games; criterion also requires at least two corrected, more correct than harmed and positive mean real net when switching from tie A across physical maps; otherwise report failed.')
    print(dump(H/'protocol.json',p))
if __name__=='__main__':main()

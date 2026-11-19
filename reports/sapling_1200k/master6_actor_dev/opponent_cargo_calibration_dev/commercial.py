"""Evaluate a frozen movement forecast only where a complete signed native joint order has a single-opponent interpretation."""
from collections import Counter, defaultdict, deque
from hashlib import sha256
import gzip
import json
from pathlib import Path
import statistics
import sys

from commercial_arms import audit_arms, lots
from model import Forecast

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PREVIOUS=HERE.parent/'joint_response_critic_dev'
MOVES=('NONE','UP','RIGHT','DOWN','LEFT')
DELTAS=((0,0),(-1,0),(0,1),(1,0),(0,-1))
sys.path.insert(0,str(HERE.parents[1]/'root_recovery_branches'))
from branches import begin, public
PANEL='f30f03153e995504cc9e6675d392baf96710beb7854dda808eea35c6bd823a69'


def sha(p):return sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def encoded(o):return json.dumps(o,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':')).encode()
def destinations(cell, size):
    r,c=divmod(cell,size)
    return [((r+dy)%size)*size+(c+dx)%size for dy,dx in DELTAS]


def value(obs,cfg,uid,command,forecast):
    """One candidate arrival by a changed collector; all other own commands are known, opponent commands are unobserved."""
    own,enemy=[obs['players'][n] for n in (obs['player'],1-obs['player'])];size=cfg['size']
    origin,load,kind=own[2][uid];assert kind=='COLLECTOR'
    order=command.get(uid,'NONE')
    endpoint=destinations(origin,size)[MOVES.index(order)];home=set(own[1].values());enemy_home=set(enemy[1].values())
    mass=load*(1-cfg['moveCost']) if endpoint!=origin else load
    result={'uid':uid,'own_order':order,'origin':origin,'endpoint':endpoint,'own_cargo_after_move':mass,
            'attributable':False,'expected_contact_net_own_cargo':0.,'safe_deposit_of_own_principal':0.}
    # Any possible recruitment or multi-party interaction at this edited destination invalidates the estimate.
    if endpoint in enemy_home:
        return dict(result,reason='enemy_base_or_possible_enemy_recruit')
    if any(uid!=u and destinations(v[0],size)[MOVES.index(command.get(u,'NONE'))]==endpoint for u,v in own[2].items()):
        return dict(result,reason='multiple_own_arrivals')
    if any(v==endpoint and command.get(k) in ('RECCOLLECTOR','RECPLANTER') for k,v in own[1].items()):
        return dict(result,reason='possible_own_recruit')
    rivals=[(k,v) for k,v in enemy[2].items() if endpoint in destinations(v[0],size)]
    if len(rivals)>1:return dict(result,reason='multiple_geometrically_possible_rivals')
    if not rivals:
        return dict(result,attributable=True,reason='no_possible_rival',safe_deposit_of_own_principal=max(0.,mass-cfg['smeltCost']) if endpoint in home else 0.)
    target,worker=rivals[0]
    if target not in forecast or worker[2]!='COLLECTOR' or worker[1]<=0:
        return dict(result,reason='possible_rival_outside_forecast',rival_uid=target)
    rival_act=destinations(worker[0],size).index(endpoint);q=forecast[target]
    move=q['move'];assert len(move)==5 and abs(sum(move)-1)<1e-5 and min(move)>=0
    rival_mass=worker[1]*(1-cfg['moveCost']) if rival_act else worker[1]
    # On a tie both die. A still-living own tree or our own base receives both cargos, otherwise we forfeit ours.
    if mass<rival_mass:net=rival_mass;settlement='own_collector_captures'
    elif mass>rival_mass:net=-mass;settlement='enemy_collector_captures'
    else:
        real_own_tree=any(v[0]==endpoint and v[1]<cfg['treeLifespan'] for v in own[3].values())
        net=rival_mass if endpoint in home or real_own_tree else -mass
        settlement='tie_owned_receiver' if endpoint in home or real_own_tree else 'tie_nonowned_receiver'
    return dict(result,attributable=True,reason='one_forecast_collector',rival_uid=target,rival_cargo_after_move=rival_mass,
                rival_direction=MOVES[rival_act],rival_probability=move[rival_act],settlement=settlement,if_contact_net_own_cargo=net,
                bank3=q.get('bank3'),bank5=q.get('bank5'),contest3=q.get('contest3'),contest5=q.get('contest5'),
                expected_contact_net_own_cargo=move[rival_act]*net)


def choose(obs,config,row,forecast):
    side=obs['player']
    edits=row['changed_uids'];evidence={}
    if any(obs['players'][side][2].get(uid,[None,None,None])[2]!='COLLECTOR' for uid in edits):
        return {'selected':'parent','reason':'not_all_edited_units_are_collectors'}
    for arm in ('parent','alternative'):evidence[arm]=[value(obs,config,u,row[arm],forecast) for u in edits]
    if any(not slot['attributable'] for slots in evidence.values() for slot in slots):
        return dict(selected='parent',reason='ambiguous_edited_endpoint',evidence=evidence)
    def difference(key):
        return sum(v[key] for v in evidence['alternative'])-sum(v[key] for v in evidence['parent'])
    contact=difference('expected_contact_net_own_cargo');deposit=difference('safe_deposit_of_own_principal')
    take=abs(contact)>1e-8 and contact+deposit>1e-8
    return dict(selected='alternative' if take else 'parent',reason='positive_comparison' if take else ('no_differential_contact' if abs(contact)<=1e-8 else 'retain_after_scoring'),
                expected_contact_delta=contact,safe_deposit_principal_delta=deposit,expected_score_delta=contact+deposit,evidence=evidence)


def outcome(row,arms,selected):
    seat=row['seat'];other=1-seat
    def diff(key):return arms[selected][key]-arms['parent'][key]
    end=[row[x+'_scores'] for x in (selected,'parent')]
    own=end[0][seat]-end[1][seat];enemy=end[0][other]-end[1][other]
    return {'own_official':own,'enemy_official':enemy,'official_margin':own-enemy,
            'root_worker_capture_own':diff('root_worker_capture_own'),'root_worker_capture_enemy':diff('root_worker_capture_enemy'),
            'own_normal_deposit':diff('root_normal_deposit_own')+diff('suffix_normal_deposit_own'),
            'enemy_normal_deposit':diff('root_normal_deposit_enemy')+diff('suffix_normal_deposit_enemy')}


def rollup(rows,policy):
    diffs=defaultdict(list)
    for row in rows:diffs[row['terrain']].append(row['outcomes'][policy])
    per_map={m:{k:statistics.mean(float(x[k]) for x in roots) for k in roots[0]} for m,roots in diffs.items()}
    return dict(roots=len(rows),maps=len(diffs),changes=sum(r['policies'][policy]['selected']=='alternative' for r in rows),
                change_maps=len({r['terrain'] for r in rows if r['policies'][policy]['selected']=='alternative'}),
                reasons=dict(Counter(r['policies'][policy]['reason'] for r in rows)),per_map=per_map,
                equal_map_delta={k:statistics.mean(m[k] for m in per_map.values()) for k in next(iter(per_map.values()),{})})


def main():
    panel_path=PREVIOUS/'signed_roots.json.gz';assert sha(panel_path)==PANEL
    native=read(HERE/'sources.json')['native'];split={c['terrain']:c['split'] for c in native}
    assert len({c['terrain'] for c in native if c['split']=='held'})==4
    panel=read(panel_path)['rows'];assert len(panel)==108 and len({r['identity'] for r in panel})==108
    model=Forecast(head_path=HERE/'head.npz')
    games=defaultdict(lambda:defaultdict(list))
    for r in panel:games[r['source']][r['t']].append(r)
    output=[]
    for path,at in sorted(games.items()):
        rawfile=ROOT/path;raw=read(rawfile);hash_of_source=sha(rawfile)
        seat=raw['candidate_seat'];cfg=raw['game_configuration'];sim=lots.FastCarbon(cfg).reset(seed=raw['seed']);official=begin(raw);hist=deque(maxlen=11)
        prepared=[]
        for t,commands in enumerate(raw['actions']):
            if t>max(at):break
            obs=public(official,seat);hist.append(obs)
            if t in at:
                forecast=model.predict(list(hist));flat={k:{**q,'move':[.2]*5} for k,q in forecast.items()}
                snapshot=sim.get_state()
                for r in at[t]:
                    assert (r['source_sha256'],r['seat'],r['t'],r['terrain'],r['parent'])==(hash_of_source,seat,t,raw['initial_terrain_u8_sha256'],commands[seat])
                    decisions={k:choose(obs,cfg,r,q) for k,q in [('forecast',forecast),('uniform',flat)]}
                    proof={k:v for k,v in obs.items() if k!='remainingOverageTime'}
                    assert sha256(encoded(proof)).hexdigest()==r['public_root_sha256'], (r['identity'],t)
                    prepared.append((r,snapshot,proof,decisions))
            sim.step(*commands);official.step(commands)
        # The official Board uses process-wide worker IDs. Seal the complete source
        # prefix before replaying either counterfactual suffix on that Board.
        for r,snapshot,proof,decisions in prepared:
            arms=audit_arms(r,snapshot,cfg,expected_public=proof)
            effect={k:outcome(r,arms,d['selected']) for k,d in decisions.items()}
            output.append({**{k:r[k] for k in ('identity','seed','seat','t','terrain','source_sha256','role','family')},
                'split':split[r['terrain']],'policies':decisions,'arms':arms,'outcomes':effect})
    assert len(output)==len(panel)
    groups={'all':output}
    groups.update({label:[r for r in output if r['split']==label] for label in ('train','calibration','held','recent_evaluation')})
    aggregate={label:{policy:rollup(rows,policy) for policy in ('forecast','uniform')} for label,rows in groups.items()}
    gate=aggregate['held']['forecast']; delta=gate['equal_map_delta']
    status='predictive_only' if gate['changes']==0 or delta['official_margin']<=0 or delta['own_normal_deposit']<=0 else 'positive_on_signed_held_actions_only'
    document=dict(schema='signed-commercial-action-check-v1',result=status,commercial_precommit_sha256=sha(HERE/'commercial_protocol.json'),
                  code_sha256=sha(Path(__file__)),ingestion_amendment_sha256=sha(HERE/'commercial_ingest_amendment.json'),arms_extractor_sha256=sha(HERE/'commercial_arms.py'),model_sha256=sha(HERE/'model.py'),
                  head_sha256=sha(HERE/'head.npz'),panel_sha256=PANEL,sources_sha256=sha(HERE/'sources.json'),
                  reports=aggregate,rows=output,fitness='No deployment games. Unknown alternatives in an actual expanded joint action menu have no signed outcome.')
    data=gzip.compress(encoded(document)+b'\n',mtime=0);p=HERE/'commercial.json.gz'
    if p.exists():assert p.read_bytes()==data
    else:q=p.with_suffix('.partial');q.write_bytes(data);q.replace(p)
    print(status,len(output),gate['changes'],delta,sha(p))


if __name__=='__main__':main()

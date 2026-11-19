"""Freeze the Master6 physical-seat corpus before policy fitting, with prior actor exclusions."""
import gzip,hashlib,json
from collections import Counter
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
BASE=HERE.parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(BASE))
from map_identity import local_board

BATCHES=['','appendix_499886f1','appendix_4bfdabf5','appendix_555c0c56','appendix_bf049043','appendix_bd3c0021']
M6='ecde65e596acbeb9ba5feca22e05f100'
PRIOR=['elite_bc_data.json','top5_h2h_data.json','cash_dagger_data.json','tree_dagger_data.json','soft_it50_data_manifest.json']
AUX=['all_top5_pressure_dev/split.json','planter_routes_dev/split.json','orchard_branches/plan.json','orchard_high_branches/plan.json','action_response/plan.json','latest5_inventory.json']
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(s):return hashlib.sha256(s.encode()).hexdigest()

def build():
    games=[];terrain={};source={}
    for batch in BATCHES:
        w=BASE/'all_top5'/batch; ms=w/'manifest.json';maps=w/'validation_maps.jsonl.gz'
        source[str(ms.relative_to(BASE))]=sha(ms);source[str(maps.relative_to(BASE))]=sha(maps)
        m=read(ms);points=[json.loads(line) for line in gzip.decompress(maps.read_bytes()).splitlines()]
        terrain.update({v['id']:v['initial_terrain_u8_sha256'] for v in points})
        for g in m['games']:
            if M6 not in g['seat_strategy_ids']:continue
            if g['seat_strategy_ids'].count(M6)!=1 or not g['frame_verified']:raise ValueError('teacher exposure unverified')
            games.append((batch,g))
    if len(games)!=302 or len(terrain)!=1373 or len(set(terrain.values()))!=1373:raise ValueError('unexpected frozen horizon')
    priids=set();priseeds=set();prior_boards=set();pri_witness={}
    for n in PRIOR:
        path=BASE/n;records=read(path)['games'];pri_witness[n]=sha(path)
        for g in records:
            priids.add(g['id']);seed=g['seed'];priseeds.add(seed)
            prior_boards.add(terrain[g['id']] if g['id'] in terrain else local_board(seed))
    # The exact development exposure inventory at the pre-training cut is frozen;
    # later native games must not alter the reproducible historical split.
    devseeds=set(read(HERE/'data_prior_local_seeds.json')['seeds'])
    current=read(HERE/'parent/registration.json')
    if current['seed'] not in devseeds:raise ValueError('missing predeclared development map')
    devboards={local_board(i) for i in devseeds}
    auxids=set()
    def walk(o):
        if isinstance(o,dict):
            if o.get('id') in terrain:auxids.add(o['id'])
            for v in o.values():walk(v)
        if isinstance(o,list):
            for v in o:walk(v)
    for n in AUX:
        path=BASE/n;pri_witness[n]=sha(path);walk(read(path))
    eligible=[]
    for batch,g in games:
        gid=g['id'];seat=g['seat_strategy_ids'].index(M6);pts=g['score_by_seat'];nonce=terrain[gid]
        margin=pts[seat]-pts[1-seat]
        prev=gid in priids or g['seed'] in priseeds or nonce in prior_boards
        dev=g['seed'] in devseeds or nonce in devboards
        reason='previous_actor_data' if prev else 'development_exposure' if dev else 'mutual_top5' if g['cohort']!='external' else 'non_clean' if not g['clean_terminal'] else None
        eligible.append({'id':gid,'seed':g['seed'],'terrain_sha256':nonce,'sha256':g['sha256'],'batch':batch,'seat':seat,
            'turns':g['transitions'],'cohort':g['cohort'],'opponent':g['seat_names'][1-seat],'wdl':'W' if margin>0 else 'L' if margin<0 else 'D',
            'margin':round(margin,6),'other_research_exposure':gid in auxids,'exclude':reason})
    pool=[z for z in eligible if z['exclude'] is None]
    even=sorted((g for g in pool if not g['other_research_exposure']),key=lambda x:key('m6-actor-1|'+x['terrain_sha256']))
    if len(even)<24:raise ValueError('insufficient disjoint auxiliary-policy maps')
    # Fourteen of the latest held maps were also used as descriptive cross-checks
    # for an older freight predictor. None trained/validated the parent actor; retain tag.
    prior_pressure={g['id'] for (batch,g) in games if batch=='appendix_4bfdabf5'}
    even.sort(key=lambda g:g['id'] in prior_pressure)
    hold={z['id'] for z in even[:18]};val={z['id'] for z in even[18:30]}
    for z in eligible:
        z['split']='excluded' if z['exclude'] else 'holdout' if z['id'] in hold else 'val' if z['id'] in val else 'train'
        z['seen_in_prior_freight_temporal_diagnostic']=z['id'] in prior_pressure
    eligible.sort(key=lambda g:key('m6-actor-file-order|'+g['id']))
    for i,z in enumerate(eligible):z['gid']=i
    doc={'version':'master6-actor-public-root/1','master_id':M6,'baseline_maps':1373,'coverage_games':302,'games':eligible,'source_sha256':source,'exposure_sha256':pri_witness,
         'previous_actor_unique_seeds':len(priseeds),'development_unique_seeds':len(devseeds),'root_policy':'Up to 8 uniform samples without replacement from each of four fixed game phases t0..74,75..149,150..224,225..298. Available ACTIVE decision roots only. With all recorded history before each observation, no future fields or teacher label used in root selection. An unrepresentable expert joint root is omitted and counted; no relabeling. All splits selected by game/terrain before fitting. Use only Master6 vs external opponents; publish all reasons for omission and all W/D/L.',
         'label_scope':'Historical public analysis includes every ladder game. Holdout excludes identified prior V3 training, validation, repository original-exam maps, selected auxiliary split and previous 26-map study. Timestamp and per-game separate descriptive exposure retained. Only policy imitation agreement can use these labels as held-out; native play needs other fresh maps.'}
    dest=HERE/'data_split.json';b=(json.dumps(doc,ensure_ascii=False,indent=1)+'\n').encode()
    if dest.exists() and dest.read_bytes()!=b:raise ValueError('existing selection differs')
    dest.write_bytes(b);print(dict(Counter(z['split'] for z in eligible)),flush=True)
if __name__=='__main__':build()

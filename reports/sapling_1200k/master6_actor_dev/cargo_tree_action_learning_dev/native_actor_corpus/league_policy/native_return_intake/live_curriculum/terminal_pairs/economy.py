"""Reconcile both terminal branches against their signed physical orders and the official environment."""
from collections import Counter
import gzip
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import play

NATURAL=HERE.parents[2]/'natural_original'
spec=importlib.util.spec_from_file_location('sapling_signed_cash_economy',NATURAL/'economy.py')
cash=importlib.util.module_from_spec(spec);spec.loader.exec_module(cash)
KINDS={
    'ordinary_deposit':('own_delivery',),
    'tree_ground_income':('ground_rent',),
    'tree_cargo_tax':('cargo_tax',),
    'other_immediate_income':('opponent_worker_delivery_to_this_base','tree_collision_cash','base_collision_cash'),
    'hiring':('hire_collector','hire_planter'),
    'planting':('plant',),
    'tree_seizure':('seize',),
    'rounding':('rounding',),
    'official_end_adjustment':('terminal_scoring_adjustment',)}

def tape(path):
    return [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines() if line]

def values(counter):
    return {key:round(sum(counter.get(k,0) for k in group),6) for key,group in KINDS.items()}

def run_arm(fixture,prefix,suffix):
    seed,seat,terrain=fixture['seed'],fixture['seat'],fixture['board_sha256']
    cfg=cash.config();cash.np.random.seed(seed)
    env=cash.make('carbon',configuration={**cfg,'randomSeed':seed})
    sim=cash.FastCarbon(cfg).reset(seed=seed);sim.cash=cash.CashEvents(sim.cash)
    cash.require(cash.sha(bytes(env.states[0].observation.full_carbon))==terrain,'terrain mismatch')
    net=[Counter(),Counter()]; start=[];root_hash=None; intercept=[Counter(),Counter()]
    for i,row in enumerate(prefix+suffix):
        cash.require(not env.done and i==row['step']==sim.step_count,'wrong ledger step or extra command')
        if i==len(prefix):
            start=list(sim.cash);root_hash=play.board_hash(env,terrain)
        cmd=row['orders'];cash.require(len(cmd)==2 and all(isinstance(c,dict) for c in cmd),'physical seat order missing')
        before=list(sim.cash);sim.cash.events=[]
        sim.step(*cmd);env.step(cmd)
        o=env.states[0].observation
        cash.require(row['board_after_sha256']==play.board_hash(env,terrain),'official board receipt differs')
        cash.require(sim.full_observation()['players']==o.players and sim.full_observation()['trees']==o.trees and sim.carbon==o.full_carbon,
                     'cash event simulator differs from official board')
        flow=cash.cash_flow(sim.cash.events)
        for side in (0,1):
            cash.check_equal(sum(flow[side].values()),sim.cash[side]-before[side],'transition sources and uses')
            if i>=len(prefix):net[side].update(flow[side])
        if i>=len(prefix):
            for uid,owner,cargo,credited,winner in sim.last_collisions:
                if winner==1-owner:
                    intercept[winner]['captured_from_other_worker']+=cargo
                    intercept[owner]['captured_by_other_worker']+=cargo
    if not suffix:
        return dict(official_prefix_length=len(prefix),terminated=bool(env.done))
    if not env.done:
        return dict(terminated=False,root_hash=root_hash,verified_prefix=len(prefix),verified_suffix=len(suffix),
                    cash_at_interruption=list(sim.cash),counts_for_terminal_pair=False)
    scores=[agent.reward for agent in env.states]
    for side in (0,1):
        net[side]['terminal_scoring_adjustment']=scores[side]-sim.cash[side]
        cash.check_equal(start[side]+sum(net[side].values()),scores[side],'root cash plus changes to official score')
    groups=[values(c) for c in net]
    margin=scores[seat]-scores[1-seat]
    cash.check_equal(start[seat]-start[1-seat]+sum(groups[seat].values())-sum(groups[1-seat].values()),margin,'complete grouping')
    return dict(terminated=True,root_hash=root_hash,root_step=len(prefix),verified_prefix=len(prefix),verified_suffix=len(suffix),
                root_cash_by_seat=start,final_cash_by_seat=list(sim.cash),official_scores_by_seat=scores,
                official_margin=round(margin,2),signed_events_by_physical_seat=net,grouped_by_physical_seat=groups,
                transferred_cargo_unrealized_or_deposited_by_seat=intercept)

def compare(f,base):
    seat=f['seat']
    paths={k:base/(k+'.jsonl.gz') for k in ('prefix','A','B')}
    if not paths['prefix'].is_file():
        return dict(seed=f['seed'],seat=seat,status='no_complete_terminal_pair',recorded_files=sorted(p.name for p in base.glob('*.json')))
    for name,p in paths.items():
        if not p.exists():continue
        card=play.read(base/(name+'.json'))
        cash.require(card['turns_sha256']==cash.sha(p.read_bytes()),'different archived '+name+' tape')
    prefix=tape(paths['prefix'])
    arm={k:run_arm(f,prefix,tape(paths[k])) for k in ('A','B') if paths[k].is_file()}
    result=dict(seed=f['seed'],seat=seat,registered_execution_order=f['arm_order'],arms=arm,
                tape_sha256={k:cash.sha(p.read_bytes()) for k,p in paths.items() if p.is_file()})
    if not arm:result['observed_prefix_only']=run_arm(f,prefix,[])
    if any(not arm.get(k,{}).get('terminated') for k in ('A','B')):
        result['status']='no_complete_terminal_pair';return result
    a,b=[arm[k] for k in ('A','B')]
    cash.require(a['root_hash']==b['root_hash'] and a['root_cash_by_seat']==b['root_cash_by_seat'],'forks differ at the boundary')
    delta=[{k:round(b['grouped_by_physical_seat'][s][k]-a['grouped_by_physical_seat'][s][k],6) for k in KINDS} for s in (0,1)]
    contribution={k:round(delta[seat][k]-delta[1-seat][k],6) for k in KINDS}
    off=round(b['official_margin']-a['official_margin'],2)
    cash.check_equal(sum(contribution.values()),off,'official paired margin change')
    ar,br=(tape(paths[k])[0]['orders'] for k in ('A','B'))
    result.update(status='verified_official_terminal_accounts',root_step=len(prefix),same_original_order_at_root=ar[1-seat]==br[1-seat],
                  distinct_student_order_at_root=ar[seat]!=br[seat],delta_B_A_by_physical_seat=delta,
                  contribution_to_focal_margin_B_A=contribution,official_margin_B_minus_A=off)
    return result

def publish(doc):
    names=('普通交仓','树地租','树收敌货税','其他即时入账','招工','种树','夺树','舍入','官方终局计分调整')
    lines=['# 两个训练局面的原版续弈收支','',
        'A 是网络排第一的动作，B 是排第二的动作。下表从两臂共同根开始记账。收支数为 B 减 A，支出按负数入账；分差贡献为我方变化减对方变化。两组都来自已用训练图。']
    for r in doc['fixtures']:
        lines+=['',f"## {r['seed']}，Sapling 席 {r['seat']}",'']
        if r['status']!='verified_official_terminal_accounts':
            lines+=['缺少成对的官方终局，仅保留有收据的前缀或后缀。'];continue
        seat=r['seat'];delta=r['official_margin_B_minus_A']
        for k in ('A','B'):
            a=r['arms'][k];v=a['official_scores_by_seat']
            lines+=[f"- {k} 臂最终 {v[seat]:,.2f} 比 {v[1-seat]:,.2f}，官方分差 {a['official_margin']:+,.2f}。"]
        lines+=['',f"分支位于第 {r['root_step']} 步，B 相对 A 的官方终局分差变化为 **{delta:+,.2f}**。两臂根上原版指令"+('相同。' if r['same_original_order_at_root'] else '不同，无法把两个结局之差仅归于 Sapling 的单条指令。'),'',
                '| 根至终局的现金项 | 我方变化 | 原版变化 | 分差贡献 |','|---|---:|---:|---:|']
        for key,title in zip(KINDS,names):
            z=r['delta_B_A_by_physical_seat']
            lines.append(f"| {title} | {z[seat][key]:+,.2f} | {z[1-seat][key]:+,.2f} | {r['contribution_to_focal_margin_B_A'][key]:+,.2f} |")
    lines+=['','每拍用两个物理席的实际命令推进，独立核对官方整张盘面、ID 计数、终止、现金与积分。双方持货之间的转移可能以后才结算，未加到当拍现金。表中各项复原的是这两条已经发生的轨迹；判断根能否作为训练标签，还需结合独立的原版应答和时钟核验。[逐席原账及源票哈希](economy.json)']
    (HERE/'economy.md').write_text('\n'.join(lines)+'\n')

def main():
    plan=play.read(HERE/'protocol.json')
    doc=dict(protocol_sha256=cash.sha((HERE/'protocol.json').read_bytes()),audit_source_sha256=cash.sha(Path(__file__).read_bytes()),
             money_events_source=cash.sha(Path(cash.__file__).read_bytes()),
             fixtures=[compare(f,HERE/'outputs'/f"{f['seed']}_{f['seat']}") for f in plan['fixtures']])
    verdict=HERE/'settlement.json'
    if verdict.exists():
        doc['independent_settlement_sha256']=cash.sha(verdict.read_bytes())
        roots=play.read(verdict)['roots']
        for r in doc['fixtures']:
            proof=next(q for q in roots if (q['seed'],q['seat'])==(r['seed'],r['seat']))
            r['separate_training_eligibility']=proof['eligible']
            if r['status']=='verified_official_terminal_accounts':
                cash.check_equal(r['official_margin_B_minus_A'],proof['score_margin_difference_B_minus_A'],'existing terminal settlement')
                for k in ('A','B'):
                    cash.require(proof['branches'][k]['official_scores_by_seat']==r['arms'][k]['official_scores_by_seat'],'existing arm score')
    cash.write(HERE/'economy.json',doc);publish(doc)

if __name__=='__main__':main()

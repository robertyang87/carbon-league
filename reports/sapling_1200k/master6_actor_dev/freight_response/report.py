"""Build a per-courier lookback with bank evidence and preserve the two native responses separately."""
from collections import Counter
import sys
from analyze import HERE,ROOT,PLAN,source_cards,read,save,digest

def path_label(c):
    if c['enemy_cell'] is None:return '尚未出现'
    if not c.get('paths'):return '基地/距基地一步'
    guards=c.get('lighter_units',[]);reachable=[r for r in guards if r['best_coverage']>0]
    if not reachable:return '无可达轻工'
    if c['changed_cover_count']:
        return '/'.join('工'+u.split('-')[-1]+' '+a for u,a in c['candidate_caps'][:3])+f"（{c['changed_cover_count']}令）"
    return '可触及部分' if max(r['best_coverage'] for r in reachable)<1 else '可触及、无合格改令'
def source_summary(dep):
    sums=Counter();cells=Counter()
    for t,k,pos,v,*_ in dep['provenance']:
        sums[k]+=v
        if k=='mined':cells[pos]+=v
    where='、'.join(f"{p}格({v:.1f})" for p,v in cells.most_common(3))
    return f"采矿 {sums['mined']:.1f}"+(f"，主要在 {where}" if cells else '')+f"，劫敌 {sums['captured_enemy']:.1f}，接队友货 {sums['allied_transfer']:.1f}，未进入该笔入库的已获货 {dep['gross_acquired']-dep['amount']:.1f}"
def early_table(data):
    lines=['# 16 条真实双网母局最大十步交付窗口','','每场取对方十步内实际正常回库金额的最高窗口，并列取最早。T 指窗口开始前的状态，入库发生在 T+1…T+10。回溯各步数按目标工人的真实载量、到其最近己方基地的距离、我方轻工可达路线记录。','',
    '下文“无可达轻工”限于从该根直接改一令、最多三步内碰到最快回库路径中至少一格的机制；不对更早布防、非最短路线、多个工人联合干预作出判断。记录过的后续收入只用来解释过去的峰值。','']
    short=Counter();n_workers=0;n_dep=0
    for j,g in enumerate(data):
        n_workers+=len(g['workers']);n_dep+=sum(len(x['deposits']) for x in g['workers'].values())
        lines+=[f"## {j+1}. {g['source']} · 地图 {g['seed']} · 我方席 {g['seat']}",'',
                f"T={g['start']} 至 {g['end']}，Master3s2 收到 {g['opponent_banked']:.2f}。在全部 {g['windows_checked']} 个十步窗口中取最大值。",'']
        for uid,r in g['workers'].items():
            dep=r['deposits'];lines+=[f"### {uid} · 合计 {sum(b['amount'] for b in dep):.2f}",'']
            for b in dep:
                if b['gross_acquired']+0.011<b['amount']:raise ValueError('negative spending loss')
                lines.append(f"- t{b['at']} 在 Master3s2 基地 {b['cell']} 格实收 {b['amount']:.2f}。{source_summary(b)}。")
            lines+=['','| T 前步数 | 实际状态 t | 工人格 | 所带碳 | 离己库 | 本方可介入状态 |','|---:|---:|---:|---:|---:|---|']
            for c in r['checkpoints']:
                lab=path_label(c);tech=lab if not c.get('changed_cover_count',0) else '有可改善双路覆盖的改令'
                short[f"{c['lag']}|{tech}"]+=1
                lines.append(f"| {c['lag']} | {c['t']} | {c['enemy_cell'] if c['enemy_cell'] is not None else '—'} | {c['load'] if c['load'] is not None else '—'} | {c['home_dist'] if c['home_dist'] is not None else '—'} | {lab} |")
            lines+=['']
    lines+=['[全部路径、轻工编号和每格采矿/碰撞实账](surges.json.gz)']
    (HERE/'surges.md').write_text('\n'.join(lines)+'\n')
    return dict(physical_trajectories=len(data),distinct_seeds=len({g['seed'] for g in data}),worker_windows=n_workers,deposit_events=n_dep,all_lag_cases=short)
def horizons(name,r,reg):
    import importlib.util
    sys.path.insert(0,str(HERE.parent/'freight_horizon'))
    spec=importlib.util.spec_from_file_location('freight_horizon_target_data',HERE.parent/'freight_horizon/data.py')
    data=importlib.util.module_from_spec(spec);spec.loader.exec_module(data)
    raw=read(ROOT/r['source']);q=read(HERE/name/'verify.json.gz');t=r['t'];s=r['seat'];e=1-s
    result={}
    for arm in ('parent','alternative'):
        sim=data.FastCarbon(raw['game_configuration']).reset(seed=raw['seed']);data.check_cfg(raw['game_configuration'])
        for acts in raw['actions'][:t]:sim.step(*acts)
        uid=r['target_worker'];cargo=sim.workers[uid].carbon
        if abs(cargo-r['enemy_load'])>.0006:raise ValueError('target root cargo identity')
        sim.cash=data.FreightCash(sim.cash);days=[];tail=read(HERE/name/f'live_{arm}_actions.json.gz');checked=q['arms'][arm]
        for j,acts in enumerate(tail):
            ev=data.add_step(sim,acts);assert all(abs(a-b)<.011 for a,b in zip(sim.cash,checked['path'][j]['cash']))
            days.append(ev)
        frac,settled,gross,principal,first=data.target(uid,cargo,e,0,days)
        assert abs(sum(z['amount'] for z in checked['deposits'] if z['worker']==uid and z['seat']==e and z['t']<=t+10)-gross)<.004
        result[arm]=dict(first_10_gross=round(gross,5),first_10_tagged_principal=round(principal,5),first_10_principal_fraction=frac,
            first_10_payment_step=first,remaining_game_gross=checked['focal_enemy_deposits'],
            first_10_fleet_bank=round(sum(z['amount'] for z in checked['deposits'] if z['seat']==e and z['kind']=='own_delivery' and z['t']<=t+10),4))
    out=dict(arms=result,deltas={k:round(result['alternative'][k]-result['parent'][k],5) for k in ('first_10_gross','first_10_tagged_principal','remaining_game_gross','first_10_fleet_bank')},
        target_implementation_sha256=digest(Path(data.__file__).read_bytes()),map=r['source'],at=t,seat=s,target_worker=r['target_worker'])
    save(HERE/name/'ten_step_principal.json',out);return out

def main():
    src=read(HERE/'surges.json.gz');games=src['games'];ids={(k,r['seed'],r['candidate_seat']):(str(p.name),r['raw_sha256'],r['header'] if 'header' in r else None)
            for k,p,r in source_cards()}
    for g in games:
        p,k,_=ids[g['source'],g['seed'],g['seat']]
        if 'full_tape_hash' in g:g['pack_sha256']=g.pop('full_tape_hash')
        g['tape_file']=p;g['action_tape_sha256']=k
    save(HERE/'surges.json.gz',src)
    census=early_table(games);reg=read(HERE/'roots.json');geos=read(HERE/'geometry.json.gz')['maps']
    lines=['# 对手现金跃升前，哪些运货路线真的拦得住','',
       f"逐工对账覆盖 {census['physical_trajectories']} 条双网对原版 Master3s2 的完整轨迹、{census['distinct_seeds']} 张已曝光开发图。每条取最高十步正常回库窗口，合计 {census['worker_windows']} 个工人–窗口、{census['deposit_events']} 笔实际交付。[逐场、逐工、逐格的回溯](surges.md)",'',
       '在新双网 621458920 席 0，t287–297 对手 7 个采集工合交 1,113.95。其中 56 号在 t277 已载 178.88、距库 6 格，t284 已载 246.98、距库 3 格。这三个事前检查点我方都没有轻工能在三步内碰到它的最快回库路线。席 1 的峰值 t264–274 收了 1,051.96；34 号一工送回 515.40，t254 已带 395.12。t259 我方可覆盖若干回库分支，却没有改一令能改善对其所有首步出口的可达性。同一窗口内另一个 48 号最终交付 243.02，t259 还有三条合法改令可改善双路覆盖，t261 已错过。','',
       '## 前向触发与真实接续','','几何与预测只读当前公开状态和先前五步历史，加上我方当步原命令。先找轻载采集工改变第一步后会增加可达回库路线，再用十步预测的回库概率和预期实收筛选。相同预测和覆盖分数按公开的单位、回合和动作标识打破并列，策略不读取地图种子。[冻结判据](protocol.json) · [部署同源选择器](policy.py) · [所有初始机会](geometry.json.gz) · [真正选中根](roots.json)','',
       '| 范围 | 两席联合步 | 敌采集工次 | 货≥60且距2–7 | 有≥2出口 | 有合法改善的工次 / 动作 | 过预测的工次 / 动作 |','|---|---:|---:|---:|---:|---:|---:|']
    pos=lines.index('## 前向触发与真实接续')
    bins=census['all_lag_cases']
    compact=['按最终发生过交付的84个工人–窗口回看，所有时间点都用同一套三步可达与多出口判定。','',
             '| 距峰值窗口开始 | 工人尚未出现 | 无可达轻工 | 有合法改令能改善双路覆盖 |','|---|---:|---:|---:|']
    for lag in (10,5,3,0):
        compact.append(f"| {lag} 步 | {bins.get(str(lag)+'|尚未出现',0)} | {bins.get(str(lag)+'|无可达轻工',0)} | {bins.get(str(lag)+'|有可改善双路覆盖的改令',0)} |")
    compact+=['','其他情况包括已到基地附近、已有部分可达路径、或现有指令已占较好位置。新双网席 1 最重的 515.40 来自该工持续采矿，其中 17 格贡献 75.99，9 格和 31 格各贡献 65.70；这笔货未含碰撞夺货。','']
    lines[pos:pos]=compact
    for n,rr in reg['maps'].items():
        m=rr['seen'];lines.append(f"| {n} | {m['joint_step']} | {m['enemy_collector']} | {m['loaded_distance']} | {m['two_exits']} | {m['target_with_legal_cover_gain']} / {m['actions_with_legal_cover_gain']} | {m.get('target_passing_model',0)} / {m.get('actions_passing_model',0)} |")
    summary={}
    for name,rr in reg['maps'].items():
        r=rr['selected'];role=rr['map'];lines+=['','## '+('新双网开发图' if name=='development' else '先前研究过的跨图诊断'),'']
        if r is None:lines.append('冻结标准下没有可介入的根。');continue
        lines+=[f"图 {role['seed']}，我方席 {r['seat']}、t{r['t']}。我方 {r['worker']} 在 {r['own_cell']} 格、载 {r['own_load']:.2f}，{r['old']} 改为 {r['alternative']} 去 {r['landing']}。敌方 {r['target_worker']} 载 {r['enemy_load']:.2f}、距其基地 {r['enemy_distance']} 格。十步回库预测概率 {r['model']['p_deposit_10']:.3f}、当前载货预期兑现 {r['model']['expected_root_cash_10']:.2f}，改令后最快路线可达覆盖从 {r['coverage_before']:.3f} 至 {r['coverage_after']:.3f}。",'']
        from analyze import FastCarbon,home
        raw=read(ROOT/r['source']);sim=FastCarbon(raw['game_configuration']).reset(seed=raw['seed']);trace={}
        for t,joint in enumerate(raw['actions'][:r['t']+1]):
            if t in (r['t']-10,r['t']-5,r['t']-3,r['t']):
                obs=sim.obs(r['seat']);own,enemy=obs['players'][r['seat']],obs['players'][1-r['seat']]
                tw=enemy[2].get(r['target_worker']);iw=own[2].get(r['worker'])
                trace[t]=dict(own_cash=own[0],enemy_cash=enemy[0],target_cell=tw[0] if tw else None,target_cargo=tw[1] if tw else None,target_home=home(tw[0],enemy[1].values()) if tw else None,our_cell=iw[0] if iw else None,our_cargo=iw[1] if iw else None)
            if t<r['t']:sim.step(*joint)
        lines+=['| 改令前时刻 | 目标格/货量/距库 | 我方轻工格/货量 | 我方现金/对方现金 |','|---|---|---|---|']
        fmt=lambda v:'—' if v is None else f'{v:.2f}' if isinstance(v,float) else str(v)
        for t,row in trace.items():
            lines.append('| '+str(t)+' | '+' / '.join(fmt(row[k]) for k in ('target_cell','target_cargo','target_home'))+' | '+' / '.join(fmt(row[k]) for k in ('our_cell','our_cargo'))+' | '+f"{row['own_cash']:.2f} / {row['enemy_cash']:.2f} |")
        lines+=['']
        vf=HERE/name/'verify.json.gz'
        if not vf.exists():lines.append('两臂原版对手响应及独立终局核算正在进行。');continue
        v=read(vf);a,b=[v['arms'][k] for k in ('parent','alternative')];op=1-r['seat']
        bank10=horizons(name,r,reg)
        lines+=['| 根指令 | 官方终局分差 | 目标工未来10步总入库 | 其中根时所载货兑现 | 目标工余下整局入库 | 对手全队余局入库 | 我方全队余局入库 | 实搜/后续调用 |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for k,arm,akey in [('原令',a,'parent'),('改令',b,'alternative')]:
            z=bank10['arms'][akey];lines+=[f"| {k} | {arm['margin']:+.2f} | {z['first_10_gross']:.2f} | {z['first_10_tagged_principal']:.2f} | {arm['focal_enemy_deposits']:.2f} | {arm['gross'][op]['bank']:.2f} | {arm['gross'][r['seat']]['bank']:.2f} | {arm['searched']}/{arm['search_calls']} |"]
        x=bank10['deltas']
        lines+=['',f"未来十步，目标工实收入库改变 {x['first_10_gross']:+.2f}；其中出手时已经持有的货最终兑现改变 {x['first_10_tagged_principal']:+.2f}。同期对手全队入库改变 {x['first_10_fleet_bank']:+.2f}。[与预测器相同的十步本金记账](./{name}/ten_step_principal.json)"]
        lines+=['',f"目标工实收改变 {v['delta_target_banked']:+.2f}，整局分差改变 {v['delta_margin']:+.2f}，ΔU {v['delta_u']:+.4f}。目标工结算数字从根到终局，含新采到的货；模型的十步预期只针对起点已有的货。两队的净现金分差变动里，常规回库 {v['sources']['bank']:+.2f}，树息 {v['sources']['tree']:+.2f}，招工 {v['sources']['recruit']:+.2f}，种树 {v['sources']['plant']:+.2f}。",'',
        f"[真实源回放]({Path('../../')/role['source']/'official_replay.json'}) · [逐步现金、目标工去向、本人每笔存款和验证](./{name}/verify.json.gz) · [原令动作带](./{name}/live_parent_actions.json.gz) · [改令动作带](./{name}/live_alternative_actions.json.gz)"]
        lines+=['','实际联合推进后，两臂头五步的目标和拦截工去向如下。只列物理格，— 表示工人已退出。','','| 状态 | 原令 敌工/我工 | 改令 敌工/我工 |','|---|---|---|']
        def cell(x):return str(x['cell']) if x else '—'
        for orig,change in zip(a['path'][:5],b['path'][:5]):
            lines.append(f"| t{orig['at']} | {cell(orig['enemy_target'])} / {cell(orig['our_target'])} | {cell(change['enemy_target'])} / {cell(change['our_target'])} |")
        summary[name]={**{k:v[k] for k in ('delta_target_banked','delta_margin','delta_u','sources')},'ten_step':bank10,'observed_root_history':trace}
    lines+=['','第二图 621455920 曾用于双网运货及一步预测研究，本轮只对新的分叉拦截标签保留；它还使用上一代双网。两图各只设一组新随机流，原版考官受实际时钟预算影响。几何覆盖表示有机会在某个分支位置相遇，同时拦住几条路线或给真实胜率带来收益仍需独立实战验证。']
    if len(summary)==2 and all(v['delta_margin']<0 for v in summary.values()):
        dv=summary['development'];hv=summary['historical_holdout']
        lines[2:2]=[f"**这条拦截改令未通过两图诊断。** 新双网开发根终局分差改变 {dv['delta_margin']:+.2f}，历史图改变 {hv['delta_margin']:+.2f}。两次实际测试都该保留原动作。这两条记录可以作为改动作的负标签，独立的十步回款预测质量仍由其留出集衡量。",'']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n')
    save(HERE/'summary.json',dict(coverage=census,model_sha256=reg['model_sha256'],outcomes=summary))
if __name__=='__main__':
    from pathlib import Path
    main()

"""Derive paired economic response tables and dominant post-root changes from independently certified action tapes."""
from run import D,OLD,ROOT,arms,read,sha,seal,require


def q(v):return f'{v:+,.2f}'


def focal(p):
    return f"{p['cell']}（{p['cargo']:.1f}）" if p else '已阵亡'


def main():
    x=read(D/'results.json');rows=[]
    header=['# 早段封路旧极端根换随机流复测','',
            '从四个已经公开结果的旧开发局面，选出终局分差改善最大和恶化最大的一根。每根保留历史对照，再用一条事前冻结的新原版续弈随机流成对重测。四个新分支全部计入。两根各两次，既有结果参与了选样，不能据此估计总体封路效果或胜率。[选根与预算](protocol.json) · [原四根](../native_response/RESULTS.md)','',
            '|图 / 席 / 步|历史分差变化|新流父代分差|新流封路分差|新流变化|方向改变|',
            '|---|---:|---:|---:|---:|---|']
    details=[]
    for k in x['cases']:
        tag=k['case'];before=read(OLD/tag/'verified.json.gz');after=read(D/tag/'verified.json.gz')
        header.append(f"|{k['seed']} / {k['seat']} / t{k['t']}|{q(k['old_metrics']['delta_margin'])}|{q(k['new_arms']['parent']['margin'])}|{q(k['new_arms']['alternative']['margin'])}|{q(k['new_metrics']['delta_margin'])}|{'是' if k['sign_flip'] else '否'}|")
        details.extend(['',f"## 图 {k['seed']} 席 {k['seat']}，t{k['t']}",'',
                        '每列比较原令与封路。格后括号是当时工人随身货量；交付和采集按各臂从共同根起累计。','',
                        '|流、步数|敌目标工位置、货量（原 → 改）|敌目标工入库变化|己方焦点工位置、货量（原 → 改）|己全队入库变化|敌全队入库变化|己全队净新增采集变化|',
                        '|---|---|---:|---|---:|---:|---:|'])
        seat=k['seat']
        totals={}
        for name,tr in [('历史',before),('新流',after)]:
            p,a=(tr['arms'][n] for n in arms)
            flows={}
            for key in p['terminal_by_source'][0]:
                flows[key]=round((a['terminal_by_source'][seat][key]-p['terminal_by_source'][seat][key])-(a['terminal_by_source'][1-seat][key]-p['terminal_by_source'][1-seat][key]),4)
            require(abs(sum(flows.values())-tr['metrics']['delta_margin'])<.04,'full-game gross decomposition differs from paired official margin')
            totals[name]=flows
            for h in (3,5,10):
                hp=p['horizons'][str(h)];ha=a['horizons'][str(h)]
                diff=lambda metric:q(ha[metric]-hp[metric])
                details.append(f"|{name} {h}|{focal(hp['target_at_end'])} → {focal(ha['target_at_end'])}|{diff('target_banked')}|{focal(hp['our_at_end'])} → {focal(ha['our_at_end'])}|{diff('own_team_banked')}|{diff('enemy_team_banked')}|{diff('own_team_actual_harvest')}|")
        def fmt(z):return '、'.join(f'{label} {q(z[key])}' for key,label in [('bank','交货'),('tree','树息'),('recruit','招工'),('plant','种植'),('seize','夺树')])
        details+=['',f"初始根的对手命令保持相同。历史和新流的对手全队命令首次分化分别在 t{k['first_different_opponent_order_old']}、t{k['first_different_opponent_order_new']}。从同根往后的实际终局分差，历史流中的现金来源为{fmt(totals['历史'])}；新流为{fmt(totals['新流'])}。各项均是己方相对对手的增量，树与交货可随后续双方改道出现大幅差异。[两臂官方终局、逐笔交货与两名工人前十步]({tag}/verified.json.gz)"]
        rows.append({'case':tag,'new_source_sha256':k['new_root_sha256'],'old_source_sha256':k['old_root_sha256'],'net_cash_source_deltas':totals})
    count=sum(k['sign_flip'] for k in x['cases'])
    if count==0:
        conclusion='两根换流后均保持原方向。每根仅两次，且按历史极值选取，收益幅度与整局泛化仍须分开判断。'
    else:
        conclusion=f'两根中有 {count} 根换流后改变方向。把此前单个分支的四位数终局变化直接当作动作标签，会把搜索和后续局势的波动一并学进去。'
    header[2:2]=['**'+conclusion+'**','']
    details+=['','## 重放与成本','',
              f"新流完整原版根后搜索合计 {x['original_after_root_search_calls']} 次；按配对顺序，两核并跑墙钟为 "+'、'.join(f"{z['elapsed_pair_seconds']:.2f} 秒" for z in x['cases'])+"。原令/改令共用该根的初始原版状态、温度 1、已提交的同时对手根命令和原搜索暖机成本。双方随后按各自最新观测出招，原版计时搜索按真实墙钟调整。两套根在两个固定 CPU 间交换原/改臂位置。新分支每 25 次根后搜索保存断点。四条新轨迹各从原始开局经官方规则重放到终局，独立快模拟核现金、双方 3/5/10 步交付、焦点工轨迹及己方每条命令。复核可在本目录执行下列命令。[工位收据](court.json) · [任务身份](execution.json) · [研究结果](results.json) · [现金来源分解](cash_sources.json)",'',
              '    OPENBLAS_NUM_THREADS=1 python run.py verify','    python report.py','']
    seal(D/'cash_sources.json',{'results_sha256':sha(D/'results.json'),'report_code_sha256':sha(D/'report.py'),'cases':rows})
    text='\n'.join(header+details)
    path=D/'README.md'
    if path.exists():require(path.read_text()==text,'published narrative differs')
    else:path.write_text(text)

if __name__=='__main__':main()

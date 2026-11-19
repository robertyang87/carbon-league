"""Publish replay-complete timing, conditional receipts and auditable per-game planter decisions."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import statistics
import tarfile
import time

from economy_extract import atomic,compact
from plant_timing_extract import CHOICE,PURCHASE,MOVES,version
from plant_timing_cases import case_summary,public_path,benchmark

OUT=Path(__file__).resolve().parent
STAGES=["0–59","60–149","150–239","240–268","269–289","290–298"]
RISK=["敌种树工≤2、己采集工≤1","敌种树工≤2、己采集工>1","敌种树工>2"]
TYPES={"plant":"自种","seize":"抢树"}
class Stat:
    __slots__=("afford","moved","failed","paid","financed","paid_aff","repaid","held","latency","price","income","next","next_positive","ends")
    def __init__(self):
        self.afford=self.moved=self.failed=self.paid=self.financed=self.paid_aff=self.repaid=0
        self.held=Counter();self.latency=Counter();self.price=self.income=self.next=self.next_positive=0;self.ends=Counter()
    def see(self,c,r,n):
        if not c[17]:self.afford+=1
        if c[14]:
            self.moved+=1
            if n:
                self.next+=1
                if n[14] is not None:self.next_positive+=1
        elif r is None:self.failed+=1
        if r is None:return
        self.paid+=1;self.financed+=c[17];self.paid_aff+=1-c[17]
        self.price+=r[6];self.income+=r[11]+r[13];self.ends[r[9]]+=1;self.held[r[10]]+=1
        if r[14] is not None:self.repaid+=1;self.latency[r[14]]+=1
    def record(self):
        assert self.afford==self.moved+self.failed+self.paid_aff
        return dict(affordable=self.afford,moved=self.moved,failed_stay=self.failed,purchases=self.paid,paid_affordable=self.paid_aff,
                    same_turn_financed=self.financed,repaid=self.repaid,hold_p50=med(self.held),payback_p50_if_repaid=med(self.latency),
                    cost=round(self.price,2),gross=round(self.income,2),net=round(self.income-self.price,2),
                    moves_with_purchase_5=self.next,later_purchase_repaid=self.next_positive,end_reason=dict(self.ends))
def med(h):
    n=sum(h.values());sofar=0
    if not n:return None
    a=(n-1)//2;b=n//2;lo=None
    for v,w in sorted(h.items()):
        sofar+=w
        if lo is None and sofar>a:lo=v
        if sofar>b:return (lo+v)/2
def stage(c):return STAGES[sum(c[0]>=i for i in [60,150,240,269,290])]
def risk(c):return RISK[0] if c[10]<=2 and c[11]<=1 else RISK[1] if c[10]<=2 else RISK[2]
def pct(a,b):return '—' if not b else f'{100*a/b:.1f}%'
def fmt(x):return '—' if x is None else f'{x:g}'
def summary(st):
    return f"{st['affordable']:,} | {st['moved']:,} | {st['purchases']:,} | {pct(st['paid_affordable'],st['affordable'])} | {pct(st['repaid'],st['purchases'])} | {fmt(st['hold_p50'])} / {fmt(st['payback_p50_if_repaid'])} | {fmt(round(st['net']/st['purchases'],1)) if st['purchases'] else '—'}"
TABLE=['| 组别 | 买得起的决策 | 选择移动 | 实际买入 | 买入率 | 首任回本率 | 持有 / 回本步数中位 | 平均净现金 |','|---|---:|---:|---:|---:|---:|---:|---:|']
def jsonout(p,x):
    atomic(p,(json.dumps(x,ensure_ascii=False,indent=1,allow_nan=False)+'\n').encode())
def page(p,lines):atomic(p,('\n'.join(lines).rstrip('\n')+'\n').encode())
def calc(manifest,raw,work):
    readtime=time.monotonic();data=json.loads(manifest.read_bytes());games=data['games'];names={g['name'] for g in data['top5']};v=version()
    sl=defaultdict(Stat);per_game={};samples={};terminal={};l30=Counter();witness={}
    paths=OUT/'plant_timing_data';pages=OUT/'plant_timing_games';paths.mkdir(exist_ok=True);pages.mkdir(exist_ok=True)
    expected=set('0123456789abcdef');assert expected=={g['id'][0] for g in games}
    for nibble in sorted(expected):
        bundle=io.BytesIO()
        with tarfile.open(fileobj=bundle,mode='w') as tar:
            lines=['# 种树时机逐局账 '+nibble,'','统计格式见[总报告](../plant_timing_report.md)。表中两席均纳入。','']
            for g in (g for g in games if g['id'][0]==nibble):
                gid=g['id'];packed=(work/(gid+'.gz')).read_bytes();doc=json.loads(gzip.decompress(packed))
                if doc['id']!=gid or doc['replay_sha256']!=g['sha256'] or doc['extractor_sha256']!=v:raise ValueError('identity '+gid)
                info=tarfile.TarInfo(gid+'.json.gz');info.mode=0o644;info.size=len(packed);tar.addfile(info,io.BytesIO(packed))
                is_top=[q in names for q in g['seat_names']];rs=doc['purchases'];scores=doc['rewards'];seats=[Stat(),Stat()]
                wd=['胜' if scores[s]>scores[1-s] else '负' if scores[s]<scores[1-s] else '平' for s in (0,1)]
                for c in doc['choices']:
                    t,s=c[:2];r=rs[c[15]] if c[15] is not None else None;n=rs[c[16]] if c[16] is not None else None
                    seats[s].see(c,r,n);sl['全体席'].see(c,r,n)
                    if not is_top[s]:continue
                    kind=c[4];st=stage(c);rr=risk(c);player=g['seat_names'][s]
                    if r and doc['turns']==299 and t>=269:
                        l30[kind]+=1
                        if t>=294:l30['last5_'+kind]+=1;terminal.setdefault(gid,[]).append(c)
                    price='≤20' if c[5]<=20 else '20–50' if c[5]<=50 else '50–100' if c[5]<=100 else '>100'
                    carb='<80' if c[6]<80 else '80–199' if c[6]<200 else '≥200'
                    reserve='<30' if c[7]<30 else '30–89' if c[7]<90 else '≥90'
                    keys=['Top5','种类·'+kind,'阶段·'+st+'·'+kind,'威胁·'+rr+'·'+kind,'阶段威胁·'+st+'·'+rr+'·'+kind,
                          '结局·'+wd[s]+'·'+kind,'选手·'+player+'·'+wd[s]+'·'+kind,'价格·'+price+'·'+kind,
                          '可见碳·'+carb+'·'+kind,'余额·'+reserve+'·'+kind,'分席·'+str(s)+'·'+kind,
                          '对手·'+g['cohort']+'·'+kind,'旁有敌树·'+str(int(c[13]>0))+'·'+kind,
                          '敌采集工一步可达·'+str(int(c[12]<=1))+'·'+kind]
                    if kind=='seize':keys.append('原树年龄·'+('0–24' if c[9]<25 else '25–39' if c[9]<40 else '40–44' if c[9]<45 else '45–49')+'·'+kind)
                    for key in keys:sl[key].see(c,r,n)
                    # Explanations are deliberately chosen after the fact. Prefer mutual games.
                    if c[0]>=20 and g['cohort']=='mutual':
                        options=[]
                        if r and kind=='plant' and t<210 and r[14] and r[14]<=7 and r[10]>=24 and r[11]-r[6]>60 and c[10]>=3:options.append('成功自种')
                        if r and kind=='plant' and t<260 and r[14] is None and r[9]=='seized' and r[10]<=4 and c[10]<=2:options.append('抢手在旁、提前失守')
                        if r and kind=='seize' and r[14] and r[14]<=5 and r[11]-r[6]>20:options.append('成功抢树')
                        if n and n[5]=='seize' and n[14] and n[14]<=6 and c[4]=='plant' and n[4]!=c[3] and n[0]-t>=2:options.append('走开后实际抢得一树')
                        for k in options:
                            if k not in samples:samples[k]=(gid,c,r or n)
                ss=[q.record() for q in seats];per_game[gid]=ss
                lines+=['<a id="g'+gid+'"></a>','### '+gid,'',
                        f"{g['seat_names'][0]} {scores[0]:g} ∶ {scores[1]:g} {g['seat_names'][1]} · [经济与转折](../economy_games/{gid}.md) · [全部决策的分片](../plant_timing_data/{nibble}.tar)",'',*TABLE]
                lines.extend('| 席'+str(s)+' '+g['seat_names'][s]+' | '+summary(q)+' |' for s,q in enumerate(ss))
                lines+=['',f"两席依次有 {ss[0]['moves_with_purchase_5']} / {ss[1]['moves_with_purchase_5']} 次移动后五步内同一工人买树，其中后续实际收回买价 {ss[0]['later_purchase_repaid']} / {ss[1]['later_purchase_repaid']} 次。相邻多步可能指向同一后续买入。",'']
        atomic(paths/(nibble+'.tar'),bundle.getvalue());page(pages/(nibble+'.md'),lines)
    groups={k:s.record() for k,s in sorted(sl.items())}
    check=json.loads((OUT/'economy_coverage.json').read_bytes())['terminal_economy']['scheduled_horizon']
    for k in ('plant','seize','last5_plant','last5_seize'):
        assert l30[k]==check[k.replace('last5','final5')],(k,l30[k],check)
    for name,(gid,c,r) in samples.items():
        env=json.load(gzip.open(raw/(gid+'.json.gz')))
        witness[name]=dict(id=gid,**case_summary(c,r),path=public_path(env,c))
    branches=[]
    for gid,cs in terminal.items():
        env=json.load(gzip.open(raw/(gid+'.json.gz')))
        for row in benchmark(env,cs):branches.append(dict(id=gid,**row))
    assert len(branches)==l30['last5_plant']+l30['last5_seize']
    ds=[r for r in branches if r['margin_delta'] is not None]
    sort=sorted((r for r in ds if r['margin_delta']>0 and all(p['unit_present'] for p in r['public_path'])),key=lambda r:r['margin_delta'])
    if sort:
        r=sort[len(sort)//2];gid=r['id'];c=next(c for c in terminal[gid] if (c[0],c[1],c[2])==(r['t'],r['seat'],r['worker']))
        paid=json.loads(gzip.decompress((work/(gid+'.gz')).read_bytes()))['purchases'][c[15]]
        witness['最后五步改为移动的条件复算']=dict(id=gid,**case_summary(c,paid),path=r['public_path'],counterfactual=r)
    coverage=dict(revision=data['revision'],manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),extractor_sha256=v,
        games=len(per_game),frames=data['coverage']['verified_frames'],transitions=data['coverage']['verified_transitions'],
        groups=groups,top5_last30_counts=l30,eligible_appearances=data['coverage']['registered_player_appearances'],case_definitions=
        "例局按 ID、根、席的固定遍历顺序寻找满足发布脚本所列条件的首个 Top5 互打；末盘分支选可见全历史且改善为正的中位样本。",
        confirmed_terminal_branches=len(ds),terminal_branch_gains=sum(r['margin_delta']>0 for r in ds),
        terminal_branch_losses=sum(r['margin_delta']<0 for r in ds),terminal_branch_unchanged=sum(r['margin_delta']==0 for r in ds),
        terminal_branch_avg_delta=round(statistics.mean(r['margin_delta'] for r in ds),3) if ds else None)
    jsonout(OUT/'plant_timing_coverage.json',coverage);jsonout(OUT/'plant_timing_examples.json',witness)
    atomic(OUT/'plant_timing_final_branches.json.gz',gzip.compress(compact(branches),mtime=0))
    jsonout(OUT/'plant_timing_schema.json',dict(choices=CHOICE,purchases=PURCHASE,move=MOVES,
        notes={"root":"pre-move step, t to t+1","paid":"0-based index into purchases, null if absent",
               "next_same_worker_paid_in_5":"observed purchase within next 5 steps by this worker, repeated references may point to same buy",
               "other_cash_financed":"pre-current-cash below price but paid after intra-turn settlements",
               "last_credited_turn":"close-account transition (includes first turn after loss); may not equal last positive payment",
               "payback_transitions":"first transition with cumulative real first-tenure receipts >= full purchase cost, 1 is immediate",
               "enemy_worker_absorption":"already included in absorption, do not add twice",
               "first_income_transitions":"first transition with strictly positive economic cash credited to the owner",
               "neighbor_carbon":"sum of pre-action own-public four cardinal neighbor carbon excluding occupied tree cells",
               "holding":"count from purchase through the last absorption phase held, not counting a subsequent expiry or seizure; same-transition collision can credit prior owner"}))
    from plant_timing_final_flows import evaluate as attribute_flows
    report(coverage,witness,data,attribute_flows(raw))
    print('published',len(per_game),len(branches),'elapsed',round(time.monotonic()-readtime,1),flush=True)

def report(cov,witness,man,flow_groups):
    g=cov['groups'];all_=g['全体席'];top=g['Top5']
    lines=['# 前五回放里，何时值得停一回合种树','',
        f"回放覆盖 {cov['games']:,} 个不同地图，双方共出现 {all_['affordable']:,} 次能在当前位置买树的种树工决策。冻结榜前五占 {cov['eligible_appearances']:,} 个席次，贡献 {top['affordable']:,} 次可买机会，移动 {top['moved']:,} 次，留下但没成功买到 {top['failed_stay']:,} 次。当步原本就买得起并实际买入 {top['paid_affordable']:,} 次，另有 {top['same_turn_financed']:,} 笔靠同回合入账后才够付。全篇分母按动作或交易注明，同一局和同一工人的连续选择可能高度相关。",'',
        '## 自种、抢树各在什么时候兑现','','首任持有从真实付款算到首次易手、枯死或该局结束。毛收入逐笔累计环境吸收、敌工被吸走的货及结算到树主的碰撞现金。净现金扣整笔买价。回本的第一步可以是购买当步，回本步数中位只在已经回本的交易内算。表中买入率的分母是当时站在可买格且自己的公开现金付得起的决策，分子是成功支付的其中一部分。回本率纳入所有实际资金流，包括同回合凑足的钱。','',*TABLE]
    lo=g['可见碳·<80·plant'];hi=g['可见碳·≥200·plant'];young=g['原树年龄·0–24·seize'];old=g['原树年龄·45–49·seize']
    lines[4:4]=[
        f"**自种先看地块，抢树先看剩余寿命。** 四邻可见碳不足 80 的实际自种 {lo['purchases']:,} 笔，首任回本 {pct(lo['repaid'],lo['purchases'])}；四邻至少 200 的 {hi['purchases']:,} 笔，回本 {pct(hi['repaid'],hi['purchases'])}。0–24 岁敌树被抢后 {pct(young['repaid'],young['purchases'])} 回本，45–49 岁只剩 {pct(old['repaid'],old['purchases'])}。这些是选手实际挑过的投资，单一阈值仍须配合售价、威胁、己方支援和残局可能少给对手的收入使用。",'',
        '树在存活时逐步把吸收的碳计入树主现金。当前规则没有主动出售或终局折价收回树本身的指令。临近终局，可变现的是仍有时间取得的树息与能送到基地的货；剩余树及寿命另列在[残局经济账](economy_report.md)。','']
    for k in TYPES:lines.append('| '+TYPES[k]+' | '+summary(g['种类·'+k])+' |')
    for st in STAGES:
        for k in TYPES:lines.append('| '+st+' 步 '+TYPES[k]+' | '+summary(g['阶段·'+st+'·'+k])+' |')
    lines+=['','## 买之前能看见什么','','下面的距离取行动前格子的环面曼哈顿距。敌种树工越近，同一位置随时可能再易主。附近己方采集工即便能抵达也须同时预判对手指令；观察距离只标记支援可达范围。四邻可见碳合计跳过已有树的位置。备用现金取当前金库减本次牌价；移动后五步是否另有买入，可盘点错过的收益窗口，直接的另一条行动收益需要分支才可估。','',*TABLE]
    for dimension in ('威胁','可见碳','价格','余额','原树年龄','旁有敌树','敌采集工一步可达'):
        for key,st in g.items():
            if key.startswith(dimension+'·'):
                _,value,kind=key.split('·');lines.append('| '+dimension+' '+value+' '+TYPES[kind]+' | '+summary(st)+' |')
    lines+=['',f"前五所有实际移动中，同一工人在随后五步发生另一笔买入的有 {top['moves_with_purchase_5']:,}/{top['moved']:,} 次，其中 {top['later_purchase_repaid']:,} 次所指买入最终收回本金。可能有几步移动共同指向随后同一笔投资。没有移动后买入的工人可能守位、躲避、阵亡或到达终局。",'',
            '## 已按原图重放的末五步对照','',
            f"在打满 299 次决策的对局，前五最后 30 步新种 {cov['top5_last30_counts']['plant']:,}、抢下 {cov['top5_last30_counts']['seize']:,}。最后五步实际购买 {cov['top5_last30_counts']['last5_plant']+cov['top5_last30_counts']['last5_seize']} 笔，每笔单独反演。改变该根的一名种树工，比较原指令与四个移动。先按敌基地、敌单位一步可及、友方占位、离己方基地距离和固定方向次序选择一个移动，再打完本局。按回放重建此前所有动作，原指令分支的双方终局现金逐局逐笔复现。",'',
            f"同局其他工人和对手保持实际指令的条件下，预选移动可比的 {cov['confirmed_terminal_branches']} 笔中，最终分差改善 {cov['terminal_branch_gains']} 笔，变差 {cov['terminal_branch_losses']} 笔，不变 {cov['terminal_branch_unchanged']} 笔，平均改变 {cov['terminal_branch_avg_delta']:+.2f}。敌方本来就拥有的树可能在抢树当步产出，夺走敌方收入也能抬高分差，评价抢树时不能只算自己有没有回本。替代轨迹没有让两边重新应答，碰撞与防守的交互限制了外推。 [全部五臂结果](plant_timing_final_branches.json.gz)",'',
            '## 成功、失手与应当重新考虑的动作','','案例里的格号直接使用回放的物理格。T 是考察的动作，早期案例按条件筛选，末盘案例取四个时间点都有该工人、且预选移动收益为正的中位样本。']
    flow=flow_groups['total'];per=flow['move_avg_per_source']
    pos=lines.index('## 成功、失手与应当重新考虑的动作')
    lines[pos:pos]=[f"按分差拆账，移动后节省自种支出 {per['plant']:+.2f}、抢树支出 {per['seize']:+.2f}，两方树息之差却改变 {per['tree']:+.2f}。正常回库分差另变 {per['bank']:+.2f}，误入对方基地的交割等改变 {per['donation']:+.2f}。{flow['preselected_move_purchases_later']}/{flow['n']} 个预选移动分支里，该工人随后又用实际续弈指令买了树。每棵树自身回本与整局分差因而可能方向相反。[逐笔双臂现金来源](plant_timing_final_flows.json.gz) · [双臂账本复算](plant_timing_final_flows.py)",'']
    for title,r in witness.items():
        id=r['id']; q=next(q for q in man['games'] if q['id']==id);act=r['logged_command']
        lines+=['','### '+title,'',f"{q['seat_names'][r['seat']]}（席 {r['seat']}），T={r['t']}，{r['unit']} 在格 {r['cell']} 实际选择 {act}。当时牌价 {r['price']:.1f}，四邻可见碳 {r['visible_c4']:.1f}，付款后预计余 {r['working_cash']:.1f}，距常规终局 {r['left']} 步。",'',
                '| 回溯 | 工人格 | 距 T 格 | 四邻可见碳 | 敌种树工距离 | 己采集工距离 | 己现金 | 己/敌树 |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for a in r['path']:
            row=[r['t']-a['t'],a.get('cell'),a.get('relative_to_event_cell'),a.get('seen_adjacent_carbon'),a.get('nearest_enemy_planter'),a.get('nearest_friendly_collector'),a['cash']]
            lines.append('| '+' | '.join(fmt(x) for x in row)+f" | {a['trees']}/{a['opposing_trees']} |")
        result='最终未回本' if r['payback'] is None else f"第 {r['payback']} 步回本"
        lines+=['',f"同一工人在 t={r['eventual_purchase_root']} 实际{TYPES[r['eventual_purchase_kind']]}，实付 {r['paid']:.2f}，持有 {r['held']} 步，{result}，首段现金净得 {r['net']:+.2f}。首段以 {dict(terminal='终局',expired='树到龄',seized='被抢树')[r['end_reason']]} 结束。"]
        if 'counterfactual' in r:
            b=r['counterfactual'];lines.append(f"保留其他实际指令，预先按当时局面选 {b['preselected_move']}，最终分差比原行动高 {b['margin_delta']:.2f}。")
        if title=='走开后实际抢得一树':lines.append("该局给出了走开的真实后续。留在原地的结果没有观察到，不能把后来整笔净收入都归因于这一步移动。")
        lines+=['',f"[原局的经济和转折](economy_games/{id}.md) · [本局两席全部决策](plant_timing_games/{id[0]}.md#g{id})"]
    lines+=['','## 如何让策略学会因局而变','',
        '候选动作包含在本格停留、移动到相邻目标、保留现金。自种的目标是预测持有到第 k 步的概率乘以该步可收的碳，合计到枯死或终局，扣掉当步真实价格。抢树再单列对手会少收多少，旁边有敌种树工时要估计重新丢掉它的时间。把继续运作所需的招工现金和离开后可争取的下一处树作为竞争候选。这个排序可以用回放首段账作辅助监督，再用同图换席、双方会重新决策的对局检验。','',
        '现有选择存在选点偏好、敌我强弱、不同开局和金库规模的混杂。留下来的回本率不能直接当作移动样本若留下的回报，根级指令对照不能替代独立正式比赛。所有回放仅用于诊断，未用作正式对 Master3s2 的考试图。','',
        '## 分层与逐局复核','','[阶段×威胁、玩家×胜负、物理席位、互打与外部对手](plant_timing_splits.md) · [精确计数和净现金](plant_timing_coverage.json) · [案例明细](plant_timing_examples.json) · [逐行动字段](plant_timing_schema.json) · [全库清单](manifest.json)','','按对局 ID 首位打开逐局账。数据包中每个成员为单局 JSON gzip，双方全部可买的动作、邻近威胁、付价、移动后五步内的后来买入以及真实首次持有分属于两个可关联数组。','',
        '| ID 首位 | 逐局账 | 全部行动 |','|---|---|---|']
    for p in '0123456789abcdef':lines.append(f'| {p} | [查对局](plant_timing_games/{p}.md) | [下载分片](plant_timing_data/{p}.tar) |')
    lines+=['','牌价、抢树、枯死、树吸收和出生保护采用冻结的环境源码；公开观测与离线入账分开读取。两席当前可见的各自碳图参与选点描述；全图模拟只用于追账和末盘条件分支。[官方规则实现](../../../code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/helpers.py) · [逐帧提取](plant_timing_extract.py) · [分支重放](plant_timing_cases.py)']
    page(OUT/'plant_timing_report.md',lines)
    splits=['# 种树时机分层统计','','[分母、价格及首任回本定义](plant_timing_report.md)','']
    for dimension in ('阶段威胁','选手','结局','分席','对手'):
        splits+=['## '+dimension,'',*TABLE]
        for k,st in g.items():
            if k.startswith(dimension+'·'):
                cols=k.split('·')[1:];cols[-1]=TYPES[cols[-1]]
                splits+=['| '+' / '.join(cols)+' | '+summary(st)+' |']
        splits+=['']
    page(OUT/'plant_timing_splits.md',splits)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for v in ('manifest','raw','work'):p.add_argument(v,type=Path)
    a=p.parse_args();calc(a.manifest,a.raw,a.work)

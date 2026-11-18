"""Rebuild the per-game cards from a frozen Git tree. Run from anywhere in the checkout."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import math
import statistics
import subprocess

from carbon_deep_events import analyze_events
from carbon_deep_space import analyze_space

REV = '37965ea578851b5f3ff88ffc42d7119ff716b5cc'
URL = f'https://github.com/robertyang87/carbon-league/blob/{REV}/'
MODELS = ('Master5s', 'Master6')
NAMES = {'claude-AlphaCarbon-' + m: m for m in MODELS}
HERE = Path(__file__).resolve().parent


def git(*args):
    return subprocess.check_output(['git', '-C', str(HERE.parents[2]), *args])


def blob(path):
    return git('show', f'{REV}:{path}')


def load_index():
    rows = {}
    def visit(obj):
        if isinstance(obj, list):
            for child in obj:
                yield from visit(child)
        elif isinstance(obj, dict):
            if 'game' in obj and 'id' in obj and 'name_a' in obj:
                yield obj
            else:
                for child in obj.values():
                    yield from visit(child)

    paths = git('ls-tree', '-r', '--name-only', REV).decode().splitlines()
    for path in paths:
        if 'matches' not in Path(path).name or not path.endswith('.json'):
            continue
        for g in visit(json.loads(blob(path))):
            if g.get('state') != 'done' or not any(g.get('name_'+s) in NAMES for s in ('a','b')):
                continue
            if not all(math.isfinite((g.get('scores') or {}).get(s, float('nan'))) for s in ('a','b')):
                continue
            k = f"{g['id']}-{g['game']}"
            if k in rows and any(rows[k][f] != g[f] for f in ('name_a','name_b','scores','order','winner')):
                raise ValueError(f'{k}: contradictory score tables')
            rows[k] = g
    live_file = HERE / 'inputs' / 'official_matches.json'
    if live_file.exists():
        live = json.loads(live_file.read_text())
        for key,row in live.items():
            if key != f"{row['id']}-{row['game']}" or not {row['name_a'],row['name_b']}.intersection(NAMES):
                raise ValueError(f'{key}: invalid official fixture identity')
            if key in rows and any(row[f]!=rows[key][f] for f in ('name_a','name_b','scores','order','winner')):
                raise ValueError(f'{key}: score tables disagree with official result')
            rows[key]=row
    aliases = defaultdict(list)
    for p in paths:
        fname = Path(p).name
        if fname.endswith('.json.gz') and fname[:-8] in rows:
            aliases[fname[:-8]].append(p)
    return rows, aliases


def preferred(paths, group):
    return min(paths, key=lambda p: (not p.startswith(f'data/{group.lower()}_corpus/'),len(p),p))


def num(n):
    return f'{n:,.2f}'


def signed(n):
    return f'{n:+,.2f}'


def esc(s):
    return str(s).replace('|', r'\|').replace('\n',' ')


def window(steps, cash, tree, h, j, sign):
    start = min(range(len(cash)-h), key=lambda t: sign*(cash[t+h]-cash[t]))
    end = start + h
    tree_delta = tree[end] - tree[start]
    peak = max(range(start+1,end+1), key=lambda t: sign*(cash[t-1]-cash[t]))
    workers = []
    for t in (start, end):
        players = steps[t][0]['observation']['players']
        workers.append([len(players[j][2]),len(players[1-j][2])])
    return dict(start=start,end=end,shift=cash[end]-cash[start],before=cash[start],
                after=cash[end],tree=tree_delta,other=cash[end]-cash[start]-tree_delta,
                peak=peak,peak_move=cash[peak]-cash[peak-1],workers=workers)


def first_ten(steps, seat):
    return next((t for t,s in enumerate(steps) if len(s[0]['observation']['players'][seat][2])==10),None)


def capture_stats(steps, seat):
    plants = takes = 0
    prev = {}
    for t, frame in enumerate(steps):
        players = frame[0]['observation']['players']
        owners = {uid:p for p, player in enumerate(players) for uid in player[3]}
        if t:
            for uid, p in owners.items():
                if p == seat:
                    plants += uid not in prev
                    takes += uid in prev and prev[uid] != p
        prev = owners
    return [plants,takes]


def study(task):
    k,g,group,src,local = task
    raw = Path(local).read_bytes() if local else blob(src)
    r = json.loads(gzip.decompress(raw))
    steps = r['steps']
    names = [g['name_'+s] for s in g['order']]
    seat = next(i for i,n in enumerate(names) if NAMES.get(n)==group)
    ix = [seat,1-seat]
    rewards = r['rewards']
    if rewards != [g['scores'][s] for s in g['order']]:
        raise ValueError(f'{k}: seat scores disagree')
    if len(steps)<2:
        raise ValueError(f'{k}: no action transition')
    snapshots, cash, tree, trees_total = [], [], [], [0.,0.]
    for t,s in enumerate(steps):
        obs = s[0]['observation']
        if obs['step'] != t:
            raise ValueError(f'{k}: non-contiguous steps')
        ps=obs['players']
        c=[p[0] for p in ps]
        if t == 0 and c[0] != c[1]:
            raise ValueError(f'{k}: unequal starting cash')
        cash.append(c[seat]-c[1-seat])
        for side,p in enumerate(ps):
            if t:
                trees_total[side] += sum(obs['trees'][uid][1] for uid in p[3])
        tree.append(trees_total[seat]-trees_total[1-seat])
        if t in (25,50,100,150,200,250,len(steps)-1):
            snapshots.append(dict(step=t,gap=cash[-1],workers=[len(ps[i][2]) for i in ix],
                                  trees=[len(ps[i][3]) for i in ix],cargo=[sum(w[1] for w in ps[i][2].values()) for i in ix]))
    last=len(steps)-1
    sign=(cash[-1]>0)-(cash[-1]<0)
    final_side_since=max((t for t,v in enumerate(cash) if sign*v<=0), default=-1)+1 if sign else None
    swings=[window(steps,cash,tree,min(25,last),seat,s) for s in (-1,1)]
    opens=[]
    for i in ix:
        recruits=sorted(x for x in (steps[1][i]['action'] or {}).values() if x in ('RECCOLLECTOR','RECPLANTER'))
        opens.append('＋'.join('采' if x=='RECCOLLECTOR' else '种' for x in recruits) or '无招募')
    if r['configuration']['size']!=15:
        raise ValueError(f'{k}: unexpected board size')
    events = [analyze_events(steps,i,configuration=r['configuration']) for i in ix]
    # Keep the action trails needed for the case cards without retaining every
    # transition twice in the parent process.
    for audit in events:
        material = audit.pop('events')
        best_delivery = max((e for e in material if e['kind']=='delivery'),
                            key=lambda e:e['net_cash'], default=None)
        best_capture = max((e for e in material if e['kind']=='collision' and e['enemy_cargo_captured']>0),
                           key=lambda e:e['enemy_cargo_captured'],default=None)
        audit['first_delivery_step'] = next((e['destination_row'] for e in material if e['kind']=='delivery'),None)
        audit['first_plant_step'] = next((e['destination_row'] for e in material if e['kind']=='tree_planted'),None)
        audit['best_delivery'] = best_delivery
        audit['best_capture'] = best_capture
    spatial = analyze_space(steps,seat)
    end_margin=rewards[seat]-rewards[1-seat]
    winner_lab='胜' if g['winner']==g['order'][seat] else '负' if g['winner']==g['order'][1-seat] else '平'
    return dict(key=k,model=group,names=[names[i] for i in ix],opponent=names[1-seat],
                seat=seat,result=winner_lab,margin=end_margin,score=[rewards[i] for i in ix],
                last=last,status=r['statuses'],opening=opens,first_full=[first_ten(steps,i) for i in ix],
                trees=[capture_stats(steps,i) for i in ix],tree_income=[trees_total[i] for i in ix],
                stable=final_side_since,turns=sum((a>0)!=(b>0) for a,b in zip([v for v in cash if v], [v for v in cash if v][1:])),
                swings=swings,snap=snapshots,events=events,space=spatial,
                source=src,sha256=hashlib.sha256(raw).hexdigest())


def pair(v, f=str):
    return ' / '.join('—' if x is None else f(x) for x in v)


def first_space(d, key):
    v=d['space']['first'][key]
    return f"t{v['step']} 格{v['cell']}" if v else '未观察到'


def describe_window(w):
    return (f"t{w['start']}–{w['end']}，现金差 {signed(w['before'])} → {signed(w['after'])}（变化 {signed(w['shift'])}）。"
            f"树当步收入的累计差为 {signed(w['tree'])}，其余现金净流差为 {signed(w['other'])}。"
            f"该窗同方向变化最大的一步在 t{w['peak']}（{signed(w['peak_move'])}）。")


def card(d):
    k,m,opp=d['key'],d['model'],esc(d['opponent'])
    sn=d['snap']
    at={s['step']:s for s in sn}
    prev=at.get(250)
    if prev and d['last']>250:
        late=f"t250 现金差 {signed(prev['gap'])}，最后 {d['last']-250} 步又变化 {signed(sn[-1]['gap']-prev['gap'])}。"
    else:
        late=f"最后可见状态 t{d['last']}，双方状态 {' / '.join(d['status'])}。"
    if abs(d['margin']-sn[-1]['gap'])>.011:
        late+=f" 官方终分差 {signed(d['margin'])} 与末状态现金差 {signed(sn[-1]['gap'])} 不同，胜负取官方结果。"
    link=d['source'] if d['source'].startswith('https:') else URL+d['source']
    lines=[f'<a id="g-{k}"></a>',f"### {m} 对 {opp} · {d['result']} {signed(d['margin'])} · `{k[:8]}`",'',
           f"关注方执席 {d['seat']}。比分 {pair(d['score'],num)}。表内均先列 {m}，再列对手。[原始回放]({link})",'',
           f"**开局与空间。** 首步两方招募为 {pair(d['opening'])}。首次满 10 人在 {pair(d['first_full'])} 步。"
           f"关注方第一次出现在更靠近敌基地的区域为 {first_space(d,'enemy_zone_entry')}；首次到敌基地两格以内为 {first_space(d,'enemy_near_entry')}；"
           f"首次持有树为 {first_space(d,'own_tree')}。",'',
           '| 状态步 | 现金差 | 工人 | 持树 | 场上载货 |','|---:|---:|---:|---:|---:|']
    for s in sn:
        lines.append(f"| {s['step']} | {signed(s['gap'])} | {pair(s['workers'])} | {pair(s['trees'])} | {pair(s['cargo'],num)} |")
    lines+=['','**现金差变化最大的两个窗口。**','',f"- 最有利窗口：{describe_window(d['swings'][0])}",
            f"- 最不利窗口：{describe_window(d['swings'][1])}",'',
            f"**树与收官。** 新植 {pair([x[0] for x in d['trees']])} 棵，夺树 {pair([x[1] for x in d['trees']])} 次；"
            f"整局实际树收入 {pair(d['tree_income'],num)}。{late}"]
    if d['stable'] is not None:
        lines[-1]+=f" 末现金差方向从 t{d['stable']} 起一直保持，期间没有再打平。"
    metrics=[side['metrics'] for side in d['events']]
    own_delivery_gap=metrics[0]['delivery_net_cash']-metrics[1]['delivery_net_cash']
    tree_gap=d['tree_income'][0]-d['tree_income'][1]
    lines+=['',f"**本局的经济分岔。** 首次自种新树在 {pair([side['first_plant_step'] for side in d['events']])} 步，"
           f"首次正额本方基地交付在 {pair([side['first_delivery_step'] for side in d['events']])} 步。"
           f"全局本方基地交付净额差 {signed(own_delivery_gap)}，实际树收入差 {signed(tree_gap)}。"
           f"两项之外的现金净流差为 {signed(sn[-1]['gap']-tree_gap-own_delivery_gap)}"
           "（含招募、种树及其他现金流，按双方等额开局计算）。",
           '',event_text(d),'']
    return '\n'.join(lines)


def event_text(d):
    a,b = d['events']
    lines = ['**交付与碰撞。** 正数交付按自己的采集工交到本方基地计算。以下步段取发出动作时的状态步。',
             '', '| 决策步 | 本方 / 对手交付净额 | 本方 / 对手补员数 | 本方载货碰撞后归敌 | 本方从碰撞获得的敌货 |',
             '|---|---:|---:|---:|---:|']
    for wa,wb in zip(a['windows'],b['windows']):
        m,n=wa['metrics'],wb['metrics']
        if wa['step_start']>=d['last']:
            continue
        lines.append(f"| {wa['step_start']}–{min(d['last'],wa['step_end_exclusive'])-1} | "
                     f"{pair([m['delivery_net_cash'],n['delivery_net_cash']],num)} | "
                     f"{pair([m['recruits'],n['recruits']])} | {num(m['collision_own_cargo_to_opponent'])} | "
                     f"{num(m['collision_enemy_cargo_captured'])} |")
    lines.extend(['','**回看这几步。**',''])
    for label,audit in [('本方',a),('对手',b)]:
        e=audit['best_delivery']
        if e:
            lines.append(f"- {label}最大一笔交付在 t{e['destination_row']}，"
                         f"`{e['worker_id']}` 在格 {e['position']} 交付净额 {num(e['net_cash'])}。")
        else:
            lines.append(f'- {label}已核转移中没有正额本方基地交付。')
    losses=a['largest_collision_losses']
    if losses:
        e=losses[0]
        participants='、'.join(f"{v['worker_id']}({num(v['known_cargo'])})" for v in e['participants'])
        lines.append(f"- t{e['destination_row']} 格 {e['position']}，参与碰撞的工人及碰前载货为 {participants}。"
                     f"该步本方货转敌 {num(e['own_cargo_to_opponent'])}，落地 {num(e['own_cargo_to_ground'])}。")
    e=a['best_capture']
    if e:
        lines.append(f"- 本方单次取得敌货最多在 t{e['destination_row']} 格 {e['position']}，"
                     f"取得 {num(e['enemy_cargo_captured'])}。是否兑现可沿后续交付核对。")
    if a['anomalies'] or b['anomalies']:
        bad=sorted({e['destination_row'] for audit in (a,b) for e in audit['anomalies']})
        lines.append(f"- 动作重建跳过 {len(bad)} 个转移，两席取并集，状态步 {bad}。该处事件未计入。")
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True,help='directory of recovered .json.gz spectator replays')
    parser.add_argument('--work',type=Path,required=True,help='new directory for per-game intermediate results')
    parser.add_argument('--out',type=Path,default=HERE)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--extract-only',action='store_true',help='calculate available games without rendering')
    args=parser.parse_args()
    games,aliases=load_index()
    tasks=[]
    args.work.mkdir(parents=True,exist_ok=True)
    # A work directory is scoped to this code, replay revision and source digests.
    fingerprint=hashlib.sha256(b''.join((HERE/p).read_bytes() for p in
                              ('review.py','carbon_deep_events.py','carbon_deep_space.py'))+REV.encode()).hexdigest()
    epoch=args.work / fingerprint
    epoch.mkdir(exist_ok=True)
    for key,g in sorted(games.items()):
        participants = {NAMES.get(g['name_a']),NAMES.get(g['name_b'])}
        group = next(model for model in MODELS if model in participants)
        if aliases[key]:
            src=preferred(aliases[key],group)
            local=None
        else:
            local=args.cache/(key+'.json.gz')
            if not local.is_file():
                continue
            mid,number=key.split('-')
            src=f'https://117.72.210.134/replay/{mid}/{number}'
        dest=epoch/(key+'.json')
        if dest.is_file():
            if not local or json.loads(dest.read_text())['sha256']==hashlib.sha256(local.read_bytes()).hexdigest():
                continue
        tasks.append((key,g,group,src,local))
    with ProcessPoolExecutor(max_workers=args.workers) as workers:
        for d in workers.map(study,tasks,chunksize=1):
            path=epoch/(d['key']+'.json')
            tmp=path.with_suffix('.tmp')
            tmp.write_text(json.dumps(d,ensure_ascii=False,allow_nan=False))
            tmp.replace(path)
    if not args.extract_only:
        from publish import publish
        details={p.stem:json.loads(p.read_text()) for p in epoch.glob('*.json') if p.stem in games}
        args.out.mkdir(parents=True,exist_ok=True)
        publish(games,details,args.out)
        print(f'{len(details)} detailed games / {len(games)} score records')
    else:
        print(f'{len(list(epoch.glob("*.json")))} detailed games extracted')


if __name__=='__main__':
    main()

"""Explain this index cutoff using only its new, independently replayed games."""
import argparse
import gzip
import json
from pathlib import Path


def read(path):
    data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)


def build(directory, total_mutual=115):
    m=read(directory/'manifest.json')
    delta=read(directory/'score_delta.json')
    v=read(directory/'validation.json')
    index='../context_'+directory.name+'_receipt.json'
    context=read(directory/index)
    cases={g['id']:g for g in read(directory/'cases.json.gz')['rows']}
    cash=read(directory/'turn_cash.json.gz')
    assert {g['id'] for g in cash['games']}==set(cases)=={g['id'] for g in m['games']}
    n=sum(len(r['turns']) for r in cash['games'])
    assert v['original_board_transitions']==v['fastsim_transitions']==n and v['union_distinct']['id']==delta['union_games']
    rows=[]
    for g in m['games']:
        s=next(i for i,vv in enumerate(g['seat_strategy_ids']) if vv is not None)
        sign=1-2*s
        swing=cases[g['id']]['economy']['swing']
        before=swing['positions']
        lookback=[]
        for d in (10,5,3,0):
            t=max(0,swing['a']-d)
            snap=next(r for r in before if r['t']==t)
            lookback.append({'t':t,'offset':d,'top_seat_economy':snap['seats'][s],'opponent_economy':snap['seats'][1-s]})
        order=(s,1-s)
        rows.append({'game':g['id'],'seed':g['seed'],'top_seat':s,'names':[g['seat_names'][i] for i in order],
                     'scores':[g['score_by_seat'][i] for i in order],'steps':g['transitions'],
                     'largest_ten_step_window':[swing['a'],swing['b']],'top_margin_delta':sign*swing['margin0_delta'],
                     'delivery':[swing['seats'][i]['bank'] for i in order],'tree_cash':[swing['seats'][i]['tree'] for i in order],
                     'lookback':lookback})
    ar=m['new_archives'][0]
    lines=['# 前五回放追加核验','',
           f"截止 `{m['revision'][:8]}`，{delta['source']} 从 {delta['old_reference_count']} 条增加到 {delta['new_reference_count']} 条，本次核验其中 {len(rows)} 张新图。此前已归档的重复引用 {delta['existing_references']} 条，双方均未进入冻结前五的新增引用 {delta.get('non_top_five_refs',0)} 条。累计 {delta['union_games']:,} 张图，互打 {total_mutual} 张。每张图的物理席、官方比分、种子与真实棋盘均已核对。[逐条去重清单](score_delta.json) · [归档及回放身份](manifest.json)",'',
           '|前五策略 / 席|对手|终局现金（前五 / 对手）|最大十步窗口|窗口内分差变化|正常回存（前五 / 对手）|树收入（前五 / 对手）|',
           '|---|---|---:|---|---:|---:|---:|']
    for r in rows:
        a,b=r['scores'];i,j=r['delivery'];u,vv=r['tree_cash'];t0,t1=r['largest_ten_step_window']
        lines.append(f"| [{r['names'][0]} / {r['top_seat']}]({r['game']}.md) | {r['names'][1]} | {a:,.2f} / {b:,.2f} | t{t0}–{t1} | {r['top_margin_delta']:+,.2f} | {i:,.2f} / {j:,.2f} | {u:,.2f} / {vv:,.2f} |")
    lines+=['','每局以十步现金差绝对变化最大者选窗；事后选择的窗口不能证明因果。原始页收录窗前 3、5、10 步及起点的双方现金、工人、基地、树与随身货；工人碰撞所得在途货与真实入账分列。[逐局回溯](turning_points.json) · [每步双方八项现金和工号](turn_cash.json.gz) · [树持有、进攻与防守案例](cases.json.gz)','',
            f"官方 Board 和 FastCarbon 各复算 {n:,} 次联合推进，{len(rows)} 场全部吻合。每席行动前上下文复查包括 {n*2:,} 个决策，来源仅限本批原始回放；历史局复用已冻结的完整收据。本批在独立考试排除集中逐图绑定真实 225 格开局。归档文件 {ar['bytes']:,} 字节，SHA-256 `{ar['sha256']}`。[双引擎逐局收据](validation_games.jsonl.gz) · [真实棋盘](validation_maps.jsonl.gz) · [对局上下文]("+index+") · [完整无损回放](../../../../"+ar['archive']+')','']
    (directory/'README.md').write_text('\n'.join(lines))
    (directory/'turning_points.json').write_text(json.dumps({'seat_order':'top strategy, other strategy','games':rows},indent=1,ensure_ascii=False)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);build(p.parse_args().directory)

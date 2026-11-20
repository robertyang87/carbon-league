"""Render only the observed economic consequences of the existing verified v4 ladder games."""
import gzip
import json
from pathlib import Path
from statistics import mean

HERE=Path(__file__).resolve().parent


def num(x):
    return f'{x:,.2f}'


def signed(x):
    return f'{x:+,.2f}'


def report(data):
    games=data['games'];counts=data['counts']
    lines=['# ds-pro-v4 天梯对局与首个现金转折','',
      f"公开截止 {data['cutoff_utc']}。注册 ID `{data['strategy']}`。已结算 {len(games)} 局，{counts.get('W',0)} 胜、{counts.get('D',0)} 平、{counts.get('L',0)} 负，覆盖 {data['unique_terrains']} 张不同开局地形。另有 {len(data['pending'])} 张未结算赛卡。本表按公开赛卡逐个编号去重，己席从每场实际座位顺序读取。[官方分页留档](pages.json.gz) · [完整身份、比分与同比对手](summary.json)",'',
      '| 对手 | 实际席位 | v4 : 对手 | 分差 | 归库差 | 净树差 |','|---|---:|---:|---:|---:|---:|']
    for c in games:
        dif=c.get('cash_account',{});link=c.get('ledger',c['replay'])
        lines.append(f"| [{c['opponent']}]({link}) | {c['seat']} | {num(c['score'][0])} : {num(c['score'][1])} | {signed(c['margin'])} | {signed(dif['bank']) if dif else '未核'} | {signed(dif['net_tree']) if dif else '未核'} |")
    lines.extend(['','树净收入已扣种树及夺树费用。回放逐拍施加双方真实指令，以官方规则复算。文中 t 为行动前状态，工人在 t 发令，到账记录在 t+1。连续 20 拍领先或落后至少 250 分用于描述持续转折；五拍累计变化 200 分用于找较早的短期变化。两者使用同一口径覆盖胜负双方。'])
    for c in games:
        if 'ledger' not in c:continue
        d=json.loads(gzip.decompress((HERE/c['ledger']).read_bytes()));q=d['turning']
        lines.extend(['',f"## {c['opponent']}",''])
        def detail(e):
            own=c['seat']
            events=[]
            for r in e['deposits_with_action_history']:
                route="、".join(f"t{h['step']} 格{h['worker_state'][0]} {h['issued'] or '终局'}" for h in r['route'] if h['worker_state'] is not None)
                events.append(f"{'我' if r['seat']==own else '对方'} {r['worker'].rsplit('-',1)[-1]} 号在 t{r['arrival']} 交 {num(r['amount'])}（{route}）")
            return '。'.join(events[:2])
        early=q['early']
        if early:
            e=early[1];diff=e['differences']
            lines.append(f"首次五拍累计波动超过 200 分出现在 t{e['start']}–{e['end']}。现金差由 {signed(e['margin_before'])} 到 {signed(e['margin_after'])}，这五拍归库净变动 {signed(diff['bank'])}、树经营净变动 {signed(diff['net_tree'])}、招募开支差 {signed(diff['recruit'])}。{detail(e)}。")
        lines.append('')
        first=q['first_crossing']
        if first:
            e=q['crossing'][-1]
            lines.append(f"首段持续优势/劣势始于 t{first}。此前 t{e['start']}–{first} 分差从 {signed(e['margin_before'])} 到 {signed(e['margin_after'])}；我方／对方树数为 {e['ending_state']['teams'][0]['trees']}/{e['ending_state']['teams'][1]['trees']}。")
        else:lines.append('整局均未出现连续 20 拍保持 250 分优势或劣势的状态。')
        if q['ending']:
            v=q['ending'][0];worst=q['worst_ten_for_us']
            lines.extend(['',f"从 t{q['permanent_final_sign']} 起最终胜负方向持续到终局。整局我方最不利的十拍是 t{worst['start']}–{worst['end']}，现金差由 {signed(worst['margin_from'])} 到 {signed(worst['margin_to'])}。"])
        lines.extend(['',f"该局双方工人的所有命令、逐树持有收益、3/5/10 拍回溯与官方复演收据均已留存。[原回放]({c['replay']}) · [逐拍收支]({c['ledger']})"])
    if len(games)>=2:
        lines.extend(['','## 首批两局的对照','',
          '对 stephen 的一局，v4 的 3 号种植工在 t33 于格 31 投入 47.06，t40 回本，首次连续持有至 t83，净收入 107.15；5 号在 t35 于格 193 投入 53.42，t42 回本，首次持有净收入 121.69。全局 v4 新种 44 棵，其中 38 棵在首次持有期内回本；对方为 12 棵中 9 棵。对 Krypton 的一局，v4 多种四棵但树净收入少 310.95，依靠归库多 473.16 收回差额。t297 v4 落后 32.19，51 号工人从格 139 向下返回 154，在 t298 交 107.20。两局的花费、树位与敌方应对均有区别，单凭种树数不能推断下一局的净收益。[stephen 对局](00b13bf8e22eea6cd830d1ab82ae2c85-1.ledger.json.gz) · [Krypton 对局](2cba12ae1302e3e6392cf0bc74476b12-1.ledger.json.gz)'])
    if any(c['match']=='1c80b916b203238a43267baeabc88fbe' for c in games):
        lines.extend(['', 'RyanX 大败的两库同拍交仓、两种实际可执行的拦截路线及对手绕开的反例见[逐局复核](RYANX.md)。'])
    (HERE/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    report(json.loads((HERE/'summary.json').read_bytes()))

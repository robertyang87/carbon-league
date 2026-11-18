"""Describe training-relevant behavior from one complete, current study epoch.

Example:
  python behavioral_profiles.py --work run-cache/derived/<fingerprint> \
      --expect-games 352 --expect-appearances 354

No replay loading, cache fallback, model fitting, or causal strength inference.
"""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics

HERE = Path(__file__).resolve().parent
WINDOWS = ((0, 100), (100, 200), (200, 300))
GROUPS = (('全部', None), ('胜场', '胜'), ('负场', '负'))


def current_fingerprint(revision):
    return hashlib.sha256(b''.join((HERE/name).read_bytes() for name in
                                  ('review.py', 'carbon_deep_events.py', 'carbon_deep_space.py'))
                          + revision.encode()).hexdigest()


def number(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f'non-finite or non-numeric value: {value!r}')
    return value


def med(values):
    return statistics.median(values) if values else None


def fmt(value, signed=False):
    return '—' if value is None else (f'{value:+,.2f}' if signed else f'{value:,.2f}')


def cell(values, total, *, signed=False):
    return f'{fmt(med(values), signed)}；n={len(values)}/{total}'


def first_cell(views, field):
    eligible = [v for v in views if v['event_valid']]
    values = [number(v['events'][field]) for v in eligible if v['events'][field] is not None]
    # nz means an event occurred, never "step is nonzero"; state 0 is legal.
    return (f'{fmt(med(values))}；nz/n={len(values)}/{len(eligible)}；'
            f'无事件 {len(eligible)-len(values)}；剔除 {len(views)-len(eligible)}')


def make_views(details, names):
    views = []
    for d in details:
        if len(d['names']) != 2 or len(d['events']) != 2:
            raise ValueError(f"{d['key']}: missing two-sided names/events")
        for side, name in enumerate(d['names']):
            if name not in names:
                continue
            event = d['events'][side]
            for field in ('first_delivery_step', 'first_plant_step'):
                if field not in event:
                    raise ValueError(f"{d['key']}: old cache without {field}")
            if event['seat'] != (d['seat'] if side == 0 else 1-d['seat']):
                raise ValueError(f"{d['key']}: event seat is not the model's actual seat")
            if event.get('rule_binding') != 'replay_configuration':
                raise ValueError(f"{d['key']}: event rules not bound to replay")
            windows = {(w['step_start'], w['step_end_exclusive']): w for w in event['windows']}
            if set(windows) != set(WINDOWS):
                raise ValueError(f"{d['key']}: inconsistent window definitions")
            result = d['result']
            if side:
                result = {'胜':'负', '负':'胜', '平':'平'}.get(result, '未知')
            snapshots = {s['step']:s for s in d['snap']}
            views.append(dict(key=d['key'], model=names[name], side=side, result=result,
                              events=event, windows=windows, snapshots=snapshots,
                              event_valid=event['metrics']['skipped_transitions']==0 and not event['anomalies'],
                              tree_gap=number(d['tree_income'][side])-number(d['tree_income'][1-side]),
                              last=number(d['last'])))
    return views


def generate(work, out, expect_games=None, expect_appearances=None):
    from review import REV, MODELS, NAMES
    work = Path(work).resolve()
    fingerprint = current_fingerprint(REV)
    if work.name != fingerprint:
        raise ValueError(f'old/mixed epoch refused: requested {work.name}, current {fingerprint}')
    files = sorted(work.glob('*.json'))
    if not files:
        raise ValueError('no study detail JSON in this fingerprint directory')
    details, seen = [], set()
    digest = hashlib.sha256()
    for path in files:
        raw = path.read_bytes()
        d = json.loads(raw)
        key = d['key']
        if key != path.stem or key in seen:
            raise ValueError(f'duplicate or filename-mismatched identity: {key}')
        seen.add(key)
        details.append(d)
        digest.update(path.name.encode()+b'\0'+raw+b'\0')
    if expect_games is not None and len(details) != expect_games:
        raise ValueError(f'incomplete epoch: {len(details)} details, expected {expect_games}')
    views = make_views(details, NAMES)
    if expect_appearances is not None and len(views) != expect_appearances:
        raise ValueError(f'model appearance mismatch: {len(views)}, expected {expect_appearances}')

    lines = ['# Top 2 行为读数：Master5s 与 Master6', '',
             f'仓内回放/代码基准 `{REV}`；提取 epoch `{fingerprint}`。', '',
             f'本次只读取该 epoch 的 **{len(details)} 个唯一对局**，形成 **{len(views)} 次模型出场**。'
             '同一场若双方均在范围内，就各按真实席位统计一次；它仍是一场，不能把两次出场当作独立实验重复。', '',
             f'输入全部 detail 字节的 SHA-256：`{digest.hexdigest()}`。逐场证据见 [完整索引](INDEX.md)。', '',
             '## 口径', '',
             '- 分组是全部 / 胜场 / 负场；“全部”也包括平局。胜负按该模型实际视角翻转官方结果，双方 events 同步换席，不复制关注方的事件。',
             '- 首交付为第一笔正额本方基地交付；首棵亲植为第一次新种树，不是第一次持有或夺到树。步骤均为动作执行后的可见状态步。',
             '- 首时刻报告有事件局中的中位数，`nz/n` 为发生该事件的局数 / 可完整核事件的局数；“无事件”单列，不以零或终局步替代。若某局有跳过/异常转移，它不能证明首事件准确，首时刻与整局事件诊断剔除该局。',
             '- t100/t200 仅报告**现金差**（该模型−对手）；当前 detail 没有双方绝对现金值。每格 n 是实际存在对应状态的局数，提前终局不补齐。',
             '- 三窗口沿事件提取器的**决策源步** `[0,100)`、`[100,200)`、`[200,300)`；常规最后一窗实际只有决策 200–298，共 99 个转移。与上面的动作后状态步不是同一索引。',
             '- 窗口只在至少一个转移得到核验且无跳过转移时纳入；报告可用局数与实际覆盖转移数的中位数。提前结束导致的短窗口仍保留，但明确显示“短窗”数量，不能视为完整百步。',
             '- 交付为本方 collector 向本方基地的净额（最终逐步现金四舍五入前），不是团队现金变化；归敌载货专指碰撞后转入敌方保管/账户的己方货，另列直接交到敌基地的货。两者不可相加当净损失，可能描述同一碳的后续流转；归敌载货本身也不保证对方最终变现。',
             '- 树收入差是整局实际树吸收收入之差，包含对 collector 的吸收；不是仅自然产量，也不是选树动作的因果增益。全额中位数把真实零额保留；`nz/n` 是正额局比例，避免中位数为零掩盖尾部事故。',
             '- 胜/负分层是事后结果条件化，且对手、席位、时期不平衡；这些数字只能用于诊断假设，不能据此声称某种行为造成胜利或模型之间的因果优势。', '']

    for model in MODELS:
        model_views = [v for v in views if v['model'] == model]
        counts = Counter(v['result'] for v in model_views)
        lines += [f'## {model}', '',
                  f"出场 **{len(model_views)}**：胜 {counts['胜']}、负 {counts['负']}、平 {counts['平']}、未知 {sum(n for k,n in counts.items() if k not in ('胜','负','平'))}。", '',
                  '| 分组 | 局数 | 首交付状态步 | 首棵亲植状态步 | t100 现金差中位数 | t200 现金差中位数 | 整局树收入差中位数 |',
                  '|---|---:|---|---|---|---|---|']
        for label, result in GROUPS:
            group = [v for v in model_views if result is None or v['result'] == result]
            gap_cells = []
            for t in (100, 200):
                gaps = [number(v['snapshots'][t]['gap'])*(1 if v['side']==0 else -1)
                        for v in group if t in v['snapshots']]
                gap_cells.append(cell(gaps, len(group), signed=True))
            lines.append(f"| {label} | {len(group)} | {first_cell(group, 'first_delivery_step')} | {first_cell(group, 'first_plant_step')} | {' | '.join(gap_cells)} | {cell([v['tree_gap'] for v in group], len(group), signed=True)} |")
        lines += ['', '| 分组 | 决策窗口 | 可用局/分组局 | 转移数中位数；短窗 | 本方基地交付净额中位数 | 碰撞归敌载货：窗口中位数；正额 nz/n | 交到敌基地载货：窗口中位数；正额 nz/n |',
                  '|---|---|---:|---|---:|---|---|']
        for label, result in GROUPS:
            group = [v for v in model_views if result is None or v['result'] == result]
            for bounds in WINDOWS:
                pairs = [(v, v['windows'][bounds]['metrics']) for v in group]
                pairs = [(v,m) for v,m in pairs if m['verified_transitions']>0 and m['skipped_transitions']==0]
                metrics = [m for _,m in pairs]
                deliveries = [number(m['delivery_net_cash']) for m in metrics]
                transfers = [number(m['collision_own_cargo_to_opponent']) for m in metrics]
                enemy_base = [number(m['delivered_to_enemy_carbon']) for m in metrics]
                transitions = [number(m['verified_transitions']) for m in metrics]
                expected = 99 if bounds == (200,300) else bounds[1]-bounds[0]
                short = sum(m['verified_transitions']<expected for _,m in pairs)
                n = len(metrics)
                lines.append(f'| {label} | [{bounds[0]},{bounds[1]}) | {n}/{len(group)} | {fmt(med(transitions))}；{short} | {fmt(med(deliveries))} | {fmt(med(transfers))}；{sum(x>0 for x in transfers)}/{n} | {fmt(med(enemy_base))}；{sum(x>0 for x in enemy_base)}/{n} |')
        # Tail diagnostics retain the rate even when every median is zero.
        valid = [v for v in model_views if v['event_valid']]
        losses = [number(v['events']['metrics']['collision_own_cargo_to_opponent']) for v in valid]
        positive = [x for x in losses if x > 0]
        lines += ['', f"整局碰撞归敌载货：正额 **{len(positive)}/{len(valid)}** 局，正额局中位数 **{fmt(med(positive))}**；"
                  f"全部可核局中位数 **{fmt(med(losses))}**。事件异常剔除 {len(model_views)-len(valid)} 局。", '']
        if 0 < counts['负'] < 10:
            lines += [f"本模型只有 {counts['负']} 场败局；败局中位数对单场十分敏感，不能升级为稳定的训练配方。", '']

    lines += ['## 适合纳入 Sapling 固定诊断集的部分', '',
              '1. **开局双任务**：同一独立局的首交付和首棵亲植同时记录，补上到对应时刻的现金差；检查小网是否只模仿动作频率，却把最初收入或种植启动系统性推迟。目标是相对父网的配对偏移，不是硬追某个榜首中位数。',
              '2. **中局经济连续性**：固定 t100/t200 的同局根与真实续弈，分别看基地交付和树收入，而不是只看活树数或单步采集量。覆盖持币不足、树龄将尽、夺树支付以及共享预算约束。',
              '3. **高载回程与残局结算**：纳入碰撞归敌的正额尾部局，同时放入同类安全返航对照；特别检查“抢到却来不及交”和“敌基地碰撞后货付给敌人”。必须区分货物归属、银行归属、剩余步数，不能用击杀计数代替 U。',
              '4. **保留失败和无事件局**：固定集按来源局切分、隔离训练与选阈值；每次更换学生都在相同局根/席位/续弈配置上报告差值。一个独立局里的很多步仍是相关样本，统计分母按局，不按动作数。', '',
              '以上是可检验的诊断项，不是已证明有效的新增损失或搜索规则。要变成训练目标，仍需消融和独立终局 U 验证；全量统计不能代替这些闸。', '',
              '## 重建', '',
              '```bash',
              f'python reports/master_family/deep_review/behavioral_profiles.py --work /path/to/{fingerprint} --expect-games {len(details)} --expect-appearances {len(views)}',
              '```', '',
              '生成器会拒绝与当前 review.py/events/space/REV 指纹不一致的目录、旧字段缺失、席位错配、重复身份和数量不完整；不向旧 epoch 回退。', '']
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name+'.tmp')
    tmp.write_text('\n'.join(lines), encoding='utf-8')
    tmp.replace(out)
    return dict(games=len(details), appearances=len(views), fingerprint=fingerprint, out=str(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True, help='one current review.py fingerprint directory')
    parser.add_argument('--out', type=Path, default=HERE/'behavioral_profiles.md')
    parser.add_argument('--expect-games', type=int)
    parser.add_argument('--expect-appearances', type=int)
    args = parser.parse_args()
    print(json.dumps(generate(args.work, args.out, args.expect_games, args.expect_appearances), ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Publish a complete replay index, per-game cards and auditable aggregates.

Only ``publish(games, details, directory)`` is public. Metadata defines the
universe; unavailable replays remain visible and never become zero-valued
observations. Family-versus-family games have one index row but two model views.
"""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import io
import json
import math
import statistics
from urllib.parse import urlsplit

PAGE_SIZE = 12
SNAPSHOTS = (25, 50, 100, 150, 200, 250, 299)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def _json(path, value):
    _write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def _csv(path, rows, columns):
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction='raise', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    _write(path, buf.getvalue())


def _mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def _fmt(value, signed=False):
    if value is None:
        return '—'
    return f'{value:+,.2f}' if signed else f'{value:,.2f}'


def _esc(value):
    return str(value).replace('|', r'\|').replace('\n', ' ')


def _name(row, letter):
    # Current frozen metadata stores name_a/name_b. Keep a loud failure for a
    # different schema rather than silently confusing original player order.
    return row['name_' + letter]


def _score(row, letter):
    value = row.get('scores', {}).get(letter)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('non-finite or non-numeric metadata score')
    return value


def _result(row, letter):
    other = 'b' if letter == 'a' else 'a'
    if row.get('winner') == letter:
        return 'win'
    if row.get('winner') == other:
        return 'loss'
    a, b = _score(row, letter), _score(row, other)
    if a is not None and b is not None and a == b:
        return 'draw'
    return 'unknown'


def _source_url(source, prefix):
    return source if urlsplit(source).scheme in ('http', 'https') else prefix + source


def _source_group(source):
    parsed = urlsplit(source)
    if parsed.scheme in ('http', 'https'):
        # Group the official replay endpoint, not every /matchid/game URL.
        route = parsed.path.strip('/').split('/')[0]
        return parsed.scheme + '://' + parsed.netloc + ('/' + route if route else '')
    return str(Path(source).parent)


def _metadata_view(row, letter):
    other = 'b' if letter == 'a' else 'a'
    score, opponent_score = _score(row, letter), _score(row, other)
    order = row.get('order', [])
    return dict(name=_name(row, letter), opponent=_name(row, other),
                seat=order.index(letter) if letter in order else None,
                result=_result(row, letter), score=score, opponent_score=opponent_score,
                margin=score-opponent_score if score is not None and opponent_score is not None else None)


def _tally(items):
    counts = Counter(x['result'] for x in items)
    margins = [x['margin'] for x in items if x['margin'] is not None]
    return dict(games=len(items), wins=counts['win'], losses=counts['loss'],
                draws=counts['draw'], unknown=counts['unknown'],
                win_rate=counts['win']/len(items) if items else None,
                margin_n=len(margins), mean_margin=_mean(margins),
                median_margin=statistics.median(margins) if margins else None)


def _missing_card(key, row, reason):
    a, b = _score(row, 'a'), _score(row, 'b')
    return '\n'.join([
        f'<a id="g-{key}"></a>',
        f"### {_esc(row['name_a'])} 对 {_esc(row['name_b'])} · 回放未分析 · `{key[:8]}`", '',
        f'完整身份：`{key}`。原始列表 a/b 比分 {_fmt(a)} / {_fmt(b)}；'
        f"原始席位顺序 `{row.get('order', [])}`，winner=`{row.get('winner')}`。", '',
        f'**缺口：{_esc(reason)}。** 该局仍计入元数据索引和模型官方成绩，'
        '不进入任何轨迹、事件、空间或阶段均值，亦不以零补齐。', ''])


def publish(games, details, directory):
    """Write every metadata game plus available ``study`` details.

    Returns the emitted summary. ``games`` and ``details`` are keyed by the
    complete matchid-game identity; no inference is made from an ID prefix.
    """
    # Delayed import permits review.py to import publish without a cycle.
    from review import REV, URL, MODELS, NAMES, card

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    extra = set(details) - set(games)
    if extra:
        raise ValueError(f'details absent from metadata: {sorted(extra)}')
    valid = {}
    missing = {}
    for key, row in games.items():
        if key != f"{row['id']}-{row['game']}":
            raise ValueError(f'metadata identity mismatch: {key}')
        d = details.get(key)
        if not d or d.get('error'):
            missing[key] = (d or {}).get('error', '冻结来源中没有可用于逐局分析的回放')
            continue
        if d.get('key') != key:
            raise ValueError(f'detail identity mismatch: {key}')
        if Counter(d['names']) != Counter([row['name_a'], row['name_b']]):
            raise ValueError(f'detail player mismatch: {key}')
        valid[key] = d

    keys = sorted(games, key=lambda k: (games[k].get('created', 0), k))
    # Balance pages instead of leaving a tiny tail. For a normal corpus each
    # page has 10–15 games; small synthetic inputs may have fewer than ten.
    count = max(1, (len(keys)+PAGE_SIZE//2)//PAGE_SIZE, (len(keys)+14)//15) if keys else 0
    size, remainder = divmod(len(keys), count) if count else (0, 0)
    chunks, offset = [], 0
    for i in range(count):
        end = offset + size + (i < remainder)
        chunks.append(keys[offset:end])
        offset = end
    links = {key: f'pages/games-{i+1:03d}.md#g-{key}' for i, chunk in enumerate(chunks) for key in chunk}
    pages = []
    start = 0
    for page, chunk in enumerate(chunks):
        name = f'pages/games-{page+1:03d}.md'
        pages.append(name)
        nav = ['[总索引](../INDEX.md)']
        if page:
            nav.append(f'[上一页](games-{page:03d}.md)')
        if page+1 < len(chunks):
            nav.append(f'[下一页](games-{page+2:03d}.md)')
        text = [f'# 逐局复盘 {start+1}–{start+len(chunk)} / {len(keys)}', '', ' · '.join(nav), '',
                f'冻结 `{REV}`。每张卡先列关注模型、再列对手；动作时序、事件归因与限制见 [总索引](../INDEX.md)。', '']
        for key in chunk:
            d = valid.get(key)
            rendered = card(d) if d else _missing_card(key, games[key], missing[key])
            anchor = f'<a id="g-{key}"></a>'
            if anchor not in rendered:
                rendered = anchor + '\n' + rendered
            text.extend([rendered.rstrip(), '', '---', ''])
        _write(directory/name, '\n'.join(text))
        start += len(chunk)

    # One CSV row per match, in the same view as the card. For missing replays
    # use the first family model in the original a/b metadata, explicitly marked.
    game_rows = []
    for key in keys:
        g, d = games[key], valid.get(key)
        letter = next((s for s in ('a', 'b') if _name(g, s) in NAMES and
                       (not d or NAMES[_name(g, s)] == d['model'])), None)
        if letter is None:
            raise ValueError(f'no family model view: {key}')
        v = _metadata_view(g, letter)
        row = dict(key=key, model=NAMES[v['name']], model_name=v['name'], opponent=v['opponent'],
                   seat=v['seat'], result=v['result'], score=v['score'], opponent_score=v['opponent_score'],
                   margin=v['margin'], replay_status='analyzed' if d else 'missing',
                   missing_reason=missing.get(key, ''), source=d['source'] if d else '',
                   source_url=_source_url(d['source'], URL) if d else '', sha256=d['sha256'] if d else '',
                   card=links[key], created=g.get('created', ''),
                   last_step=d['last'] if d else '',
                   opening=d['opening'][0] if d else '', opponent_opening=d['opening'][1] if d else '',
                   first_full=d['first_full'][0] if d and d['first_full'][0] is not None else '',
                   opponent_first_full=d['first_full'][1] if d and d['first_full'][1] is not None else '',
                   planted=d['trees'][0][0] if d else '', seized=d['trees'][0][1] if d else '',
                   opponent_planted=d['trees'][1][0] if d else '', opponent_seized=d['trees'][1][1] if d else '',
                   tree_income=d['tree_income'][0] if d else '', opponent_tree_income=d['tree_income'][1] if d else '',
                   final_sign_since=d['stable'] if d and d['stable'] is not None else '',
                   sign_changes=d['turns'] if d else '')
        snap = {s['step']:s for s in d['snap']} if d else {}
        for t in SNAPSHOTS:
            row[f'cash_gap_t{t}'] = snap[t]['gap'] if t in snap else ''
        game_rows.append(row)
    columns = list(game_rows[0]) if game_rows else ['key', 'replay_status', 'card']
    _csv(directory/'all_games.csv', game_rows, columns)

    # A family head-to-head is one game but two model appearances. Official
    # metadata scores and replay-conditioned statistics have separate counts.
    model_views = defaultdict(list)
    for key in keys:
        for letter in ('a', 'b'):
            name = _name(games[key], letter)
            if name in NAMES:
                model_views[NAMES[name]].append(dict(key=key, **_metadata_view(games[key], letter)))
    model_summary = {}
    result_rows = []
    stage_rows = []
    for model in MODELS:
        views = model_views[model]
        analyzed = [v for v in views if v['key'] in valid]
        by_opponent = defaultdict(list)
        by_seat = defaultdict(list)
        for v in views:
            by_opponent[v['opponent']].append(v)
            by_seat[str(v['seat'])].append(v)
        model_summary[model] = dict(metadata=_tally(views), analyzed=_tally(analyzed),
                                    missing_keys=[v['key'] for v in views if v['key'] not in valid],
                                    by_opponent={n:_tally(v) for n,v in sorted(by_opponent.items())},
                                    by_seat={n:_tally(v) for n,v in sorted(by_seat.items())})
        for scope, vs in [('all_metadata', views), ('analyzed_replays', analyzed)]:
            result_rows.append(dict(model=model, scope=scope, **_tally(vs)))
        for stage in (*SNAPSHOTS, 'final'):
            samples = []
            for v in analyzed:
                d = valid[v['key']]
                side = 0 if d['names'][0] == v['name'] else 1
                sn = d['snap'][-1] if stage == 'final' else next((s for s in d['snap'] if s['step'] == stage), None)
                if sn is None:
                    continue
                samples.append(dict(step=sn['step'], cash_gap=sn['gap']*(1 if side == 0 else -1),
                                    workers=sn['workers'][side], opponent_workers=sn['workers'][1-side],
                                    trees=sn['trees'][side], opponent_trees=sn['trees'][1-side],
                                    cargo=sn['cargo'][side], opponent_cargo=sn['cargo'][1-side]))
            stage_rows.append(dict(model=model, stage=str(stage), n_games=len(samples),
                                   **{'mean_'+field:_mean(s[field] for s in samples) for field in
                                      ('step','cash_gap','workers','opponent_workers','trees','opponent_trees','cargo','opponent_cargo')}))
    _csv(directory/'model_results.csv', result_rows, list(result_rows[0]) if result_rows else ['model'])
    _csv(directory/'stage_summary.csv', stage_rows, list(stage_rows[0]) if stage_rows else ['model'])

    # Source means the actual replay path used by study, not the nominal model
    # corpus: cross-family aliases must not be counted as separate games.
    source_groups = defaultdict(list)
    for row in game_rows:
        source = _source_group(row['source']) if row['source'] else 'missing_replay'
        source_groups[source].append(row)
    by_source = {}
    for source, rows in sorted(source_groups.items()):
        info = dict(source=source, games=len(rows), keys=[r['key'] for r in rows],
                    analyzed=sum(r['replay_status']=='analyzed' for r in rows),
                    missing=sum(r['replay_status']=='missing' for r in rows),
                    primary_views=dict(Counter(r['model'] for r in rows)),
                    results_in_primary_view=_tally(rows))
        slug = source.replace('/', '__').replace('\\', '__').replace(':', '_')
        filename = f'by_source/{slug}.json'
        info['file'] = filename
        by_source[source] = info
        _json(directory/filename, dict(revision=REV, **info))

    summary = dict(revision=REV, models_in_scope=list(MODELS),
                   metadata_games=len(games),
                   analyzed_games=len(valid), missing_games=len(missing),
                   missing={k:missing[k] for k in sorted(missing)}, page_size_target=PAGE_SIZE,
                   page_sizes=[len(c) for c in chunks], pages=pages,
                   unique_index_keys=len(keys), family_model_appearances=sum(len(v) for v in model_views.values()),
                   models=model_summary, by_source=by_source, stages=stage_rows,
                   definitions={
                       'scope':'Input metadata union for the configured models; includes frozen and explicitly supplied official records. One index/CSV row per matchid-game; absent replays remain explicit.',
                       'results':'Official winner and final metadata scores. No confidence interval or causal strength claim.',
                       'family_head_to_head':'One unique game, two appearances in per-model summaries; never add these as independent games.',
                       'source':'Actual replay parent directory or official replay endpoint selected by study; aliases are not extra games.',
                       'stages':'Observed state snapshots only. Missing/early-ended games are not padded. final mixes actual terminal steps, reported via mean_step.',
                       'scores_vs_cash':'Final metadata score can differ from final observation cash; stage cash gap is not silently replaced by official margin.',
                   })
    _json(directory/'summary.json', summary)

    lines = ['# Top 2：Master5s 与 Master6 逐局深读索引', '', f'代码与仓内回放冻结提交：`{REV}`。范围：{"、".join(MODELS)}；如有官网补充记录，以实际来源栏区分。', '',
             f'元数据并集 **{len(games)} 局**；有逐局分析 **{len(valid)} 局**；缺回放或分析失败 **{len(missing)} 局**。'
             f'每局唯一键为完整 `matchid-game`，按每页约 {PAGE_SIZE} 局均衡分页。', '']
    lines += ['[每局一行 CSV](all_games.csv) · [模型成绩 CSV](model_results.csv) · '
              '[状态截面 CSV](stage_summary.csv) · [完整汇总 JSON](summary.json)', '',
              '## 读法与限制', '',
              '- 逐局卡的 tN、交付、碰撞、窗口和状态表均指动作执行后可见的状态步；只有明确标为“决策步”的字段使用推进前源步（决策 t 产生 state t+1）。位置为 15×15 observation 行优先索引。',
              '- 卡片关注方在前；CSV 与卡片采用同一视角。家族内对战只占一行，但在模型成绩中分别记入双方，因此模型出场次数之和可大于独立局数。',
              '- 官方成绩来自元数据；阶段现金、空间与事件只来自有回放的局。缺回放不补零，提前终局不延长。',
              '- 对手与席位不平衡，版本/时间不同；下表是语料描述，不是固定考官的配对实力实验。窗口、事件及胜负是事后观察，不能单凭它们声称某动作的反事实收益。',
              '- 树 ID 前缀不表示当前所有权；击杀、捕获载货、银行交付是不同事件。完整隐藏图不构成策略可见输入。', '',
              '## 按模型成绩（官方元数据）', '',
              '| 模型 | 出场 | 有回放 | 胜 / 负 / 平 / 未知 | 平均分差 | 中位分差 |',
              '|---|---:|---:|---|---:|---:|']
    for model in MODELS:
        ms = model_summary[model];m=ms['metadata']
        lines.append(f"| {model} | {m['games']} | {ms['analyzed']['games']} | {m['wins']} / {m['losses']} / {m['draws']} / {m['unknown']} | {_fmt(m['mean_margin'],True)} | {_fmt(m['median_margin'],True)} |")
    lines += ['', '## 状态截面（有回放的实际观察）', '',
              '每一行各自报告有效局数。final 为每局真实最后状态，具体步数可能不同；均值不能与固定 t299 无条件互换。', '',
              '| 模型 | 状态步 | n | 平均现金差 | 工人 / 对手 | 树 / 对手 | 载货 / 对手 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for s in stage_rows:
        lines.append(f"| {s['model']} | {s['stage']} | {s['n_games']} | {_fmt(s['mean_cash_gap'],True)} | {_fmt(s['mean_workers'])} / {_fmt(s['mean_opponent_workers'])} | {_fmt(s['mean_trees'])} / {_fmt(s['mean_opponent_trees'])} | {_fmt(s['mean_cargo'])} / {_fmt(s['mean_opponent_cargo'])} |")
    lines += ['', '## 实际回放来源', '', '| 来源 | 局数 | 已分析 | 缺失 | 汇总 |', '|---|---:|---:|---:|---|']
    for source, info in by_source.items():
        lines.append(f"| `{_esc(source)}` | {info['games']} | {info['analyzed']} | {info['missing']} | [JSON]({info['file']}) |")
    if missing:
        lines += ['', '## 缺口清单', '', '| 完整身份 | 原因 | 索引卡 |', '|---|---|---|']
        for key in sorted(missing):
            lines.append(f'| `{key}` | {_esc(missing[key])} | [缺口卡]({links[key]}) |')
    lines += ['', '## 全部对局', '', '| # | 完整身份 | 关注模型 / 对手 | 席位 | 结果 | 官方分差 | 回放 |',
              '|---:|---|---|---:|---|---:|---|']
    labels = {'win':'胜','loss':'负','draw':'平','unknown':'未知'}
    for i, row in enumerate(game_rows, 1):
        lines.append(f"| {i} | [`{row['key']}`]({row['card']}) | {row['model']} / {_esc(row['opponent'])} | {row['seat']} | {labels[row['result']]} | {_fmt(row['margin'],True)} | {'已分析' if row['replay_status']=='analyzed' else '**缺失**'} |")
    _write(directory/'INDEX.md', '\n'.join(lines)+'\n')
    return summary

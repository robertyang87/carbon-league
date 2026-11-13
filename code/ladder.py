"""Carbon League 天梯客户端（标准库，无依赖）。

用法（在 code/ 下）：
  python3 ladder.py ping                         # 连通性（不登录）
  python3 ladder.py login <用户名>                # 密码从环境变量 LADDER_PASS 或交互输入读取，不落盘
  python3 ladder.py whoami                       # 当前登录身份 + 今日剩余提交次数
  python3 ladder.py submit <文件> <策略名> [说明文件]  # 说明默认取文件头 docstring
  python3 ladder.py mine                         # 我的策略（含定位进度、榜外排名）
  python3 ladder.py board [N]                    # 排行榜前 N（默认 30）
  python3 ladder.py matches <策略ID> [输出.json]   # 拉该策略全部对局（compact，分页）
  python3 ladder.py replays <matches.json> <目录>  # 下载回放（登录后；≤30 次/分钟；已存在跳过）

环境变量：
  LADDER_BASE     默认 https://132.232.176.153 。云容器里裸 IP 的 HTTPS 会被出网代理重置（无 SNI），
                  先起 `python3 sni_bridge.py`，再设 LADDER_BASE=http://127.0.0.1:18443（Host 头自动设为 IP）。
                  sslip.io 之类域名不可用：腾讯云对未备案域名拦截。
  LADDER_COOKIES  会话 cookie 文件，默认 ~/.carbon_ladder_cookies（仓库外，不要提交）。
证书：服务器是自签名证书，与旧脚本 curl -k 一致不校验。
"""
import ast, gzip, http.client, json, os, ssl, sys, time, uuid, urllib.error, urllib.request

BASE = os.environ.get("LADDER_BASE", "https://132.232.176.153").rstrip("/")
HOST = os.environ.get("LADDER_HOST", "132.232.176.153" if "127.0.0.1" in BASE else "")
COOKIES = os.path.expanduser(os.environ.get("LADDER_COOKIES", "~/.carbon_ladder_cookies"))

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE
# 手工管理 cookie：服务器 cookie 带 Secure，经本地明文桥时标准 cookiejar 不会回传。
_cookies = {}
if os.path.exists(COOKIES):
    try:
        _cookies = json.load(open(COOKIES))
    except ValueError:
        _cookies = {}
_opener = urllib.request.build_opener(
    urllib.request.ProxyHandler(),  # 读取 HTTPS_PROXY
    urllib.request.HTTPSHandler(context=_ctx),
)
_csrf = None


import threading as _th
_ck_lock = _th.Lock()


def _absorb(hdrs):
  with _ck_lock:
    for v in hdrs.get_all("Set-Cookie") or []:
        k, _, val = v.split(";", 1)[0].partition("=")
        _cookies[k.strip()] = val.strip()
    fd = os.open(COOKIES, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(_cookies, f)


def _ensure_bridge():
    """本地桥没起时自动拉起（容器重启 / 进程被回收后常见）。"""
    import subprocess
    log = open(os.path.expanduser("~/.carbon_bridge.log"), "a")
    subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sni_bridge.py")],
                     stdout=log, stderr=log, start_new_session=True)
    time.sleep(2)


class AmbiguousRequestError(RuntimeError):
    """A non-GET request whose outcome is UNKNOWN: the transport failed
    after the request may have reached the server (F08). Blindly retrying
    can double-apply it (duplicate submission / quota burn), so callers
    must first check server state via GET (see _recover_post).
    possibly_sent=False marks failures where the connection was provably
    never established (e.g. the local bridge refused it), so one retry is
    safe."""

    def __init__(self, msg, possibly_sent=True):
        super().__init__(msg)
        self.possibly_sent = possibly_sent


def _is_refused(e):
    seen = set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, ConnectionRefusedError):
            return True
        e = getattr(e, "reason", None) or getattr(e, "__cause__", None)
    return False


def _open_retry(req, timeout):
    """GET: idempotent -- connection refused starts the local bridge;
    reset/disconnect/timeout retry with backoff (max 6 attempts). HTTP
    errors never retry.
    Non-GET: NEVER blind-retried (F08). Exactly one attempt; a transport
    failure raises AmbiguousRequestError so the caller can verify server
    state before any explicit recovery resend."""
    method = req.get_method()
    if method != "GET":
        try:
            return _opener.open(req, timeout=timeout)
        except urllib.error.HTTPError:
            raise  # definitive server answer: not ambiguous
        except (urllib.error.URLError, ConnectionError, http.client.HTTPException,
                TimeoutError, OSError) as e:
            if _is_refused(e) and "127.0.0.1" in BASE:
                _ensure_bridge()  # connection never established: nothing was sent
                raise AmbiguousRequestError(
                    f"{method} {req.full_url}: local bridge was down (started); "
                    "request provably not sent, one retry is safe",
                    possibly_sent=False) from e
            raise AmbiguousRequestError(
                f"{method} {req.full_url}: transport failed after the request "
                f"may have been sent ({type(e).__name__}: {e}); outcome unknown, "
                "do NOT blind-retry -- verify state via GET first",
                possibly_sent=True) from e
    for attempt in range(6):
        try:
            return _opener.open(req, timeout=timeout)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, ConnectionError, http.client.HTTPException, TimeoutError, OSError) as e:
            if attempt == 5:
                raise
            reason = getattr(e, "reason", e)
            if isinstance(reason, ConnectionRefusedError) and "127.0.0.1" in BASE:
                _ensure_bridge()
            else:
                time.sleep(2 * 2 ** attempt)


# --- Post-hoc state tri-state (owner tightening of the F08 protocol) -------
# After an ambiguous POST (timeout / disconnect), a state check via GET can
# only conclude one of three things:
#   APPLIED  -- a positive record that the request took effect (e.g. the
#               submitted strategy is now listed under /api/mine).
#   REJECTED -- an EXPLICIT terminal failure/rejection record tied to this
#               request's identity (e.g. a submission-log entry in a failed
#               state). This is the ONLY outcome that permits exactly one
#               explicit recovery resend.
#   UNKNOWN  -- everything else: not found in a listing, listing lag, the
#               check itself failing or timing out. "Not found" is NOT
#               proof of non-application: the first attempt may still have
#               landed. UNKNOWN always stops with AmbiguousRequestError.
# Basis: the GET surface this client uses (/api/session, /api/mine,
# /api/leaderboard) exposes only current-state listings; it has NO
# per-request terminal-record / submission-log endpoint. The check helpers
# below therefore never return REJECTED today -- REJECTED exists for the
# day such a terminal record exists, and absence from a listing must never
# be re-interpreted as REJECTED.
APPLIED = "applied"
REJECTED = "rejected"
UNKNOWN = "unknown"


def _norm_state(v):
    """Normalize a check result to the tri-state. Legacy booleans are read
    conservatively: True -> APPLIED, False/None/anything else -> UNKNOWN
    (absence of a record is not a rejection record)."""
    if v is True or v == APPLIED:
        return APPLIED
    if v == REJECTED:
        return REJECTED
    return UNKNOWN


def _check_state(check_applied):
    try:
        return _norm_state(check_applied())
    except Exception:
        return UNKNOWN  # the check itself failed/timed out: unknown, not rejected


def _recover_post(desc, do_post, check_applied, manual=None):
    """Run a non-idempotent POST under the F08 ambiguous-outcome protocol.

    do_post() performs the request once and returns its parsed result.
    check_applied() queries fresh server state via GET and returns one of
    APPLIED / REJECTED / UNKNOWN (see the tri-state comment above). Returns
    do_post()'s result, or the string "confirmed-applied" when a state
    check proves the first (or the recovery) attempt landed. A recovery
    resend happens ONLY on REJECTED (an explicit terminal failure record
    for this request identity), and exactly once. UNKNOWN -- including
    "not found in the listing" and a failed/timed-out check -- raises
    AmbiguousRequestError with manual-verification guidance; it never
    auto-resends."""
    try:
        return do_post()
    except AmbiguousRequestError as e:
        if not e.possibly_sent:
            return do_post()  # provably unsent: one safe retry
        state = _check_state(check_applied)
        if state == APPLIED:
            return "confirmed-applied"
        if state != REJECTED:
            hint = f" 人工核对指引：{manual}" if manual else ""
            raise AmbiguousRequestError(
                f"{desc}: 结果不确定 (state=UNKNOWN：未查到生效记录，也没有"
                f"该请求身份的明确失败终态；查不到不等于未生效) -- 已停手，"
                f"禁止自动重发。{hint}", possibly_sent=True)
        # REJECTED: explicit terminal failure record for this request
        # identity -- exactly ONE explicit recovery resend.
        try:
            return do_post()
        except AmbiguousRequestError:
            if _check_state(check_applied) == APPLIED:
                return "confirmed-applied"
            hint = f" 人工核对指引：{manual}" if manual else ""
            raise AmbiguousRequestError(
                f"{desc}: 结果不确定 (recovery resend outcome unconfirmed) -- "
                f"已停手，禁止再重发。{hint}", possibly_sent=True)


def _login_state():
    """APPLIED iff the session positively shows a logged-in user. No user
    in the session response is UNKNOWN (the login may still have landed /
    the session view may lag), never REJECTED: /api/session returns no
    terminal record for a login attempt."""
    try:
        return APPLIED if session().get("user") is not None else UNKNOWN
    except Exception:
        return UNKNOWN


def _submit_state(name):
    """APPLIED iff a strategy with this exact name is positively listed in
    /api/mine. Absence is UNKNOWN (the submission may still be processing
    or the listing may lag), never REJECTED: /api/mine is a current-state
    listing with no per-submission terminal record."""
    try:
        return APPLIED if any(x.get("name") == name
                              for x in request("/api/mine").get("strategies", [])) else UNKNOWN
    except Exception:
        return UNKNOWN


def _retire_state(sid):
    """APPLIED iff the strategy id is positively ABSENT from /api/mine
    (the delete's effect is observed). Still being listed is UNKNOWN (the
    delete may not have been processed yet), never REJECTED: the listing
    carries no terminal record for the delete request itself."""
    try:
        return UNKNOWN if sid in {x["id"] for x in request("/api/mine").get("strategies", [])} else APPLIED
    except Exception:
        return UNKNOWN


def request(path, method="GET", body=None, headers=None, raw=False, timeout=60):
    h = {"Accept": "application/json"}
    if HOST:
        h["Host"] = HOST
    if _csrf:
        h["X-CSRF-Token"] = _csrf
    if isinstance(body, (dict, list)):
        body = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    if _cookies:
        h["Cookie"] = "; ".join(f"{k}={v}" for k, v in _cookies.items())
    origin = "https://" + (HOST or BASE.split("://", 1)[1].split("/")[0])
    h.setdefault("Origin", origin)
    h.setdefault("Referer", origin + "/")
    h.update(headers or {})
    req = urllib.request.Request(BASE + path, data=body, method=method, headers=h)
    for attempt in range(4 if method == "GET" else 1):   # GET 幂等：读响应中途被重置也整体重试
        try:
            r = _open_retry(req, timeout)
            with r:
                data = r.read()
                status = r.status
                _absorb(r.headers)
            break
        except urllib.error.HTTPError as e:
            data, status = e.read(), e.code
            _absorb(e.headers)
            break
        except AmbiguousRequestError:
            raise
        except (ConnectionError, http.client.HTTPException, OSError) as e:
            if method != "GET":
                raise AmbiguousRequestError(
                    f"{method} {path}: response stream failed after send "
                    f"({type(e).__name__}: {e}); outcome unknown -- verify "
                    "state via GET before any resend", possibly_sent=True) from e
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    if raw:
        return status, data
    try:
        js = json.loads(data)
    except ValueError:
        raise RuntimeError(f"{method} {path}: HTTP {status}, 非 JSON 响应: {data[:200]!r}")
    if status >= 400:
        raise RuntimeError(f"{method} {path}: HTTP {status}: {js.get('error', js)}")
    return js


def session():
    global _csrf
    s = request("/api/session")
    _csrf = s.get("csrf")
    return s


def cmd_ping():
    t = time.time()
    d = request("/api/leaderboard")
    print(f"OK {BASE}  {len(d.get('strategies', []))} 条榜单  {time.time() - t:.2f}s")


def cmd_login(user):
    pw = os.environ.get("LADDER_PASS")
    if not pw:
        import getpass
        pw = getpass.getpass(f"{user} 的密码: ")
    session()

    r = _recover_post("login",
                      lambda: request("/api/login", "POST", {"username": user, "password": pw}),
                      _login_state,
                      manual=f"运行 `python3 ladder.py whoami` 按登录名「{user}」核对："
                             "显示已登录即第一次已生效、切勿重复登录；显示未登录也不等于"
                             "未生效（会话视图可能未刷新），稍后再查，仍不确定先人工确认再决定。")
    if r == "confirmed-applied":
        print("登录请求曾断连，但状态核对确认会话已建立")
    cmd_whoami()


def cmd_whoami():
    s = session()
    u = s.get("user")
    print(f"登录: {u.get('name')}  今日剩余提交: {s.get('remaining')}" if u else "未登录")


def docstring_of(path):
    try:
        return ast.get_docstring(ast.parse(open(path, encoding="utf-8").read())) or ""
    except SyntaxError:
        return ""


def cmd_submit(path, name, desc_path=None):
    src = open(path, "rb").read()
    assert 0 < len(src) <= 10_000_000, f"文件大小 {len(src)} 超限"
    desc = open(desc_path, encoding="utf-8").read() if desc_path else docstring_of(path)
    s = session()
    if not s.get("user"):
        sys.exit("未登录：先 python3 ladder.py login <用户名>")
    print(f"提交 {path} ({len(src):,} B) 为「{name}」，今日剩余 {s.get('remaining')} 次")

    def _submit_once():
        b = uuid.uuid4().hex
        parts = []
        for k, v in (("name", name), ("description", desc)):
            parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="submission.py"\r\n'
                     f'Content-Type: text/x-python; charset=utf-8\r\n\r\n'.encode() + src + b"\r\n")
        parts.append(f"--{b}--\r\n".encode())
        return request("/api/submit", "POST", b"".join(parts),
                       {"Content-Type": f"multipart/form-data; boundary={b}"}, timeout=300)

    r = _recover_post(f"submit '{name}'", _submit_once,
                      lambda: _submit_state(name),
                      manual=f"运行 `python3 ladder.py mine` 按策略名「{name}」核对在册记录："
                             "查到即已生效、切勿重发；查不到不等于未生效（可能仍在处理或"
                             "列表延迟），且本站无按提交身份的终态流水可证明被拒——人工"
                             "确认前禁止重发，以免重复提交白烧当日额度。")
    if r == "confirmed-applied":
        print(f"提交请求曾断连，但状态核对确认「{name}」已在我的策略中（未重发）")
        mine = {x["name"]: x for x in request("/api/mine").get("strategies", [])}
        print(json.dumps(mine.get(name, {}), ensure_ascii=False, indent=1)[:2000])
    else:
        print(json.dumps(r, ensure_ascii=False, indent=1)[:2000])
    cmd_whoami()


def cmd_retire(sid, confirm=""):
    """Take a strategy OFF the ladder (the site's delete: stops its matches, irreversible; games / replays stay; the name
    cannot be reused). Only for finished PROBES (user's standing instruction, 2026-11-10). Needs the literal 'yes'."""
    mine = {s["id"]: s for s in request("/api/mine").get("strategies", [])}
    s = mine.get(sid)
    if not s:
        sys.exit(f"not one of ours: {sid}")
    print(f"retire {s['name']} ({sid}): {s.get('games')} games, rating {round(s.get('rating', 0))}")
    if confirm != "yes":
        sys.exit("add 'yes' to really retire it")
    r = _recover_post(f"retire {sid}",
                      lambda: request(f"/api/strategies/{sid}/delete", "POST", {}),
                      lambda: _retire_state(sid),
                      manual=f"运行 `python3 ladder.py mine` 按策略 id「{sid}」核对："
                             "不在册即下架已生效；仍在册不等于被拒（可能尚未处理），"
                             "人工确认前禁止重发。")
    if r == "confirmed-applied":
        print(f"下架请求曾断连，但状态核对确认 {sid} 已不在我的策略中（未重发）")
    else:
        print(json.dumps(r, ensure_ascii=False))


def cmd_mine():
    session()
    for s in request("/api/mine").get("strategies", []):
        keep = {k: v for k, v in s.items() if not isinstance(v, (dict, list)) and k != "description"}
        print(json.dumps(keep, ensure_ascii=False))


def cmd_board(n=30):
    d = request("/api/leaderboard")
    rows = d.get("strategies", [])[: int(n)]
    for i, s in enumerate(rows, 1):
        print(f"{i:3d} {s.get('name', '?')[:34]:34s} {s.get('rating', 0):7.0f} ±{s.get('rating_sd', 0):3.0f} "
              f"{s.get('wins', 0):3d}/{s.get('games', 0):<3d} {s.get('board', ''):5s} {s.get('ranking_phase', '')}  {s.get('id', '')}")
    if not rows:
        print(json.dumps(d, ensure_ascii=False)[:1500])


def cmd_matches(sid, out=None):
    got, off = [], 0
    while True:
        d = request(f"/api/matches?compact=1&limit=100&offset={off}&strategy={sid}")
        got += d.get("matches", [])
        if d.get("next_offset") is None:
            break
        off = d["next_offset"]
        time.sleep(0.7)
    seen, uniq = set(), []
    for m in got:
        k = (m.get("id"), m.get("game"))
        if k not in seen:
            seen.add(k)
            uniq.append(m)
    out = out or f"matches_{sid}.json"
    json.dump(uniq, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"{len(uniq)} 局 -> {out}")


def cmd_replays(mfile, outdir):
    session()
    os.makedirs(outdir, exist_ok=True)
    ms = json.load(open(mfile))
    n = 0
    for m in ms:
        if m.get("winner") is None and not m.get("scores"):
            continue  # 未结束
        fn = os.path.join(outdir, f"{m['id']}-{m['game']}.json.gz")
        if os.path.exists(fn):
            continue
        st, data = request(f"/data/{m['id']}/{m['game']}", raw=True, timeout=120)
        if st != 200:
            print(f"{m['id']}-{m['game']}: HTTP {st}")
        else:
            if data[:2] != b"\x1f\x8b":
                data = gzip.compress(data)
            open(fn, "wb").write(data)
            n += 1
        time.sleep(2.1)  # ≤30 次/分钟
    print(f"新下载 {n} 局 -> {outdir}")


if __name__ == "__main__":
    a = sys.argv[1:]
    cmds = {"ping": cmd_ping, "login": cmd_login, "whoami": cmd_whoami, "submit": cmd_submit,
            "retire": cmd_retire, "mine": cmd_mine, "board": cmd_board, "matches": cmd_matches, "replays": cmd_replays}
    if not a or a[0] not in cmds:
        sys.exit(__doc__)
    cmds[a[0]](*a[1:])

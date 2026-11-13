"""Render Carbon League trace files (traces/*.json.gz) to iOS-friendly MP4s.

Layout 960x1080 @12fps: header bar (step / cash / worker & tree counts),
15x15 carbon heat map (collectors=circles sized by cargo, planters=triangles,
blue=us, orange=juzi; trees = green discs with owner-coloured ring + age),
legend footer. 1s title card + 1s final-frame freeze.
All numbers drawn come from the trace frames themselves (never hand-typed).
"""
import gzip, json, math, os, subprocess, sys
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(os.path.abspath(__file__))
TRACES = os.path.join(BASE, "traces")
OUTDIR = os.path.join(BASE, "videos")
W, H = 960, 1080
FPS = 12
CELL = 56
MAP_X, MAP_Y = (W - CELL * 15) // 2, 165
HDR_H = 150
LEG_Y = 1018

FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_R = "/home/hatch/.fonts/wqy-microhei.ttc"

def font(path, size):
    return ImageFont.truetype(path, size)

F_TITLE = font(FONT_B, 46)
F_H1 = font(FONT_B, 34)
F_H2 = font(FONT_B, 29)
F_LEG = font(FONT_R, 21)
F_AGE = font(FONT_B, 17)
F_CARGO = font(FONT_B, 16)

US = (46, 124, 246)      # blue = our policy
OPP = (245, 130, 32)     # orange = juzi
BG = (13, 17, 23)
HDR_BG = (22, 29, 40)
TXT = (235, 240, 245)
DIM = (160, 170, 180)

VER_NAME = {"policy_v1": "v1", "policy_v0_1": "v0.1", "policy_v0_2": "v0.2"}

def heat(c):
    t = math.sqrt(max(0.0, min(100.0, c)) / 100.0)
    lo, hi = (17, 23, 17), (34, 197, 94)
    return tuple(int(lo[i] + (hi[i] - lo[i]) * t) for i in range(3))

def cell_xy(pos):
    r, c = divmod(pos, 15)
    return MAP_X + c * CELL, MAP_Y + r * CELL

def draw_frame(fr, meta, us_p):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # header
    d.rectangle([0, 0, W, HDR_H], fill=HDR_BG)
    step = fr["step"]
    cash = fr["cash"]
    workers = fr["workers"]
    trees = fr["trees"]
    cnt = {0: [0, 0], 1: [0, 0]}  # player -> [workers, trees]
    for wid in workers:
        cnt[int(wid.split("-")[1])][0] += 1
    for t in trees.values():
        cnt[t[1]][1] += 1
    ver = VER_NAME.get(meta["policy"], meta["policy"])
    side_txt = "我方先手" if meta["policy_side"] == 0 else "我方后手"
    d.text((24, 14), f"Step {step} / 300", font=F_H1, fill=TXT)
    d.text((470, 18), f"{ver} vs juzi · seed {meta['seed']} · {side_txt}", font=F_H2, fill=DIM)
    for row, p in enumerate((us_p, 1 - us_p)):
        col = US if p == us_p else OPP
        name = f"我方({ver})" if p == us_p else "juzi"
        y = 62 + row * 44
        d.rectangle([24, y + 4, 44, y + 24], fill=col)
        d.text((56, y), f"{name}  资金 {cash[p]:,.0f}   工人 {cnt[p][0]}   树 {cnt[p][1]}",
               font=F_H2, fill=TXT)
    # map cells
    carbon = fr["carbon"]
    for pos in range(225):
        x, y = cell_xy(pos)
        d.rectangle([x, y, x + CELL - 1, y + CELL - 1], fill=heat(carbon[pos]))
    # bases
    for p_str, lst in meta["bases"].items():
        p = int(p_str)
        col = US if p == us_p else OPP
        for _name, pos in lst:
            x, y = cell_xy(pos)
            d.rectangle([x + 3, y + 3, x + CELL - 4, y + CELL - 4], outline=(255, 255, 255), width=3)
            d.rectangle([x + 9, y + 9, x + CELL - 10, y + CELL - 10], fill=col)
    # trees
    for _tid, (pos, owner, age) in trees.items():
        x, y = cell_xy(pos)
        cx, cy = x + CELL // 2, y + CELL // 2
        ring = US if owner == us_p else OPP
        d.ellipse([cx - 19, cy - 19, cx + 19, cy + 19], fill=(27, 94, 32), outline=ring, width=4)
        s = str(age)
        tw = d.textlength(s, font=F_AGE)
        d.text((cx - tw / 2, cy - 11), s, font=F_AGE, fill=(255, 255, 255))
    # workers (offset duplicates sharing a cell)
    by_pos = {}
    for wid, w in workers.items():
        by_pos.setdefault(w[0], []).append((wid, w))
    offs = [(0, 0), (-13, -13), (13, -13), (-13, 13), (13, 13), (0, -15), (0, 15), (-15, 0), (15, 0)]
    for pos, ws in by_pos.items():
        x, y = cell_xy(pos)
        cx, cy = x + CELL // 2, y + CELL // 2
        for i, (wid, (ppos, cargo, kind)) in enumerate(ws):
            p = int(wid.split("-")[1])
            col = US if p == us_p else OPP
            ox, oy = offs[i % len(offs)]
            wx, wy = cx + ox, cy + oy
            if kind == "C":
                r = 10 + min(cargo, 500) / 500 * 11
                d.ellipse([wx - r, wy - r, wx + r, wy + r], fill=col, outline=(255, 255, 255), width=2)
                if cargo >= 200:
                    s = str(int(cargo))
                    tw = d.textlength(s, font=F_CARGO)
                    d.text((wx - tw / 2, wy - 9), s, font=F_CARGO, fill=(255, 255, 255))
            else:
                r = 15
                d.polygon([(wx, wy - r), (wx - r, wy + r - 2), (wx + r, wy + r - 2)],
                          fill=col, outline=(255, 255, 255))
    # legend
    d.rectangle([0, LEG_Y, W, H], fill=HDR_BG)
    d.text((24, LEG_Y + 7),
           "格子越绿碳越多 · 圆=采集员(越大携带越多) · 三角=种植员 · 绿圈=树(圈色=主人,数字=树龄)",
           font=F_LEG, fill=DIM)
    d.text((24, LEG_Y + 34), "方块=基地 · 蓝=我方 · 橙=juzi", font=F_LEG, fill=DIM)
    return img

def draw_title(meta, us_p):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    ver = VER_NAME.get(meta["policy"], meta["policy"])
    fc = meta["final_cash"]
    us_cash, opp_cash = fc[us_p], fc[1 - us_p]
    side_txt = "我方先手 (P0)" if meta["policy_side"] == 0 else "我方后手 (P1)"
    lines = [
        ("Carbon League 对局回放", F_TITLE, TXT, 300),
        (f"{ver}  vs  juzi", F_TITLE, TXT, 400),
        (f"seed {meta['seed']} · {side_txt}", F_H1, DIM, 500),
        (f"终分  我方 {us_cash:,.2f} : {opp_cash:,.2f} juzi", F_H1, TXT, 620),
    ]
    for text, f, col, y in lines:
        tw = d.textlength(text, font=f)
        d.text(((W - tw) / 2, y), text, font=f, fill=col)
    return img

def render(trace_name):
    path = os.path.join(TRACES, trace_name)
    with gzip.open(path, "rt") as fh:
        data = json.load(fh)
    meta, frames = data["meta"], data["frames"]
    assert len(frames) == 301, f"{trace_name}: {len(frames)} frames"
    f300 = frames[300]["cash"]
    assert abs(f300[0] - meta["final_cash"][0]) < 0.01 and abs(f300[1] - meta["final_cash"][1]) < 0.01, \
        f"{trace_name}: frame300 cash {f300} != meta final {meta['final_cash']}"
    us_p = meta["policy_side"]
    ver = VER_NAME.get(meta["policy"], meta["policy"])
    out = os.path.join(OUTDIR, f"{ver.replace('.', '_')}_vs_juzi_seed{meta['seed']}_side{us_p}.mp4")
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-preset", "veryfast", "-crf", "23", "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    title = draw_title(meta, us_p)
    for _ in range(FPS):
        proc.stdin.write(title.tobytes())
    last = None
    for fr in frames:
        last = draw_frame(fr, meta, us_p)
        proc.stdin.write(last.tobytes())
    for _ in range(FPS):
        proc.stdin.write(last.tobytes())
    proc.stdin.close()
    rc = proc.wait()
    assert rc == 0, f"ffmpeg failed for {trace_name}"
    print("OK", os.path.basename(out), os.path.getsize(out))
    return out

if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    names = sys.argv[1:]
    if not names:
        names = sorted(f for f in os.listdir(TRACES) if f.endswith(".json.gz")
                       and ("policy_v1_" in f or "policy_v0_1_" in f))
    for n in names:
        render(n)

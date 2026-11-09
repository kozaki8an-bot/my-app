#!/usr/bin/env python3
"""HaulUp Logistics — 1080x1080 square ad. Frame renderer (Pillow) piped into FFmpeg.

Usage:  python3 render.py            -> build/video_silent.mp4 (then see build.sh for mux)
        python3 render.py --frames 12 90 200   -> PNG previews in build/preview/
Timing comes from build/words.json (Whisper word timestamps of the real voice-over).
"""
import json, math, subprocess, sys, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

HERE = os.path.dirname(os.path.abspath(__file__))
A = lambda *p: os.path.join(HERE, "assets", *p)
W = H = 1080
FPS = 30
DUR = 21.0
NAVY = (7, 20, 46)
NAVY2 = (12, 36, 86)
GREEN = (139, 225, 42)
ORANGE = (242, 106, 27)
WHITE = (255, 255, 255)

ANTON = lambda s: ImageFont.truetype(A("fonts", "Anton-Regular.ttf"), s)
INTER = lambda s: ImageFont.truetype("/usr/share/fonts/opentype/inter/Inter-Bold.otf", s)

# scene boundaries (seconds) — aligned to the narration
T_S2, T_S3, T_S4, T_S5 = 3.70, 9.80, 13.85, 16.00
XF = 0.30  # cross-fade length

# ---------- helpers ----------
clamp = lambda x, a=0.0, b=1.0: max(a, min(b, x))
lerp = lambda a, b, t: a + (b - a) * t
def eo(t): t = clamp(t); return 1 - (1 - t) ** 3
def eo5(t): t = clamp(t); return 1 - (1 - t) ** 5
def eob(t):  # ease-out with tiny overshoot
    t = clamp(t); c = 1.2; return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2
def prog(t, t0, d): return clamp((t - t0) / d)

def cover(path, scale=1.5, flip=False):
    im = Image.open(path).convert("RGB")
    if flip: im = im.transpose(Image.FLIP_LEFT_RIGHT)
    s = max(W, H) / min(im.size) * scale
    return im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)

def view(big, p, z0=1.0, z1=1.15, c0=(0.5, 0.5), c1=(0.5, 0.5), out=(W, H)):
    """Ken-Burns crop of a pre-scaled image. p in 0..1."""
    bw, bh = big.size
    base = min(bw, bh)
    z = lerp(z0, z1, p)
    s = base / z
    cx = lerp(c0[0], c1[0], p) * bw; cy = lerp(c0[1], c1[1], p) * bh
    x0 = clamp(cx - s / 2, 0, bw - s); y0 = clamp(cy - s / 2, 0, bh - s)
    return big.resize(out, Image.BICUBIC, box=(x0, y0, x0 + s, y0 + s))

def vgrad(w, h, top, bot):
    a = np.linspace(0, 1, h)[:, None, None]
    arr = np.array(top)[None, None, :] * (1 - a) + np.array(bot)[None, None, :] * a
    return Image.fromarray(np.repeat(arr, w, 1).astype("uint8"), "RGBA" if len(top) == 4 else "RGB")

def overlay(base, layer, pos=(0, 0), alpha=1.0):
    """Paste RGBA layer with extra global alpha; clips at edges."""
    if layer.mode != "RGBA": layer = layer.convert("RGBA")
    a = layer.getchannel("A")
    if alpha < 1: a = a.point(lambda v: int(v * alpha))
    base.paste(layer.convert("RGB"), (int(pos[0]), int(pos[1])), a)

def text_layer(text, font, fill=WHITE, shadow=True, pad=14):
    l, t, r, b = font.getbbox(text)
    w, h = r - l + pad * 2, b - t + pad * 2
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if shadow:
        sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(sh).text((pad - l + 3, pad - t + 5), text, font=font, fill=(0, 0, 0, 150))
        im = Image.alpha_composite(im, sh.filter(ImageFilter.GaussianBlur(5)))
        d = ImageDraw.Draw(im)
    d.text((pad - l, pad - t), text, font=font, fill=fill)
    return im

def rrect(w, h, r, fill, outline=None, ow=0):
    k = 3
    im = Image.new("RGBA", (w * k, h * k), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle((0, 0, w * k - 1, h * k - 1), r * k, fill=fill,
                                         outline=outline, width=ow * k)
    return im.resize((w, h), Image.LANCZOS)

def label(text, font, fg, bg, padx=34, pady=14, r=16):
    t = text_layer(text, font, fg, shadow=False, pad=0)
    box = rrect(t.width + padx * 2, t.height + pady * 2, r, bg)
    box.alpha_composite(t, (padx, pady))
    return box

def slide(base, layer, x, y, t, t0, d=0.45, dx=-90, dy=0, blur=True):
    """Controlled slide-in with fade + light motion blur."""
    e = eo5(prog(t, t0, d))
    if e <= 0: return
    if blur:
        rad = (1 - e) * 7
        if rad > 0.6: layer = layer.filter(ImageFilter.GaussianBlur(rad))
    overlay(base, layer, (x + dx * (1 - e), y + dy * (1 - e)), alpha=e)

def centered(layer): return (W - layer.width) // 2

# ---------- assets ----------
IMG3 = cover(A("img", "3.webp"), 1.6)
IMG4 = cover(A("img", "4.webp"), 1.5)
IMG5 = cover(A("img", "5.webp"), 1.4)
IMG4_BLUR = view(IMG4, 0, 1.0, 1.0, out=(540, 540)).filter(ImageFilter.GaussianBlur(7)).resize((W, H), Image.BICUBIC)
LOGO = Image.open(A("img", "logo.png")).convert("RGBA")
GRAD_TOP = vgrad(W, 620, (7, 20, 46, 235), (7, 20, 46, 0))
GRAD_BOT = vgrad(W, 420, (7, 20, 46, 0), (7, 20, 46, 235))

def tinted(img, a=0.55):
    return Image.blend(img, Image.new("RGB", img.size, NAVY), a)

# ---------- fuel card (generic, illustrative) ----------
def make_card():
    k = 2; w, h = 580 * k, 366 * k
    card = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    g = vgrad(w, h, (30, 33, 40), (8, 9, 12)).convert("RGBA")
    mask = rrect(w, h, 36 * k, (255, 255, 255, 255)).getchannel("A") if False else None
    m = Image.new("L", (w, h), 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), 36 * k, fill=255)
    card.paste(g, (0, 0), m)
    d = ImageDraw.Draw(card)
    # green wave lines along the bottom
    for i in range(7):
        pts = [(x, h * 0.80 + 28 * k * math.sin(x / (95 * k) + i * 0.35) + i * 7 * k) for x in range(0, w, 6)]
        d.line(pts, fill=(139, 225, 42, 255 - i * 30), width=2 * k)
    d = ImageDraw.Draw(card)
    d.rounded_rectangle((0, 0, w - 1, h - 1), 36 * k, outline=(139, 225, 42, 255), width=3 * k)
    # chip
    d.rounded_rectangle((44 * k, 120 * k, 124 * k, 190 * k), 12 * k, fill=(214, 178, 96, 255))
    for yy in (141, 155, 169):
        d.line((44 * k, yy * k, 124 * k, yy * k), fill=(150, 120, 60, 255), width=2 * k)
    d.line((84 * k, 120 * k, 84 * k, 190 * k), fill=(150, 120, 60, 255), width=2 * k)
    # pump icon
    px, py = 400 * k, 50 * k
    d.rounded_rectangle((px, py, px + 70 * k, py + 110 * k), 8 * k, outline=(90, 94, 100, 255), width=5 * k)
    d.rectangle((px + 14 * k, py + 14 * k, px + 56 * k, py + 44 * k), outline=(90, 94, 100, 255), width=4 * k)
    d.line((px + 70 * k, py + 30 * k, px + 100 * k, py + 56 * k, px + 100 * k, py + 96 * k), fill=(90, 94, 100, 255), width=5 * k)
    f = ImageFont.truetype(A("fonts", "Anton-Regular.ttf"), 40 * k)
    d.text((44 * k, 48 * k), "FUEL CARD", font=f, fill=(235, 238, 240, 255))
    d.text((44 * k, 238 * k), "••••  ••••  ••••  ••••", font=INTER(30 * k), fill=(190, 195, 200, 255))
    return card
CARD = make_card()

# ---------- words / subtitles ----------
WORDS = json.load(open(os.path.join(HERE, "build", "words.json")))
def build_chunks():
    chunks, cur = [], []
    for w in WORDS:
        cur.append(w)
        txt = " ".join(x["w"] for x in cur)
        end_punct = w["w"][-1] in ".?!"
        soft = w["w"][-1] == "," and len(txt) > 18
        if end_punct or soft or len(txt) > 46:
            chunks.append(cur); cur = []
    if cur: chunks.append(cur)
    return chunks
CHUNKS = build_chunks()

def wrap_two(words, font, maxw=930):
    lines, cur = [], []
    for w in words:
        trial = cur + [w]
        if font.getlength(" ".join(x["w"] for x in trial)) > maxw and cur:
            lines.append(cur); cur = [w]
        else: cur = trial
    lines.append(cur)
    return lines

SUB_FONT = INTER(54)
def draw_subtitles(img, t):
    for i, ch in enumerate(CHUNKS):
        s = ch[0]["s"] - 0.04
        nxt = CHUNKS[i + 1][0]["s"] - 0.04 if i + 1 < len(CHUNKS) else 99
        e = min(ch[-1]["e"] + 0.30, nxt)
        if s <= t < e:
            lines = wrap_two(ch, SUB_FONT)
            lh = 70; bh = lh * len(lines) + 30
            maxw = max(SUB_FONT.getlength(" ".join(x["w"] for x in ln)) for ln in lines)
            bw = int(maxw + 64)
            box = rrect(bw, bh, 22, (5, 12, 30, 205))
            d = ImageDraw.Draw(box)
            for li, ln in enumerate(lines):
                lw = SUB_FONT.getlength(" ".join(x["w"] for x in ln))
                x = (bw - lw) / 2; y = 15 + li * lh + 2
                for w in ln:
                    active = w["s"] <= t < w["e"] + 0.05
                    d.text((x, y), w["w"], font=SUB_FONT, fill=GREEN if active else WHITE)
                    x += SUB_FONT.getlength(w["w"] + " ")
            a = min(1, (t - s) / 0.12, (e - t) / 0.12)
            overlay(img, box, ((W - bw) // 2, 1080 - 70 - bh), alpha=clamp(a))
            return

# ---------- scenes ----------
def scene1(t):
    p = clamp(t / 3.9)
    img = view(IMG3, p, 1.0, 1.12, (0.5, 0.56), (0.46, 0.56))
    overlay(img, GRAD_TOP, (0, 0)); overlay(img, GRAD_BOT, (0, H - 420))
    f = ANTON(176)
    slide(img, text_layer("OWN A CAR", f), 54, 40, t, 0.0, 0.35, dx=-80)
    slide(img, text_layer("HAULER?", f), 54, 215, t, 0.10, 0.35, dx=-80)
    lab = label("GET A COMPANY FUEL CARD", ANTON(62), (10, 10, 10), GREEN)
    slide(img, lab, 64, 430, t, 1.38, 0.45, dx=-110)
    return img

def scene2(t):
    tl = t - T_S2
    p = clamp(tl / 6.1)
    img = tinted(IMG4_BLUR, 0.45).copy()
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse((200, 380, 900, 880), fill=(139, 225, 42, 70))
    overlay(img, glow.filter(ImageFilter.GaussianBlur(90)))
    overlay(img, GRAD_TOP, (0, 0), 0.8)
    f = ANTON(150)
    slide(img, text_layer("COMPANY", f), 60, 46, t, T_S2 + 0.15, 0.4)
    slide(img, text_layer("FUEL CARD", f, GREEN), 60, 200, t, T_S2 + 0.28, 0.4)
    sup = label("FOR CAR HAULER OWNER-OPERATORS", ANTON(44), WHITE, ORANGE, 28, 12, 14)
    slide(img, sup, 66, 385, t, T_S2 + 0.9, 0.45, dx=-110)
    # card entrance + gentle float
    e = eob(prog(t, T_S2 + 0.5, 0.9))
    ang = lerp(-26, -8, e) + math.sin(tl * 1.3) * 1.2
    sc = lerp(0.6, 0.9, e) * (1 + 0.012 * math.sin(tl * 0.9))
    c = CARD.resize((int(CARD.width * sc / 2), int(CARD.height * sc / 2)), Image.LANCZOS)
    c = c.rotate(ang, Image.BICUBIC, expand=True)
    shadow = Image.new("RGBA", c.size, (0, 0, 0, 0))
    shadow.putalpha(c.getchannel("A").point(lambda v: int(v * 0.55)))
    shadow = shadow.filter(ImageFilter.GaussianBlur(22))
    cx = lerp(1250, 540, eo5(prog(t, T_S2 + 0.5, 0.8))); cy = 672 + math.sin(tl * 1.6) * 8
    overlay(img, shadow, (cx - c.width / 2 + 10, cy - c.height / 2 + 28), 0.9)
    overlay(img, c, (cx - c.width / 2, cy - c.height / 2), clamp(prog(t, T_S2 + 0.5, 0.35)))
    return img

def map_bg(t, pr):
    img = vgrad(W, H, NAVY, NAVY2).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    for x in range(0, W, 72): d.line((x, 0, x, H), fill=(120, 160, 230, 20), width=1)
    for y in range(0, H, 72): d.line((0, y, W, y), fill=(120, 160, 230, 20), width=1)
    for pts in ([(-20, 700), (260, 620), (520, 760), (1100, 560)], [(300, -20), (420, 300), (380, 700), (520, 1100)],
                [(-20, 300), (500, 400), (800, 250), (1100, 330)]):
        d.line(pts, fill=(160, 190, 255, 38), width=14, joint="curve")
    route = [(90, 800), (250, 690), (420, 760), (600, 600), (760, 680), (900, 520), (990, 300)]
    # sample polyline
    seg = []
    for i in range(len(route) - 1):
        for k in range(30): seg.append((lerp(route[i][0], route[i + 1][0], k / 30), lerp(route[i][1], route[i + 1][1], k / 30)))
    n = max(2, int(len(seg) * pr)); seg = seg[:n]
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).line(seg, fill=(139, 225, 42, 200), width=22, joint="curve")
    overlay(img, glow.filter(ImageFilter.GaussianBlur(12)))
    d = ImageDraw.Draw(img, "RGBA")
    d.line(seg, fill=GREEN, width=9, joint="curve")
    for pt, col in ((route[0], WHITE), (seg[-1], ORANGE)):
        d.ellipse((pt[0] - 17, pt[1] - 17, pt[0] + 17, pt[1] + 17), fill=col, outline=NAVY, width=5)
    return img

def scene3(t):
    tl = t - T_S3
    img = map_bg(t, eo(prog(t, T_S3 + 0.1, 3.6)))
    # photo frame
    ph = view(IMG5, clamp(tl / 4.4), 1.15, 1.0, (0.5, 0.55), (0.5, 0.5), out=(960, 400))
    mask = Image.new("L", (960, 400), 0); ImageDraw.Draw(mask).rounded_rectangle((0, 0, 959, 399), 26, fill=255)
    fr = ph.convert("RGBA"); fr.putalpha(mask)
    border = rrect(968, 408, 30, (0, 0, 0, 0), outline=GREEN, ow=4)
    e = eo5(prog(t, T_S3 + 0.05, 0.5))
    py = 50 - (1 - e) * 90
    overlay(img, fr, (60, py), e); overlay(img, border, (56, py - 4), e)
    chips = [("LOAD SEARCH", 10.85), ("ROUTE PLANNING", 12.0), ("DISPATCH SUPPORT", 13.15)]
    for i, (txt, t0) in enumerate(chips):
        w, h = 960, 84
        ch = rrect(w, h, 20, (6, 16, 40, 215), outline=(139, 225, 42, 255), ow=3)
        ImageDraw.Draw(ch).rounded_rectangle((14, 14, 22, h - 15), 4, fill=GREEN)
        ch.alpha_composite(text_layer(txt, ANTON(54), WHITE, shadow=False, pad=0), (50, 12))
        dot = Image.new("RGBA", (44, 44), (0, 0, 0, 0)); dd = ImageDraw.Draw(dot)
        dd.ellipse((0, 0, 43, 43), fill=GREEN)
        dd.line((11, 23, 19, 31, 33, 14), fill=(10, 20, 10), width=6)
        ch.alpha_composite(dot, (w - 76, 20))
        slide(img, ch, 60, 485 + i * 100, t, t0 - 0.12, 0.45, dx=-140)
    return img

def laptop(t0, t):
    w, h = 720, 440
    im = Image.new("RGBA", (w, h + 40), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), 22, fill=(20, 24, 33, 255), outline=(70, 78, 96, 255), width=4)
    d.rounded_rectangle((18, 18, w - 19, h - 19), 10, fill=(10, 24, 56, 255))
    d.rectangle((18, 18, 170, h - 19), fill=(8, 18, 42, 255))
    for i in range(5):
        d.rounded_rectangle((34, 44 + i * 46, 154, 72 + i * 46), 8, fill=(30, 52, 100, 255) if i else GREEN)
    # map area with route
    d.rectangle((190, 40, w - 40, 250), fill=(14, 36, 80, 255))
    pr = eo(prog(t, t0 + 0.3, 1.2))
    pts = [(215, 220), (310, 150), (400, 195), (500, 100), (600, 150), (680, 70)]
    n = max(2, int(len(pts) * 0 + 2 + pr * (len(pts) - 2)))
    d.line(pts[:n], fill=GREEN, width=6, joint="curve")
    d.ellipse((205, 210, 225, 230), fill=WHITE); d.ellipse((pts[n - 1][0] - 10, pts[n - 1][1] - 10, pts[n - 1][0] + 10, pts[n - 1][1] + 10), fill=ORANGE)
    # generic rows (no figures)
    for i in range(3):
        y = 275 + i * 46
        d.rounded_rectangle((190, y, w - 40, y + 34), 8, fill=(18, 40, 88, 255))
        d.ellipse((204, y + 9, 220, y + 25), fill=GREEN if i != 1 else ORANGE)
        d.rounded_rectangle((236, y + 11, 236 + 180 - i * 20, y + 23), 5, fill=(70, 100, 160, 255))
        d.rounded_rectangle((w - 160, y + 11, w - 62, y + 23), 5, fill=(50, 78, 130, 255))
    d.rounded_rectangle((-30, h - 4, w + 30, h + 30), 14, fill=(150, 158, 175, 255))
    return im

def phone():
    w, h = 210, 400
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), 34, fill=(16, 18, 26, 255), outline=(90, 98, 118, 255), width=4)
    d.rounded_rectangle((14, 20, w - 15, h - 20), 22, fill=(10, 24, 56, 255))
    d.ellipse((w / 2 - 38, 70, w / 2 + 38, 146), fill=GREEN)
    d.arc((w / 2 - 18, 90, w / 2 + 18, 126), 200, 340, fill=(10, 20, 10), width=7)  # handset-ish
    d.rounded_rectangle((38, 175, w - 38, 205), 12, fill=(60, 90, 150, 255))
    d.rounded_rectangle((38, 225, w - 70, 255), 12, fill=(40, 66, 120, 255))
    d.rounded_rectangle((38, 300, w - 38, 346), 22, fill=ORANGE)
    return im
PHONE = phone()

def scene4(t):
    tl = t - T_S4
    img = map_bg(t, 1.0).filter(ImageFilter.GaussianBlur(3))
    img = tinted(img, 0.35)
    f = ANTON(112)
    slide(img, text_layer("DEDICATED", f), 56, 36, t, T_S4 + 0.0, 0.4)
    slide(img, text_layer("DISPATCH SUPPORT", f, GREEN), 56, 170, t, T_S4 + 0.1, 0.4)
    sub = label("CAR HAULERS ACROSS THE USA", ANTON(46), WHITE, ORANGE, 28, 12, 14)
    slide(img, sub, 62, 330, t, T_S4 + 0.45, 0.45, dx=-110)
    lp = laptop(T_S4, t)
    e = eo5(prog(t, T_S4 + 0.15, 0.6))
    overlay(img, lp, (90, 455 + (1 - e) * 120 + math.sin(tl * 1.4) * 4), e)
    e2 = eob(prog(t, T_S4 + 0.5, 0.6))
    ph = PHONE.rotate(-8, Image.BICUBIC, expand=True)
    overlay(img, ph, (710 + (1 - e2) * 260, 440 + math.sin(tl * 1.7 + 1) * 5), clamp(e2))
    return img

def scene5(t):
    tl = t - T_S5
    img = vgrad(W, H, NAVY, NAVY2).convert("RGB")
    # hauler photo anchored to the bottom, fading up into the navy
    z = lerp(1.0, 1.12, clamp(tl / 5.0)); cw = 1900 / z; ch = cw * 640 / 1080
    cx, cy = IMG4.width * lerp(0.50, 0.54, clamp(tl / 5.0)), IMG4.height * 0.52
    ph = IMG4.resize((W, 640), Image.BICUBIC, box=(cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2))
    fade = Image.fromarray(np.repeat(np.clip(np.linspace(0, 1.6, 640), 0, 1)[:, None] * 255, W, 1).astype("uint8"), "L")
    img.paste(ph, (0, 440), fade)
    overlay(img, GRAD_BOT, (0, H - 420), 0.7)
    lw = 400; lh = round(LOGO.height * lw / LOGO.width)
    lg = LOGO.resize((lw, lh), Image.LANCZOS)
    plate = rrect(lw + 48, lh + 30, 24, (255, 255, 255, 255)); plate.alpha_composite(lg, (24, 15))
    e = eo5(prog(t, T_S5 + 0.1, 0.5))
    overlay(img, plate, (centered(plate), 50 - (1 - e) * 40), e)
    bw, bh = 900, 150
    btn = rrect(bw, bh, 30, ORANGE, outline=(255, 170, 110, 255), ow=3)
    tx = text_layer("GET THE DETAILS", ANTON(98), WHITE, shadow=False, pad=0)
    btn.alpha_composite(tx, ((bw - tx.width) // 2, (bh - tx.height) // 2 - 2))
    eb = eob(prog(t, T_S5 + 0.7, 0.7))
    pulse = 1 + 0.012 * math.sin(max(0, tl - 1.6) * 3.2)
    sc = lerp(0.88, 1.0, clamp(eb, 0, 1.05)) * pulse
    b2 = btn.resize((int(bw * sc), int(bh * sc)), Image.LANCZOS)
    glow = Image.new("RGBA", b2.size, (0, 0, 0, 0)); glow.putalpha(b2.getchannel("A").point(lambda v: int(v * 0.5)))
    glow = glow.filter(ImageFilter.GaussianBlur(24))
    ea = clamp(prog(t, T_S5 + 0.7, 0.45))
    bx, by = (W - b2.width) // 2, 275 + (1 - clamp(eb, 0, 1)) * 40
    overlay(img, glow, (bx, by + 14), ea * 0.8); overlay(img, b2, (bx, by), ea)
    sup = text_layer("LEAVE YOUR INFO — WE'LL CONTACT YOU", ANTON(50), WHITE)
    slide(img, sup, centered(sup), 455, t, T_S5 + 1.25, 0.5, dx=0, dy=40)
    return img

SCENES = [(0.0, scene1), (T_S2, scene2), (T_S3, scene3), (T_S4, scene4), (T_S5, scene5)]

def frame(t):
    idx = max(i for i, (s, _) in enumerate(SCENES) if t >= s - 1e-6)
    img = SCENES[idx][1](t)
    # cross-fade into this scene at its start; and out of previous (render previous at same t)
    start = SCENES[idx][0]
    if idx > 0 and t < start + XF:
        prev = SCENES[idx - 1][1](t)
        a = eo(prog(t, start, XF))
        img = Image.blend(prev, img, a)
    draw_subtitles(img, t)
    return img

if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "build", "preview"), exist_ok=True)
    if "--frames" in sys.argv:
        for fr in map(int, sys.argv[sys.argv.index("--frames") + 1:]):
            frame(fr / FPS).save(os.path.join(HERE, "build", "preview", f"f{fr:04d}.png"))
        sys.exit()
    n = int(DUR * FPS)
    out = os.path.join(HERE, "build", "video_silent.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
                           "-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart", out],
                          stdin=subprocess.PIPE)
    for i in range(n):
        ff.stdin.write(frame(i / FPS).tobytes())
        if i % 60 == 0: print(f"{i}/{n}", flush=True)
    ff.stdin.close(); ff.wait()

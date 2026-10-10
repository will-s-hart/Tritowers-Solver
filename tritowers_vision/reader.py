"""Photo reader v3: photo or screenshot in, solver draft out.

  1. register.register: find the card layout in the image (no screen border or exact crop needed).
  2. scene.infer_presence (inside registration): which of the 28 cards are still on the table, using the game rules,
     so face up / covered / empty are consistent by construction.
  3. glyphs: read the index of every exposed card and of the waste card from the original pixels.
  4. Deck check (at most four of a rank) and review flags. Nothing hidden is ever inferred.

The draft keeps the calibrated.read contract (cards / needs_human_review / complete / stock_counter / note), so callers
can switch without other changes.
"""
from collections import Counter
from dataclasses import dataclass
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from . import glyphs, scene
from .image import normalize_image
from .register import Registration, register

WASTE_SCALE = 1.2                 # waste index size relative to a tableau index (measured 39 vs 33 units)
PRESENCE_REVIEW = 0.2             # presence evidence below this share of a card is flagged for review
NOTE = "Draft only. Suit not read. Stock counter not read. Human must confirm before solving."


@dataclass
class PhotoRead:
    draft: dict
    overlay: Image.Image
    rectified: Image.Image
    registration: Registration


def _load(source):
    return normalize_image(source)


def _state(present, p):
    if p not in present: return "empty"
    return "face_up" if scene.exposed(present, p) else "covered"


def read_photo(source, templates=(), manual_corners=None):
    """Read a photo or screenshot (path, bytes, file object or PIL image). templates: private photo glyphs, optional."""
    image = _load(source)
    reg = register(image, manual_corners)
    rgb = np.asarray(image)
    cards, found = {}, {}
    for p in range(1, 29):
        state = _state(reg.present, p)
        cards[f"tableau-{p:02d}"] = {"state": state, "rank": None, "score": 0.0, "margin": 0.0, "presence": round(float(reg.margins.get(p, 0.0)), 2)}
        if state == "face_up":
            slot = f"tableau-{p:02d}"
            found[slot], cards[slot]["crop"] = glyphs.extract(glyphs.corner_patch(rgb, reg.H, scene.RECTS[p][:2]), 1.0)
    cards["waste"] = {"state": "face_up", "rank": None, "score": 0.0, "margin": 0.0}
    found["waste"], cards["waste"]["crop"] = glyphs.extract(glyphs.corner_patch(rgb, reg.H, scene.waste_rect(reg.waste_dy)[:2], glyphs.WASTE_CORNER), WASTE_SCALE)
    counters = {slot: cards[slot]["crop"].get("counters") for slot in found if cards[slot]["crop"].get("counters") is not None}
    ranks, fit = glyphs.read_ranks(found, templates, counters)
    if fit >= glyphs.FONT_FIT:
        ranks = glyphs.agree(found, ranks, counters)
    for slot, (rank, score, margin, tier) in ranks.items():
        cards[slot].update(rank=rank, score=round(score, 2), margin=round(margin, 2), tier=tier)
    review = set()
    counts = Counter(c["rank"] for c in cards.values() if c["rank"])
    for r, n in counts.items():                      # a real deck has four of each rank: drop the weakest extra reads
        if n > 4:
            weakest = sorted((c["score"], s) for s, c in cards.items() if c["rank"] == r)[:n - 4]
            for _, s in weakest: cards[s]["rank"] = None; cards[s]["tier"] = "deck_conflict"
    for s, c in cards.items():
        if c["state"] == "face_up" and c["rank"] is None: review.add(s)
        if s != "waste" and abs(c.get("presence", 1.0)) < PRESENCE_REVIEW: review.add(s)
    if not reg.trusted:                              # layout not found reliably: no rank is trustworthy either
        review.update(cards)
        for c in cards.values():
            if c["rank"]: c["rank"] = None; c["tier"] = "untrusted_layout"
            c["inferred_state"] = c["state"]
            c["state"] = "unknown"
    order = list(cards)
    draft = {"cards": cards, "needs_human_review": [s for s in order if s in review], "complete": not review,
             "stock_counter": None, "note": NOTE,
             "registration": {"method": reg.method, "quality": round(reg.quality, 3), "trusted": reg.trusted, "checks": reg.checks},
             "font_fit": round(fit, 3)}
    rectified = Image.fromarray(cv2.warpPerspective(rgb, reg.H, scene.FRAME, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP))
    return PhotoRead(draft, overlay(image, reg, cards, review), rectified, reg)


def board_tokens(draft):
    """Solver board text tokens and waste: rank, '?' (covered or not read), '--' (empty). Nothing is guessed."""
    tokens = []
    for i in range(1, 29):
        c = draft["cards"][f"tableau-{i:02d}"]
        tokens.append("--" if c["state"] == "empty" else (c["rank"] or "?"))
    return tokens, draft["cards"]["waste"]["rank"] or ""


def labelled_glyphs(source, board, waste):
    """(rank, glyph) templates from a labelled image (board: 28 tokens). Used to build private photo templates."""
    image = _load(source); reg = register(image); rgb = np.asarray(image); out = []
    if not reg.trusted:
        return out
    for p, tok in enumerate(board, 1):
        if tok in glyphs.RANKS:
            g, _ = glyphs.extract(glyphs.corner_patch(rgb, reg.H, scene.RECTS[p][:2]), 1.0)
            if g is not None and g.sum(): out.append((tok, g))
    if waste in glyphs.RANKS:
        g, _ = glyphs.extract(glyphs.corner_patch(rgb, reg.H, scene.waste_rect(reg.waste_dy)[:2], glyphs.WASTE_CORNER), WASTE_SCALE)
        if g is not None and g.sum(): out.append((waste, g))
    return out


_COLOURS = {"read": (40, 200, 90), "review": (255, 200, 0), "covered": (200, 40, 60), "empty": (150, 150, 150)}


def overlay(image, reg, cards, review, max_side=1280):
    """The input image with every slot outlined: green read, yellow check this, red covered, grey empty."""
    f = min(1.0, max_side / max(image.size))
    im = image.resize((round(image.width * f), round(image.height * f))) if f < 1 else image.copy()
    d = ImageDraw.Draw(im); H = np.diag([f, f, 1.0]) @ reg.H
    unit = float(np.linalg.norm(scene.project(H, [(512, 400), (612, 400)])[1] - scene.project(H, [(512, 400)])[0])) / 100
    try: font = ImageFont.load_default(size=max(12, int(22 * unit)))
    except TypeError: font = ImageFont.load_default()
    items = [(f"tableau-{p:02d}", scene.RECTS[p], p >= 19 or scene.exposed(reg.present, p)) for p in range(1, 29)]
    items.append(("waste", scene.waste_rect(reg.waste_dy), True))
    for slot, (x0, y0, x1, y1), full in items:
        c = cards[slot]; y1 = y1 if full else y0 + 80
        kind = "review" if slot in review else ("read" if c["state"] == "face_up" else c["state"])
        poly = [tuple(q) for q in scene.project(H, [(x0 + 3, y0 + 3), (x1 - 3, y0 + 3), (x1 - 3, y1 - 3), (x0 + 3, y1 - 3)])]
        d.polygon(poly, outline=_COLOURS.get(kind, (255, 255, 255)), width=max(2, int(3 * unit)))
        label = c["rank"] or ("?" if c["state"] == "face_up" else "")
        if label:
            tx, ty = scene.project(H, [(x0 + 52, y0 + 8)])[0]
            d.text((tx, ty), label, font=font, fill=_COLOURS[kind], stroke_width=max(1, int(2 * unit)), stroke_fill=(0, 0, 0))
    if not reg.trusted:
        d.text((10, 10), "Layout not found reliably: check every card", font=font, fill=(255, 200, 0), stroke_width=2, stroke_fill=(0, 0, 0))
    return im

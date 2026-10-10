"""Rank index glyphs read straight from the original image through the layout homography.

Compared with rank2 (fixed crops of the 1024x768 rectified frame, fixed grey-level ink threshold): the index corner is
warped once from the full-resolution photo, ink is measured against the card's own face colour (so exposure, colour
casts and red or black ink all work the same), and blobs touching the patch edge (neighbouring cards, backs behind,
parchment) are dropped. Glyphs are normalised exactly like rank2 (32 x 40, aspect kept), so templates stay compatible.

Matching: private same-skin photo templates first, then a bundled bank rendered from open fonts; enclosed-hole counts
penalise impossible ranks; every gate abstains rather than guesses (thresholds measured on the five pilot screenshots
and simulated re-shoots of them, tools/vision_eval.py).
"""
from pathlib import Path
import os
import numpy as np
import cv2

from . import rank2
from .rank2 import GH, GW, RANKS, normalize

PX = 3                                   # output pixels per layout unit for the corner patch
GLYPH_H = 37                             # rank band height for a tableau index, layout units (real ranks are 32-35)
TABLEAU_CORNER = (-5, -5, 52, 58)        # patch around a tableau card's top-left corner, layout units (Q tails run low)
WASTE_CORNER = (-6, -6, 62, 78)          # the waste index is larger
BANK = Path(__file__).with_name("data") / "font_glyphs.npz"
# Acceptance gates: (best score, margin over the best other rank) for each tier.
PHOTO_SCORE, PHOTO_MARGIN = 0.85, rank2.PHOTO_MARGIN   # 0.85: no wrong accepts when a rank has no template (leave-one-out)
FONT_SCORE, FONT_MARGIN = 0.75, 0.10    # no wrong accepts on 465 real and re-shot pilot glyphs, nor on held-out synthetic fonts
FONT_FIT = 0.75                          # below this image-level fit the font tier names nothing (see font_fit)
_BANK = None


def corner_patch(image_rgb, H, card_xy, box=TABLEAU_CORNER):
    """RGB patch of the index corner of the card whose top-left layout point is card_xy. H: layout -> image pixels."""
    x0, y0 = card_xy[0] + box[0], card_xy[1] + box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    pts = np.float64([[x0, y0], [x0 + w, y0], [x0, y0 + h]])
    img_pts = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    s = max(np.linalg.norm(img_pts[1] - img_pts[0]) / w, np.linalg.norm(img_pts[2] - img_pts[0]) / h)
    hi = max(PX, int(np.ceil(s)))                 # sample at least at the image's own resolution, then average down
    T = np.array([[1 / hi, 0, x0], [0, 1 / hi, y0], [0, 0, 1]], np.float64)
    patch = cv2.warpPerspective(image_rgb, H @ T, (w * hi, h * hi), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REPLICATE)
    if hi != PX: patch = cv2.resize(patch, (w * PX, h * PX), interpolation=cv2.INTER_AREA)
    return patch


def ink_distance(patch):
    """Colour distance of every pixel from the card face, and the ink threshold for it."""
    a = patch.astype(np.float32)
    v = a.max(axis=2); chroma = (v - a.min(axis=2)) / (v + 1)
    face_px = (v >= np.percentile(v, 60)) & (chroma < 0.35)
    face = np.median(a[face_px], axis=0) if face_px.sum() > 20 else np.percentile(a.reshape(-1, 3), 90, axis=0)
    d = cv2.GaussianBlur(np.sqrt(((a - face) ** 2).sum(axis=2)), (0, 0), 0.8)    # calms moire and JPEG noise
    d8 = np.clip(d, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return d, max(45.0, min(float(otsu), 0.45 * float(np.linalg.norm(face))))


def ink_mask(patch):
    """Pixels that differ from the card face colour (rank, pip, anything that is not the face)."""
    d, thr = ink_distance(patch)
    return (d > thr).astype(np.uint8)


def raw_glyph(patch, scale=1.0, *, with_box=False):
    """Binary rank glyph (variable size) or None. scale: index size relative to a tableau card.

    The rank occupies a band from its top down to the first clear gap (the space above the suit pip; smaller indexes
    on other skins put the pip well inside a fixed band), at most GLYPH_H; every blob starting inside the band belongs
    to it (blur and moire can split a stroke), and the band's bottom edge cuts off a pip that has merged into it."""
    ink = ink_mask(patch)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    H, W = ink.shape
    unit = PX * scale
    cands = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < 4 * unit * unit: continue                                          # specks
        if x <= 0 or y <= 0 or x + w >= W or y + h >= H: continue                   # touches the patch edge
        if h > 50 * unit or w > 36 * unit: continue                                 # pips, art, card edges
        cands.append(i)
    tops = [st[i, cv2.CC_STAT_TOP] for i in cands if st[i, cv2.CC_STAT_HEIGHT] >= 15 * unit]
    if not tops: return None
    top = min(tops); bottom = top + int(GLYPH_H * unit)
    first = min((i for i in cands if st[i, cv2.CC_STAT_TOP] == top), key=lambda i: st[i, cv2.CC_STAT_LEFT])
    left = st[first, cv2.CC_STAT_LEFT]
    column = [i for i in cands if top <= st[i, cv2.CC_STAT_TOP] < bottom
              and st[i, cv2.CC_STAT_LEFT] < left + 30 * unit
              and (st[i, cv2.CC_STAT_LEFT] >= left - 3 * unit
                   or (st[i, cv2.CC_STAT_HEIGHT] >= .65 * st[first, cv2.CC_STAT_HEIGHT]
                       and st[i, cv2.CC_STAT_WIDTH] >= 3 * unit))]
    rows = np.isin(lab, column)[top:bottom].any(axis=1)
    gap = max(2, int(round(2 * unit)))
    for y in range(int(0.45 * GLYPH_H * unit), len(rows) - gap):     # first empty run past a plausible glyph height
        if not rows[y:y + gap].any(): bottom = top + y; break
    keep = [i for i in column if st[i, cv2.CC_STAT_TOP] < bottom - min(4 * unit, 0.2 * (bottom - top))]
    m = np.isin(lab, keep)[top:bottom].astype(np.float32); ys, xs = np.nonzero(m)
    if not len(xs): return None
    raw = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    box = (int(xs.min()), int(top + ys.min()), int(xs.max() + 1), int(top + ys.max() + 1))
    return (raw, box) if with_box else raw


def glyph(patch, scale=1.0):
    return normalize(raw_glyph(patch, scale))



def extract(patch, scale=1.0):
    """A glyph and separate crop-validity evidence, before any rank matching.

    Reject blank/partial crops and suit pips exposed by losing the rank to a
    connected card edge. A high match score cannot override ownership checks.
    """
    result = raw_glyph(patch, scale, with_box=True)
    if result is None:
        return None, {"valid": False, "reason": "no_complete_index"}
    raw, (x0, y0, x1, y1) = result
    unit = PX * scale
    height = (y1 - y0) / unit
    # A browser-resampled photo can register a card ~10 units low; the index
    # is still whole (the bottom check) and far above the suit pip (~50).
    valid = bool(15 <= height <= 40 and 3 <= x0 / unit <= 30 and 3 <= y0 / unit <= 26
                 and x1 < patch.shape[1] - 2 and y1 < patch.shape[0] - 2)
    detail = {"valid": valid, "reason": "index_inside_card" if valid else "partial_or_displaced_index",
              "box": [x0, y0, x1, y1]}
    if valid:
        detail["counters"] = raw_holes(raw > .5)
    return (normalize(raw) if valid else None), detail

def _render_bank(fonts):
    from PIL import Image, ImageDraw, ImageFont
    out = []
    for path in fonts:
        try: font = ImageFont.truetype(str(path), 72)
        except OSError: continue
        try: font.set_variation_by_name("Bold")
        except Exception: pass
        for r in RANKS:
            for squeeze in ((1.0, 0.8) if r == "10" else (1.0,)):
                im = Image.new("L", (260, 140), 0); ImageDraw.Draw(im).text((10, 10), r, font=font, fill=255)
                a = np.array(im) > 128; ys, xs = np.nonzero(a)
                if not len(xs): continue
                g = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
                if squeeze != 1.0: g = cv2.resize(g, (max(1, int(g.shape[1] * squeeze)), g.shape[0]), interpolation=cv2.INTER_AREA)
                out.append((r, normalize(g)))
    return out


def font_bank():
    """Bundled open-font glyphs (data/font_glyphs.npz), else rank2's runtime font templates."""
    global _BANK
    if _BANK is None:
        if BANK.exists():
            z = np.load(BANK, allow_pickle=False)
            _BANK = [(str(r), g.astype(np.float32) / 255.0) for r, g in zip(z["ranks"], z["glyphs"])]
        else:
            _BANK = list(rank2.synthetic_templates())
    return _BANK


# Enclosed holes per rank index (8 has two, A 4 6 9 Q 10 one, the rest none; this skin and the bank draw a closed 4). Counting
# holes separates the curly 3 from 8 and 5 from 6, which shape correlation alone only just tells apart.
HOLES = {"A": {1}, "2": {0}, "3": {0}, "4": {1}, "5": {0}, "6": {1}, "7": {0}, "8": {2}, "9": {1}, "10": {1}, "J": {0}, "Q": {1}, "K": {0}}
HOLE_PENALTY = 0.12


def holes(g, thr=0.5):
    """Number of enclosed background regions in a normalised glyph (specks under 3 px ignored)."""
    inv = np.pad((g <= thr).astype(np.uint8), 1, constant_values=1)
    n, _, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    return sum(1 for i in range(2, n) if st[i, cv2.CC_STAT_AREA] >= 3)


def raw_holes(mask):
    """Enclosed counters of a full-resolution glyph mask.

    Shrinking a glyph to GH rows can close a narrow opening (the curl of a
    3) into a false counter; counting at the extracted resolution avoids it.
    Specks under 5% of the glyph height across are ignored.
    """
    inv = np.pad((~np.asarray(mask, bool)).astype(np.uint8), 1, constant_values=1)
    n, _, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    least = max(3.0, (0.05 * mask.shape[0]) ** 2)
    return sum(1 for i in range(2, n) if st[i, cv2.CC_STAT_AREA] >= least)


# Only the immutable, public font bank is cached. Private inputs and in-image
# examples live for one read, so switching photos cannot contaminate later reads.
_FONT_MATRIX = None


def _prepare(templates):
    ranks, rows = [], []
    for rank, template in templates:
        if template.shape != (GH, GW):
            continue
        blurred = cv2.GaussianBlur(template, (0, 0), 0.8)
        for shifted in rank2._shifts(blurred):
            centered = shifted.ravel() - shifted.mean()
            rows.append(centered)
            ranks.append(rank)
    if not rows:
        return [], np.empty((0, GH * GW), np.float32), np.empty(0, np.float32)
    matrix = np.asarray(rows, np.float32)
    return ranks, matrix, np.linalg.norm(matrix, axis=1)


def _prepared(templates):
    global _FONT_MATRIX
    if templates is font_bank():
        if _FONT_MATRIX is None:
            _FONT_MATRIX = _prepare(templates)
        return _FONT_MATRIX
    return _prepare(templates)


def _ranked(g, templates, counters=None):
    """Best score per rank; counters overrides the hole count measured on g."""
    ranks, matrix, norms = _prepared(templates)
    if not ranks:
        return []
    blurred = cv2.GaussianBlur(g, (0, 0), 0.8)
    query = blurred.ravel() - blurred.mean()
    # einsum avoids a multithreaded BLAS launch for these small, frequent dots.
    scores = np.einsum("ij,j->i", matrix, query, dtype=np.float64) / (norms * np.linalg.norm(query) + 1e-6)
    h, best = holes(g) if counters is None else counters, {}
    for rank, score in zip(ranks, scores):
        value = float(score) - (HOLE_PENALTY if h not in HOLES.get(rank, {h}) else 0)
        best[rank] = max(best.get(rank, -1.0), value)
    return sorted(best.items(), key=lambda kv: -kv[1])


def _left_aligned(a, width):
    """A glyph scaled to GH rows, its left edge at a fixed column, cut at width columns."""
    out = np.zeros((GH, GW), np.float32)
    cols = np.nonzero(a.max(axis=0) > .05)[0]
    if not len(cols):
        return out
    a = a[:, cols.min():cols.max() + 1][:, :width]
    left = (GW - width) // 2
    out[:, left:left + a.shape[1]] = a
    return out


_CLIPPED = {}


def _clipped_bank(width):
    """Font templates cut at one visible width: ranks, unit shifted rows, hole counts (cached)."""
    if width not in _CLIPPED:
        ranks, rows, counts = [], [], []
        for rank, template in font_bank():
            cut = _left_aligned(template, width)
            shifted = np.asarray([s.ravel() - s.mean() for s in rank2._shifts(cv2.GaussianBlur(cut, (0, 0), 0.8))], np.float32)
            ranks.append(rank); counts.append(holes(cut))
            rows.append(shifted / (np.linalg.norm(shifted, axis=1, keepdims=True) + 1e-6))
        _CLIPPED[width] = (ranks, np.stack(rows), np.asarray(counts))
    return _CLIPPED[width]


def warm():
    """Prepare the font matrix and the clipped banks for the widths a deck index can show."""
    _prepared(font_bank())
    for width in range(12, GW + 1):
        _clipped_bank(width)


def clipped_ranked(raw, templates=None):
    """Rank a glyph whose right side is hidden by an overlapping card.

    Every template (the font bank, or same-image examples) is cut at the same
    visible width as the query, so a partial glyph is compared only with the
    visible part of each rank; a template cannot lose for strokes the photo
    cannot show. Hole counts are taken from the cut template, so an opened
    zero is not penalised.
    """
    h, w = raw.shape
    width = max(1, min(GW, int(round(w * GH / h))))
    query = np.zeros((GH, GW), np.float32)
    query[:, (GW - width) // 2:(GW - width) // 2 + width] = cv2.resize(
        raw.astype(np.float32), (width, GH), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(query, (0, 0), 0.8)
    q = blurred.ravel() - blurred.mean()
    qn, qh, best = np.linalg.norm(q) + 1e-6, raw_holes(raw > .5), {}
    if templates is None:
        ranks, rows, counts = _clipped_bank(width)
        scores = np.einsum("tsj,j->ts", rows, q / qn, dtype=np.float64).max(axis=1) - HOLE_PENALTY * (counts != qh)
        for rank, score in zip(ranks, scores):
            best[rank] = max(best.get(rank, -1.0), float(score))
        return sorted(best.items(), key=lambda kv: -kv[1])
    for rank, template in templates:
        cut = _left_aligned(template, width)
        rows = np.asarray([s.ravel() - s.mean() for s in rank2._shifts(cv2.GaussianBlur(cut, (0, 0), 0.8))], np.float32)
        score = float(np.max(rows @ q / (np.linalg.norm(rows, axis=1) * qn + 1e-6)))
        score -= HOLE_PENALTY if holes(cut) != qh else 0
        best[rank] = max(best.get(rank, -1.0), score)
    return sorted(best.items(), key=lambda kv: -kv[1])


def ranked_many(gs, counters=None):
    """_ranked against the font bank for many glyphs with one matrix product."""
    ranks, matrix, norms = _prepared(font_bank())
    if not len(gs):
        return []
    blurred = np.stack([cv2.GaussianBlur(g, (0, 0), 0.8).ravel() for g in gs])
    query = blurred - blurred.mean(axis=1, keepdims=True)
    scores = (query @ matrix.T) / (np.linalg.norm(query, axis=1)[:, None] * norms[None] + 1e-6)
    names = sorted(set(ranks), key=RANKS.index)
    columns = {name: np.array([i for i, r in enumerate(ranks) if r == name]) for name in names}
    best = {name: scores[:, idx].max(axis=1) for name, idx in columns.items()}
    out = []
    for k, g in enumerate(gs):
        h = holes(g) if counters is None or counters[k] is None else counters[k]
        row = {name: float(best[name][k]) - (HOLE_PENALTY if h not in HOLES.get(name, {h}) else 0) for name in names}
        out.append(sorted(row.items(), key=lambda kv: -kv[1]))
    return out


def font_tier_enabled():
    """TT_FONT_TIER=0 turns the bundled font tier off, so only private photo templates can name a rank."""
    return os.environ.get("TT_FONT_TIER", "1").strip() != "0"


def match(g, photo_templates=(), *, font_scores=None, counters=None):
    """(rank or None, score, margin, tier). Photo templates (same skin) first, then the bundled font bank.

    The photo tier's margin is taken against all 13 ranks: a rank with no photo template competes through its font
    score, so a glyph whose true rank has no template cannot win just because its rivals are missing."""
    if g is None or g.sum() == 0: return None, 0.0, 0.0, "none"
    font = dict(_ranked(g, font_bank(), counters)) if font_scores is None else font_scores
    photo = dict(_ranked(g, photo_templates, counters)) if len(photo_templates) else {}
    best = (0.0, 0.0)
    if photo:
        top = max(photo, key=photo.get)
        rivals = [photo.get(r, font.get(r, -1.0)) for r in RANKS if r != top]
        m = photo[top] - max(rivals)
        if photo[top] >= PHOTO_SCORE and m >= PHOTO_MARGIN: return top, photo[top], m, "photo"
        best = (photo[top], m)
    if font and font_tier_enabled():
        order = sorted(font.items(), key=lambda kv: -kv[1]); m = order[0][1] - order[1][1]
        if order[0][1] >= FONT_SCORE and m >= FONT_MARGIN: return order[0][0], order[0][1], m, "font"
        best = (order[0][1], m)
    return None, best[0], best[1], "abstain"


def font_fit(found):
    """Median best font-bank score over the glyphs of one image: about 0.86 on the calibrated skin, far lower when the
    skin's index font is unlike every bundled font (then no rank should be named from fonts)."""
    tops = [_ranked(g, font_bank())[0][1] for g in found if g is not None and g.sum()]
    return float(np.median(tops)) if tops else 0.0


def read_ranks(found, templates=(), counters=None):
    """Rank every glyph of one image: {key: (rank or None, score, margin, tier)}.

    counters optionally gives each glyph's hole count measured before scaling.

    1. match() each glyph; font-tier reads are dropped when the image's font fit is below FONT_FIT.
    2. Confident reads become same-skin templates for the glyphs that abstained (a board repeats ranks, and copies of
       a rank on one screen are near-identical); only the photo-tier gate can accept them."""
    templates, counters = list(templates), counters or {}
    usable = [k for k, g in found.items() if g is not None and g.sum()]
    batch = dict(zip(usable, ranked_many([found[k] for k in usable], [counters.get(k) for k in usable])))
    font_scores = {k: dict(batch[k]) if k in batch else {} for k in found}
    tops = [max(scores.values()) for scores in font_scores.values() if scores]
    fit = float(np.median(tops)) if tops else 0.0
    out = {}
    for k, g in found.items():
        r = match(g, templates, font_scores=font_scores[k], counters=counters.get(k))
        if r[3] == "font" and fit < FONT_FIT: r = (None, r[1], r[2], "font_unfit")
        out[k] = r
    local = [(r[0], found[k]) for k, r in out.items() if r[0]]
    for k, g in found.items():
        if out[k][0] is None and local:
            r = match(g, templates + local, font_scores=font_scores[k], counters=counters.get(k))
            if r[3] == "photo": out[k] = (r[0], r[1], r[2], "same_image")
    return out, fit


def consensus(keys, reads, similar, ranking, vetoed=None):
    """Name a rank that no single crop proves but several identical crops do.

    Copies of one rank on a photo are near-identical. Unread crops whose
    font ranking puts the same rank R first (at the font score) and that
    look alike (photo score for every pair) are named R together, but only
    when R is not yet read on this photo and every rival within the font
    margin has been read here and looks unlike them by the photo margin. Two
    separate groups for one rank cancel out. Nothing is inferred from how
    many cards of a rank remain. similar(a, b) and ranking(a) take keys.
    """
    seen = {}
    for key, read in reads.items():
        if read[0] and key in keys:
            seen.setdefault(read[0], []).append(key)
    candidates = {}
    for key in keys:
        if reads[key][0] is None:
            ranked = ranking(key)
            if len(ranked) > 1 and ranked[0][1] >= FONT_SCORE and ranked[0][0] not in seen:
                candidates[key] = ranked
    named = {}
    for rank in sorted({ranked[0][0] for ranked in candidates.values()}):
        members = sorted(key for key, ranked in candidates.items() if ranked[0][0] == rank)
        if vetoed:
            members = [key for key in members if not vetoed(key, rank)]
        pair = {(a, b): similar(a, b) for a in members for b in members if a < b}
        look = lambda a, b: pair[(a, b) if a < b else (b, a)]
        group = [a for a in members if all(look(a, b) >= PHOTO_SCORE for b in members if b != a)]
        if len(group) < 2:
            continue
        cohesion = min(look(a, b) for a in group for b in group if a < b)
        if all(rival in seen and max(similar(key, other) for other in seen[rival]) <= cohesion - PHOTO_MARGIN
               for key in group for rival, score in candidates[key][1:] if score > candidates[key][0][1] - FONT_MARGIN):
            for key in group:
                top, second = candidates[key][0][1], candidates[key][1][1]
                named[key] = (rank, top, top - second, "same_image_consensus")
    return named


def exclusion(keys, reads, similar, ranking, vetoed=None):
    """Name a lone crop when every close font rival is ruled out by this photo.

    A rival within the font margin is ruled out only when it has been read
    on this photo and the crop looks unlike every copy of it by the photo
    margin below the photo score. The named rank must not be read here yet.
    """
    seen = {}
    for key, read in reads.items():
        if read[0] and key in keys:
            seen.setdefault(read[0], []).append(key)
    named = {}
    for key in keys:
        if reads[key][0] is not None:
            continue
        ranked = ranking(key)
        if len(ranked) < 2 or ranked[0][1] < FONT_SCORE or ranked[0][0] in seen:
            continue
        rank, top = ranked[0]
        if vetoed and vetoed(key, rank):
            continue
        rivals = [rival for rival, score in ranked[1:] if score > top - FONT_MARGIN]
        if all(rival in seen and max(similar(key, other) for other in seen[rival]) <= PHOTO_SCORE - PHOTO_MARGIN
               for rival in rivals):
            named[key] = (rank, top, top - ranked[1][1], "same_image_exclusion")
    return named


def agree(found, reads, counters=None):
    """Consensus for complete glyphs of one image, then the photo tier against the newly named copies."""
    counters = counters or {}
    keys = [k for k, g in found.items() if g is not None and g.sum()]
    if not keys or not font_tier_enabled():
        return reads
    def similar(a, b):
        return min(_ranked(found[a], [("-", found[b])])[0][1], _ranked(found[b], [("-", found[a])])[0][1])
    ranking = lambda k: _ranked(found[k], font_bank(), counters.get(k))
    for _ in range(4):
        named = consensus(keys, reads, similar, ranking) or exclusion(keys, reads, similar, ranking)
        if not named:
            break
        reads.update(named)
        local = [(reads[k][0], found[k]) for k in keys if reads[k][0]]
        for k in keys:
            if reads[k][0] is None:
                r = match(found[k], local, counters=counters.get(k))
                if r[3] == "photo": reads[k] = (r[0], r[1], r[2], "same_image")
    return reads


def build_bank(fonts, out=BANK):
    """Render the bundled font bank. Used by tools/build_font_bank.py; fonts must be openly licensed."""
    bank = _render_bank(fonts)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ranks=np.array([r for r, _ in bank]), glyphs=np.round(np.stack([g for _, g in bank]) * 255).astype(np.uint8))
    return len(bank)

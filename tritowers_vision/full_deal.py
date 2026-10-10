"""Automatic reader for the machine's 14 / 14 / 25 all-cards display.

Only public font templates and evidence from this photograph are used. A grid
is an editable draft; unidentified ranks stay unknown. A deck rank cut by the
next card is compared only with the same visible part of each candidate.
"""
from dataclasses import dataclass
from itertools import combinations
from collections import Counter
import cv2
import numpy as np
from PIL import Image, ImageDraw
from . import glyphs
from .image import normalize_image


@dataclass
class FullDealRead:
    draft: dict
    overlay: Image.Image
    rectified: Image.Image


def _ink(rgb):
    a = rgb.astype(np.float32)
    paper = cv2.dilate(a, np.ones((45, 45), np.uint8))
    return (((a / (paper + 1)).mean(axis=2) < .68) & (paper.max(axis=2) > 110)).astype(np.uint8)


def _objects(rgb):
    mask = _ink(rgb)
    joined = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    joined = cv2.dilate(joined, np.ones((1, 3), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(joined)
    scale = max(rgb.shape[:2]) / 1600
    candidates = [tuple(v) for v in stats[1:] if 20*scale < v[3] < 105*scale and 4*scale < v[2] < 105*scale
                  and .13 < v[2]/v[3] < 2.1 and v[4] > max(v[3]*v[2]*.17,80*scale*scale)]
    if len(candidates) < 35 or len(candidates) > 240:
        return np.empty((0,4)), np.empty(0), mask
    boxes = [(x, y, w, h) for x, y, w, h, area in candidates]
    ranked = glyphs.ranked_many([glyphs.normalize(mask[y:y+h, x:x+w]) for x, y, w, h in boxes])
    scores = [r[0][1] if r else 0 for r in ranked]
    return np.asarray(boxes, float).reshape(-1, 4), np.asarray(scores), mask


def _lines(boxes, scores, width):
    centres = boxes[:, :2] + boxes[:, 2:] / 2
    ids = np.where((boxes[:, 3] > width * .015) & (boxes[:, 3] < width * .045) & (scores > .57))[0]
    c, b = centres[ids], boxes[ids]
    lines = []
    for i, p in enumerate(c):
        for q in c[i+1:]:
            if abs(q[0] - p[0]) < width * .36:
                continue
            slope = (q[1] - p[1]) / (q[0] - p[0])
            if abs(slope) > .45:
                continue
            fit = np.array([slope, p[1] - slope * p[0]])
            good = np.abs(c[:, 1] - np.polyval(fit, c[:, 0])) < b[:, 3] * .28
            if good.sum() < 10:
                continue
            fit = np.polyfit(c[good, 0], c[good, 1], 1)
            good = np.abs(c[:, 1] - np.polyval(fit, c[:, 0])) < b[:, 3] * .28
            if good.sum() < 10 or np.ptp(c[good, 0]) < width * .50:
                continue
            mid = float(np.polyval(fit, width / 2))
            line = {"count": int(good.sum()), "mid": mid, "fit": fit, "ids": ids[good]}
            near = next((j for j, other in enumerate(lines) if abs(other["mid"] - mid) < width * .012), None)
            if near is None:
                lines.append(line)
            elif line["count"] > lines[near]["count"]:
                lines[near] = line
    return sorted(sorted(lines, key=lambda line: -line["count"])[:12], key=lambda line: line["mid"])


def _row_boxes(line, boxes, width, count=None):
    seeds = boxes[line["ids"]]
    height_fit = np.polyfit(seeds[:, 0], seeds[:, 3], 1)
    heights = np.polyval(height_fit, boxes[:, 0])
    centres = boxes[:, :2] + boxes[:, 2:] / 2
    good = ((abs(centres[:, 1] - np.polyval(line["fit"], centres[:, 0])) < heights * .38)
            & (boxes[:, 3] > heights * .68) & (boxes[:, 3] < heights * 1.45))
    selected = boxes[good]
    selected = selected[np.argsort(selected[:, 0])]
    # A small decoration or the lower fragment of a rank is not a new card.
    heights = np.polyfit(selected[:, 0], selected[:, 3], 1)
    selected = selected[selected[:, 3] > .72 * np.polyval(heights, selected[:, 0])]
    merged = []
    # A mark beyond the row's end would stretch a span-based pitch and merge
    # neighbouring ranks; the median spacing is unaffected by it.
    centres = selected[:,0]+selected[:,2]/2
    pitch = np.ptp(centres) / ((count or len(selected))-1)
    if len(centres) >= 5:
        pitch = min(pitch, 1.15*float(np.median(np.diff(centres))))
    typical_height = np.median(selected[:,3])
    for b in selected:
        x, y, w, h = b
        if merged:
            q = merged[-1]
            close = (x+w/2-q[0]-q[2]/2) < .58*pitch*(h+q[3])/(2*typical_height)
            if (x < q[0] + q[2] - 1 or close) and abs(y + h/2 - q[1] - q[3]/2) < min(h, q[3]) * .3:
                left, top = min(x, q[0]), min(y, q[1])
                right, bottom = max(x + w, q[0] + q[2]), max(y + h, q[1] + q[3])
                if right - left < 1.9 * max(h, q[3]):
                    merged[-1] = np.array([left, top, right-left, bottom-top])
                    continue
        merged.append(b.copy())
    # Compression ringing can bridge two neighbouring ranks into one blob
    # about twice the usual width; it stands for two slots, not one.
    centres = np.array([b[0]+b[2]/2 for b in merged])
    split = []
    for k, (x, y, w, h) in enumerate(merged):
        near = np.diff(centres[max(0, k-3):k+4])
        if len(near) >= 3 and w > 1.45*np.median(near):
            split += [np.array([x, y, w/2, h]), np.array([x+w/2, y, w/2, h])]
        else:
            split.append(np.array([x, y, w, h]))
    return np.asarray(split)


def _projective(indices, x, count):
    """Fit a perspective-spaced row. Normalising avoids ill-conditioned pixels."""
    centre, span = float(np.mean(x)), float(np.ptp(x))
    v = (x - centre) / span
    A = np.column_stack((np.ones(len(x)), indices, -indices * v))
    a, b, c = np.linalg.lstsq(A, v, rcond=None)[0]
    k = np.arange(count)
    if np.any(1 + c * k < .4):
        return None
    predicted = centre + span * (a + b * k) / (1 + c * k)
    gaps = np.diff(predicted)
    if min(gaps) <= 0 or max(gaps) / min(gaps) > 3.3:
        return None
    residual = np.abs(predicted[indices] - x) / np.interp(indices, k[:-1] + .5, gaps)
    if max(residual) > .30:
        return None
    return predicted, float(np.mean(residual ** 2))


def _assign(row, count, expected_left, expected_right):
    # At most three missed rank objects; never fill a layout made mostly of guesses.
    pitch = (expected_right-expected_left)/(count-1)
    centres = row[:,0]+row[:,2]/2
    row = row[(centres >= expected_left-.65*pitch) & (centres <= expected_right+.65*pitch)]
    n = len(row)
    if n < count - 3 or n > count + 2:
        return None
    proposals = []
    subsets = combinations(range(n), count) if n > count else [tuple(range(n))]
    for subset in subsets:
        observed = row[list(subset)]
        x = observed[:, 0] + observed[:, 2] / 2
        for positions in combinations(range(count), len(observed)):
            fitted = _projective(np.asarray(positions), x, count)
            if fitted is None:
                continue
            predicted, score = fitted
            pitch = np.median(np.diff(predicted))
            endpoint = ((predicted[0] - expected_left) / pitch) ** 2 + ((predicted[-1] - expected_right) / pitch) ** 2
            if endpoint > 2.5:
                continue
            proposals.append((score + .08 * endpoint, predicted, observed, positions))
    if not proposals:
        return None
    proposals.sort(key=lambda value: value[0])
    pitch = float(np.median(np.diff(proposals[0][1])))
    if len(proposals)>1 and proposals[1][0]-proposals[0][0] < .015:
        if np.max(abs(proposals[1][1]-proposals[0][1])) > pitch*.3:
            return None
    _, x, observed, positions = proposals[0]
    centres = observed[:, :2] + observed[:, 2:] / 2
    yfit = np.polyfit(centres[:, 0], centres[:, 1], 1)
    hfit = np.polyfit(centres[:, 0], observed[:, 3], 1)
    return {"x": x, "y": np.polyval(yfit, x), "h": np.polyval(hfit, x), "fit": yfit,
            "objects": {k: b for k, b in zip(positions, observed)}, "score": proposals[0][0]}


def _geometry(rgb):
    boxes, scores, mask = _objects(rgb)
    if len(boxes) < 35:
        return None
    lines = _lines(boxes, scores, rgb.shape[1])
    choices = []
    for triple in combinations(lines, 3):
        a, b, c = triple
        gaps = np.diff([line["mid"] for line in triple])
        if min(gaps) < rgb.shape[1] * .065 or max(gaps) / min(gaps) > 1.40 or c["count"] < 19:
            continue
        if max(abs(b["fit"][0] - a["fit"][0]), abs(c["fit"][0] - b["fit"][0])) > .20:
            continue
        rows = [_row_boxes(line, boxes, rgb.shape[1], count) for line,count in zip(triple,(14,14,24))]
        if not all(count-3 <= len(row) <= count+2 for row, count in zip(rows, (14,14,24))):
            continue
        # The rank columns share the screen's left and right edges. A missed
        # endpoint is recovered from the other two rows, never from deck counts.
        starts = [r[0,0] + r[0,2]/2 for r in rows]
        ends = [r[-1,0] + r[-1,2]/2 for r in rows]
        # A bezel mark beyond the screen can extend one or two rows; each
        # observed end is tried, and both card rows must fit the same edges.
        options = []
        for left in {float(np.median(starts)), *map(float, starts[:2])}:
            for right in {float(np.median(ends)), *map(float, ends[:2])}:
                pair = [_assign(row, 14, left, right) for row in rows[:2]]
                if all(row is not None for row in pair):
                    options.append((sum(row['score'] for row in pair), pair))
        if not options:
            continue
        assigned = min(options, key=lambda option: option[0])[1]
        middle = assigned[1]
        deck_left = middle["x"][0] + .4 * middle["h"][0]
        deck_right = middle["x"][-1] + middle["h"][-1]
        assigned.append(_assign(rows[2], 24, deck_left, deck_right))
        if any(row is None for row in assigned):
            continue
        choices.append((sum(row["score"] for row in assigned), assigned))
    if not choices:
        return None
    choices.sort(key=lambda value: value[0])
    return choices[0][1], mask



def _strip(rgb, row, i, shear, lo, hi, top, bottom, scale=2):
    """Sample a row-aligned patch; columns follow the cards' vertical edges."""
    x, y = row['x'][i], row['y'][i]
    W, H = max(8, round((hi-lo)*scale)), max(8, round((bottom-top)*scale))
    M = np.float32([[1/scale, shear/scale, x+lo+shear*top],
                    [row['fit'][0]/scale, 1/scale, y+top+row['fit'][0]*lo]])
    return cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def _card_frames(rgb, row, shear):
    """Left border and top border of every card in a display row.

    Each card covers the right part of its left neighbour. A border runs down
    the whole card, so the median horizontal gradient over a tall strip peaks
    there; rank strokes and pips cover too little of the strip to compete.
    The card tops form one straight edge above the ranks. Borders that are
    inconsistent with the rest of the row are replaced by the row's fit, so
    a glare spot cannot move one card's ownership. Returns (left x at the
    rank centre line, top offset in rank heights), or None.
    """
    n, offsets, tops = len(row['x']), [], []
    for i in range(n):
        h = row['h'][i]
        pitch = row['x'][i]-row['x'][i-1] if i else row['x'][1]-row['x'][0]
        lo, hi = -min(.95*pitch, 1.7*h), -.40*h
        grey = cv2.cvtColor(_strip(rgb, row, i, shear, lo, hi, -.7*h, 2.2*h), cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(cv2.GaussianBlur(grey.astype(np.float32), (0, 0), 1.2), cv2.CV_32F, 1, 0)
        profile = np.abs(np.median(gx, axis=0))
        j = int(np.argmax(profile))
        strong = profile[j] > max(6., 4*float(np.median(profile)))
        offsets.append((row['x'][i]-(row['x'][i]+lo+j/2))/h if strong else np.nan)
        # The visible face of this card, above its rank: dark surround over a bright card.
        face = _strip(rgb, row, i, shear, lo+.15*h if strong else -.3*h, .25*h, -1.3*h, -.2*h, 4)
        gy = cv2.Sobel(cv2.GaussianBlur(cv2.cvtColor(face, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 2), cv2.CV_32F, 0, 1)
        column = np.median(gy, axis=1)
        k = int(np.argmax(column))
        tops.append(-1.3+k/4/h if column[k] > 4 else np.nan)
    offsets, tops = np.asarray(offsets), np.asarray(tops)
    if np.isfinite(offsets).sum() < n/2:
        return None
    typical = float(np.nanmedian(offsets))
    good = np.isfinite(offsets) & (abs(offsets-typical) < .25)
    edges = np.where(good, row['x']-offsets*row['h'], row['x']-typical*row['h'])
    k = np.arange(n)
    good = np.isfinite(tops) & (abs(tops-np.nanmedian(tops)) < .15) if np.isfinite(tops).sum() >= n/2 else np.zeros(n, bool)
    if good.sum() >= max(3, n/2):
        fit = np.polyfit(k[good], tops[good], 1)
        good &= abs(tops-np.polyval(fit, k)) < .06
        tops = np.polyval(np.polyfit(k[good], tops[good], 1), k)
    else:
        tops = np.full(n, -1.0)
    return edges, tops


def _cluster(mask, unit, W, H):
    """The rank's own components inside a card strip, grown from a central seed.

    Parts that touch the strip's top or bottom (pips, court art, borders,
    the next row) never belong to the index. A thin flag or base that split
    from the body is kept when it lies over the body; a separate tall stroke
    (the one of a ten) must be close beside it. Specks further away, and
    hairline border fragments, are not.
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    parts = [j for j, (bx, by, bw, bh, area) in enumerate(stats[1:], 1)
             if area >= .004*unit*unit and by > 0 and by+bh < H]
    seeds = [j for j in parts if stats[j, 3] >= .30*unit and stats[j, 4] >= .015*unit*unit
             and abs(stats[j, 1]+stats[j, 3]/2-H/2) < .25*unit]
    if not seeds:
        return None, np.empty((0, 5), int)
    chosen = [max(seeds, key=lambda j: stats[j, 4])]
    grown = True
    while grown:
        grown = False
        x0 = min(stats[j, 0] for j in chosen); x1 = max(stats[j, 0]+stats[j, 2] for j in chosen)
        for j in parts:
            if j in chosen:
                continue
            bx, by, bw, bh, area = stats[j]
            if by < H/2-.62*unit or by+bh > H/2+.62*unit:
                continue
            gap = max(x0-(bx+bw), bx-x1, 0)
            over = bx+bw > x0+.1*unit and bx < x1-.1*unit
            stroke = bh >= .45*unit and bw >= .09*unit and area >= .015*unit*unit
            # Only the one of a ten stands further to the left of a rank.
            one = bh >= .75*unit and bw <= .35*unit and bx+bw <= x0 and gap < .25*unit
            if (over and area >= .006*unit*unit) or (gap < .16*unit and stroke) or (one and stroke):
                chosen.append(j); grown = True
    return np.isin(labels, chosen), stats[chosen]


def _stray_stroke(strict, glyph, unit):
    """A tall stroke right beside the rank, in its rows, that it did not take.

    The thin, cut arc of a ten's zero can fall outside the rank, or join a
    covering card's drawn border; the lone one left behind would otherwise
    look like a complete J. Only the part within the rank's rows counts, so
    pips below and plain hairline borders do not.
    """
    ys, xs = np.where(glyph)
    rows = slice(ys.min(), ys.max()+1)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(strict[rows].astype(np.uint8))
    for j, (bx, by, bw, bh, area) in enumerate(stats[1:], 1):
        if bh < .6*unit or bw < .09*unit or (glyph[rows] & (labels == j)).any():
            continue
        if max(xs.min()-(bx+bw), bx-xs.max()-1, 0) < .25*unit:
            return True
    return False


def _with_faint_strokes(distance, threshold, glyph, borders):
    """Hysteresis: fainter ink joined to the rank's strokes belongs to it.

    The skin draws hairlines (an A's left leg, a four's diagonal) that blur
    below the ink threshold. Only faint pixels connected to the strict glyph
    within its own rows are added, so separate paper texture stays out, and
    never through the card borders at the strip's edges.
    """
    ys = np.where(glyph.any(axis=1))[0]
    band = slice(ys.min(), ys.max()+1)
    _, labels = cv2.connectedComponents((distance[band] > .65*threshold).astype(np.uint8) & borders[band])
    joined = np.unique(labels[glyph[band]])
    full = glyph.copy()
    full[band] |= np.isin(labels, joined[joined > 0])
    return full


def _aspect(row):
    """Width correction for each slot's glyph: the row's typical pitch-to-height ratio over its own.

    Card pitch and rank height share the layout, so their ratio changes only
    with perspective. Restoring it lets copies of one rank across the row,
    and the font bank, be compared at a common aspect.
    """
    ratio = np.gradient(row['x'])/row['h']
    return np.clip(np.median(ratio)/ratio, .7, 1.4)


def _stretched(mask, factor):
    if mask is None or abs(factor-1) < .02:
        return mask
    h, w = mask.shape
    return cv2.resize(mask.astype(np.float32), (max(1, round(w*factor)), h), interpolation=cv2.INTER_LINEAR) > .5


def _extract(rgb, row, i, shear, frame=None, stretch=1.0):
    """(glyph, box, valid, detail) for one index; the detail records ownership.

    Ownership runs from this card's left border to the next card's border,
    below the cards' top border. Without detected borders it falls back to
    halfway between rank centres. A deck rank cut by the next card's border
    is marked occluded for the clipped reader: a hidden zero is not a K.
    """
    x, y, h = row['x'][i], row['y'][i], row['h'][i]
    last = i+1 >= len(row['x'])
    top = -.75*h
    # The size cap follows the observed index; borders bound it absolutely.
    observed = row['objects'].get(i)
    shift = 0.0 if observed is None else float(np.clip(observed[0]+observed[2]/2-x, -.4*h, .4*h))
    if frame is not None:
        edges, tops = frame
        left = edges[i] - x
        right = (edges[i+1] - x) if not last else .90*h
        cover = not last and edges[i+1]-x < .90*h
        top = max(top, (tops[i]+.05)*h)
    else:
        left = ((row['x'][i-1]-x)/2) if i else -h
        right = ((row['x'][i+1]-x)/2) if not last else h
        cover = False
    lo, hi = max(left+.04*h, shift-.90*h), min(right-.02*h, shift+.90*h)
    # Keep the band symmetric about the rank so the central-seed test holds.
    bottom = max(.72*h, -top)
    scale = 2
    box = (x+lo, y+top, x+hi, y+bottom)
    patch = _strip(rgb, row, i, shear, lo, hi, top, bottom, scale)
    H, W = patch.shape[:2]
    unit = h*scale
    distance, threshold = glyphs.ink_distance(patch)
    ink = distance > threshold
    # A border drawn as a line fills whole columns at the strip's sides.
    # Removing them keeps a cut stroke that touches a border from being
    # discarded with it; the strip then ends where the border begins.
    side = max(2, round(.2*unit))
    above = ink[:max(1, round(H/2-.55*unit))]
    lines = (ink.mean(axis=0) > .8) | ((ink.mean(axis=0) > .5) & (above.mean(axis=0) > .8))
    # A drawn border is anti-aliased; its soft edge columns go with it.
    lines = cv2.dilate(lines.astype(np.uint8)[None], np.ones((1, 5), np.uint8))[0].astype(bool)
    lines[side:W-side] = False
    ink[:, lines] = False
    columns = np.where(lines)[0]
    first = int(columns[columns < side].max())+1 if (columns < side).any() else 0
    last_column = int(columns[columns >= W-side].min()) if (columns >= W-side).any() else W
    strict, chosen = _cluster(ink, unit, W, H)
    detail = {'occluded': False}
    if strict is None:
        return None, box, False, detail
    # Faint ink is never followed through this card's border, nor through the
    # shadow of the card covering it.
    borders = np.ones((H, W), np.uint8)
    sliver = max(2, round(.06*unit))
    borders[:, :first+sliver] = 0
    borders[:, last_column:] = 0
    if cover:
        borders[:, last_column-sliver:] = 0
    glyph = _with_faint_strokes(distance, threshold, strict, borders)
    if _stray_stroke(ink, glyph, unit):
        return None, box, False, detail
    yy, xx = np.where(glyph)
    y0, y1, x0, x1 = yy.min(), yy.max()+1, xx.min(), xx.max()+1
    touches_left, touches_right = x0 <= first+1, x1 >= last_column-1
    detail['occluded'] = bool(cover and touches_right)
    valid = bool(.70*unit <= y1-y0 <= 1.25*unit and not touches_left and (not touches_right or detail['occluded']))
    if not valid:
        return None, box, False, detail
    raw = _stretched(glyph[y0:y1, x0:x1], stretch)
    inclusive, _ = _cluster((distance > .5*threshold) & borders.astype(bool), unit, W, H)
    if inclusive is not None:
        sy, sx = np.where(inclusive)
        inclusive = _stretched(inclusive[sy.min():sy.max()+1, sx.min():sx.max()+1], stretch)
    detail.update(raw=raw, inclusive=inclusive, tall=int(sum(chosen[:, 3] > .65*unit)),
                  counters=glyphs.raw_holes(raw))
    if detail['occluded']:
        return None, box, True, detail
    g = glyphs.normalize(raw)
    # Two tall strokes and no closed counter: a ten whose zero is hidden,
    # which only the clipped reader may name, never a complete K.
    if detail['tall'] > 1 and detail['counters'] == 0:
        return None, box, False, detail
    return g, box, True, detail


def _contradicted(rank, detail, clipped=False):
    """True when a lower ink threshold confidently shows a different rank.

    A stroke too faint even for the faint-stroke step (a four's diagonal under
    glare) leaves a J-like stem; the inclusive glyph then names the four.
    """
    if detail.get('inclusive') is None:
        return False
    inclusive = detail['inclusive']
    ranked = (glyphs.clipped_ranked(inclusive) if clipped
              else glyphs._ranked(glyphs.normalize(inclusive), glyphs.font_bank(), glyphs.raw_holes(inclusive)))
    (other, score), margin = ranked[0], ranked[0][1]-ranked[1][1]
    return other != rank and score >= glyphs.FONT_SCORE and margin >= glyphs.FONT_MARGIN


def _read_clipped(detail, font, photo, contradicted):
    """Name a deck rank cut by the next card only from its visible part.

    Same-image examples (complete ranks already read on this photo) are cut
    the same way (photo: best score per rank). A photo-tier read needs the
    photo gates; a font-tier read also needs the photo evidence to agree, so
    a deck J that looks like this photo's own Js is never named after a font
    3. Separate tall strokes can only be a ten here, and a ten is named only
    with both strokes visible.
    """
    combined = sorted(((rank, photo.get(rank, score)) for rank, score in font), key=lambda kv: -kv[1])
    (rank, score), margin = combined[0], combined[0][1]-combined[1][1]
    tier = None
    if rank in photo and score >= glyphs.PHOTO_SCORE and margin >= glyphs.PHOTO_MARGIN:
        tier = 'same_image_clipped'
    elif glyphs.font_tier_enabled() and font[0][0] == rank:
        (rank, score), margin = font[0], font[0][1]-font[1][1]
        if score >= glyphs.FONT_SCORE and margin >= glyphs.FONT_MARGIN:
            tier = 'font_clipped'
    if tier and (detail['tall'] > 1) == (rank == '10') and not contradicted(rank):
        return rank, score, margin, tier
    return None, score, margin, 'abstain'


def _view(slot, found, details):
    detail = details[slot]
    return ('clip', detail['raw']) if detail['occluded'] else ('full', found[slot], detail['counters'])


def _explains(a, b):
    """How well glyph b explains glyph a, comparing only what a shows."""
    (ka, pa), (kb, pb) = a[:2], b[:2]
    template = pb if kb == 'full' else glyphs.normalize(pb)
    if ka == 'clip':
        return glyphs.clipped_ranked(pa, [('-', template)])[0][1]
    if kb == 'clip':
        return glyphs.clipped_ranked(pb, [('-', pa)])[0][1]
    return glyphs._ranked(pa, [('-', template)])[0][1]


def _similar(a, b):
    return min(_explains(a, b), _explains(b, a))


def _font_ranking(view):
    if view[0] == 'clip':
        return glyphs.clipped_ranked(view[1])
    return glyphs._ranked(view[1], glyphs.font_bank(), view[2])


def _read_photo(found, details, valid, reads):
    """Clipped deck ranks, same-image matching and consensus, until stable.

    Each unread crop is compared with every crop already named on this
    photo; complete and clipped crops are compared over their common part.
    Comparisons are memoised for this read only.
    """
    views = {slot: _view(slot, found, details) for slot in details
             if valid[slot] and (details[slot]['occluded'] or found[slot] is not None)}
    memo = {}
    def remember(key, compute):
        if key not in memo:
            memo[key] = compute()
        return memo[key]
    explains = lambda a, b: remember(('explains', a, b), lambda: _explains(views[a], views[b]))
    similar = lambda a, b: min(explains(a, b), explains(b, a))
    ranking = lambda a: remember(('font', a), lambda: _font_ranking(views[a]))
    clipped = lambda a: views[a][0] == 'clip'
    contradicted = lambda a, rank: remember(('veto', a, rank), lambda: _contradicted(rank, details[a], clipped(a)))
    def score(a, b, rank):
        """Best evidence that example b (read as rank) shows what crop a shows."""
        if not clipped(a) and not clipped(b):
            return remember(('photo', a, b, rank), lambda: glyphs._ranked(views[a][1], [(rank, views[b][1])], views[a][2])[0][1])
        return explains(a, b) if clipped(a) else explains(b, a)
    for _ in range(6):
        changed = False
        examples = [(s, reads[s][0]) for s in views if reads[s][0]]
        for slot in views:
            if reads[slot][0]:
                continue
            photo = {}
            for other, rank in examples:
                photo[rank] = max(photo.get(rank, -1.0), score(slot, other, rank))
            if clipped(slot):
                new = _read_clipped(details[slot], ranking(slot), photo, lambda rank: contradicted(slot, rank))
            else:
                order = sorted(((r, photo.get(r, value)) for r, value in ranking(slot)), key=lambda kv: -kv[1])
                (rank, value), margin = order[0], order[0][1]-order[1][1]
                new = None
                if (rank in photo and value >= glyphs.PHOTO_SCORE and margin >= glyphs.PHOTO_MARGIN
                        and not contradicted(slot, rank)):
                    new = (rank, value, margin, 'same_image')
            if new and new[0]:
                reads[slot] = new
                changed = True
        if not changed:
            args = (list(views), reads, similar, ranking, contradicted)
            named = glyphs.consensus(*args) or glyphs.exclusion(*args)
            if not named:
                break
            reads.update(named)
    return reads


def _joker_face_present(rgb, row, shear):
    """Check that the exposed leftmost card exists; its identity is a rule.

    Sampling the full exposed stripe also rejects crops that removed the joker
    while leaving all 24 normal deck ranks visible.
    """
    x,y,h = row['x'][0],row['y'][0],row['h'][0]
    pitch = row['x'][1]-x
    xx,yy = np.meshgrid(np.linspace(-1.60*pitch,-.90*pitch,15),np.linspace(-.20*h,2.4*h,30))
    xs=x+xx+shear*yy; ys=y+yy+row['fit'][0]*xx
    if xs.min()<1 or ys.min()<1 or xs.max()>=rgb.shape[1]-1 or ys.max()>=rgb.shape[0]-1:
        return False
    pixels=rgb[np.rint(ys).astype(int),np.rint(xs).astype(int)].astype(float)
    v=pixels.max(axis=2); chroma=(v-pixels.min(axis=2))/(v+1)
    return bool(np.mean((v>135)&(chroma<.65))>.60)

def read_full_deal(source):
    """Return an editable 53-card draft, or None for unsupported geometry.

    Tableau slots follow display rows left-to-right. Stock slots follow draw
    order from the rightmost deck card toward the leftmost machine joker.
    """
    image = normalize_image(source)
    rgb = np.asarray(image)
    geometry = _geometry(rgb)
    if geometry is None:
        return None
    rows, _ = geometry
    # Vertical card edges share the drift between the two equally sized rows.
    shear = float(np.median((rows[1]['x']-rows[0]['x'])/(rows[1]['y']-rows[0]['y'])))
    shear = float(np.clip(shear, -.35, .35))
    if not _joker_face_present(rgb,rows[2],shear):
        return None
    found, boxes, valid, details = {}, {}, {}, {}
    for r,row in enumerate(rows):
        frame, aspect = _card_frames(rgb,row,shear), _aspect(row)
        for i in range(len(row['x'])):
            slot = f'tableau-{r*14+i+1:02d}' if r<2 else ('waste' if i==23 else f'stock-{23-i:02d}')
            found[slot], boxes[slot], valid[slot], details[slot] = _extract(rgb,row,i,shear,frame,aspect[i])
    reads, fit = glyphs.read_ranks(found, counters={slot: d['counters'] for slot, d in details.items() if 'counters' in d})
    if fit < glyphs.FONT_FIT:
        return None
    for slot, detail in details.items():
        if reads[slot][0] and _contradicted(reads[slot][0], detail):
            reads[slot] = (None, reads[slot][1], reads[slot][2], 'ink_conflict')
    reads = _read_photo(found, details, valid, reads)
    cards = {slot:{'state':'face_up','rank':rank,'score':round(score,3),'margin':round(margin,3),'tier':tier,
                   'crop':{'valid':valid[slot]}} for slot,(rank,score,margin,tier) in reads.items()}
    counts=Counter(card['rank'] for card in cards.values() if card['rank'])
    for rank,count in counts.items():
        if count>4:
            for slot in sorted((s for s in cards if cards[s]['rank']==rank),key=lambda s:cards[s]['score'])[:count-4]:
                cards[slot].update(rank=None,tier='deck_conflict')
    cards['stock-24']={'state':'face_up','rank':'*','score':0.0,'margin':0.0,'tier':'machine_rule','inferred':True}
    review=[s for s,c in cards.items() if c['rank'] is None]
    stock=[cards[f'stock-{i:02d}']['rank'] or '?' for i in range(1,25)]
    draft={'photo_kind':'full_deal','cards':cards,'stock':stock,'needs_human_review':review,'complete':not review,
           'stock_counter':24,'stock_counter_source':'full_deal_layout','font_fit':round(fit,3),'registration':{'method':'automatic_full_deal','trusted':True,
           'quality':round(max(0.,1.-sum(row['score'] for row in rows)),3),'checks':{'rows':[14,14,25],'joker_card_present':True}},
           'note':'Full-deal draft. Check every card and confirm the stock order before solving. The machine joker is last.'}
    overlay=image.copy(); draw=ImageDraw.Draw(overlay)
    for slot,box in boxes.items():
        color=(255,190,0) if slot in review else (20,220,80)
        draw.rectangle(box,outline=color,width=2)
        label = 'W' if slot == 'waste' else ('S'+str(int(slot[6:])) if slot.startswith('stock-') else str(int(slot[8:])))
        draw.text((box[0],box[1]-12),label+':'+(cards[slot]['rank'] or '?'),fill=color,stroke_width=1,stroke_fill=(0,0,0))
    return FullDealRead(draft,overlay,image.copy())

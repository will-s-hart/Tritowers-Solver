"""Find the card layout in a photo or screenshot: a homography from the 1024x768 layout frame to image pixels.

The screen border is not needed. Every starting guess (the whole frame, the bright screen region, the old edge-based
screen quadrilateral and a coarse multi-scale search for the tower silhouette) is refined with ECC against a rendering
of the expected scene: card silhouettes from the presence search in scene.py, plus the skin's static parchment band and
dark lower band, which anchor sparse late-game boards. The guess whose rendering best matches the photo wins. This
copes with loose crops, bezels, portrait phone photos, tilt and partial glare, which a fixed crop of the image cannot.
"""
from dataclasses import dataclass, field
import numpy as np
import cv2

from . import scene
from .layout import PARCHMENT_BAND, VARIABLE_BOXES

WORK = 640                       # long side of the working image used for alignment
MIN_QUALITY = 0.88               # below this the alignment is not trusted (good fits score 0.92-0.98 on the pilot set)
GOOD_ENOUGH = 0.95               # stop trying other starting guesses once a fit is this good


def _ramp(x, lo, hi):
    """0 at lo, 1 at hi (decreasing when lo > hi)."""
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def evidence(rgb, reference=None):
    """Per-pixel (face, back, parchment) evidence in [0, 1] from an RGB uint8 array, after white balance on the faces.

    Faces are bright and nearly neutral, backs are red-dominant, parchment is a saturated mid-dark brown; the rest (dark
    band, HUD, ink, pips) is other. The white reference is the brightest near-neutral pixels, which on a game screen
    are the card faces, so colour casts and exposure changes from photos are removed before the thresholds apply."""
    a = rgb.astype(np.float32)
    v = a.max(axis=2); mn = a.min(axis=2); chroma = (v - mn) / (v + 1.0)
    cand = (chroma < 0.32) & (v > np.percentile(v, 40))
    if cand.sum() > 200:
        thr = np.percentile(v[cand], 85); ref = np.median(a[cand & (v >= thr)], axis=0)
    else:
        ref = np.percentile(a.reshape(-1, 3), 99, axis=0)
    if reference is not None:
        ref = np.asarray(reference, np.float32)
    a = np.clip(a * (240.0 / np.maximum(ref, 25.0)), 0, 255)
    v = a.max(axis=2); mn = a.min(axis=2); chroma = (v - mn) / (v + 1.0)
    red = (a[..., 0] - np.maximum(a[..., 1], a[..., 2])) / (a[..., 0] + 1.0)
    face = _ramp(v, 150, 185) * _ramp(chroma, 0.30, 0.17)
    back = np.minimum(_ramp(red, 0.17, 0.32) * _ramp(v, 110, 140), 1.0 - face)      # parchment is darker (V ~100)
    hue = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2HSV)[..., 0].astype(np.float32)
    parch = _ramp(chroma, 0.28, 0.40) * _ramp(v, 35, 55) * _ramp(v, 215, 190) * ((hue >= 4) & (hue <= 32))
    return face.astype(np.float32), back.astype(np.float32), np.minimum(parch, 1.0 - face - back).astype(np.float32)


def cardness(face, back):
    """1 where a card is: face or back evidence, with enclosed holes (pips, ink, court art, counters) filled."""
    m = ((face + back) > 0.5).astype(np.uint8)
    inv = (1 - m).astype(np.uint8)
    n, lab = cv2.connectedComponents(inv, connectivity=4)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    holes = (inv > 0) & ~np.isin(lab, border)
    return (m.astype(bool) | holes).astype(np.float32)


FULL = (0, 0, 1024, 768)
PARCH_W = 0.4                    # template level for parchment between cards (cards are 1, everything else 0)


def silhouette(present, res, waste_dy=0, roi=FULL):
    """Alignment template: cards 1, visible parchment PARCH_W, other 0."""
    out = np.zeros((int(round((roi[3] - roi[1]) * res)), int(round((roi[2] - roi[0]) * res))), np.float32)
    x0, y0, x1, y1 = PARCHMENT_BAND
    out[int(round((y0 - roi[1]) * res)):int(round((y1 - roi[1]) * res)), int(round((x0 - roi[0]) * res)):int(round((x1 - roi[0]) * res))] = PARCH_W
    out[scene.render(present, res, waste_dy, roi) > 0] = 1.0
    return out


def _observed(card, parch):
    return np.maximum(card, PARCH_W * np.clip(parch * 1.6, 0, 1) * (1 - card)).astype(np.float32)


def _tpl_matrix(res, roi=(0, 0, 1024, 768)):
    """Layout frame -> template pixels."""
    return np.array([[res, 0, -res * roi[0]], [0, res, -res * roi[1]], [0, 0, 1]], np.float64)


def warp_to_layout(img, H, res, roi=(0, 0, 1024, 768), flags=cv2.INTER_LINEAR):
    """Sample an image (array) at layout-frame positions: H maps layout -> image pixels."""
    W = H @ np.linalg.inv(_tpl_matrix(res, roi))
    size = (int(round((roi[2] - roi[0]) * res)), int(round((roi[3] - roi[1]) * res)))
    return cv2.warpPerspective(img, W, size, flags=flags | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _corr(t, o, mask=None):
    t = t.astype(np.float64); o = o.astype(np.float64)
    if mask is not None: t, o = t[mask], o[mask]
    t = t - t.mean(); o = o - o.mean()
    return float((t * o).sum() / (np.sqrt((t * t).sum() * (o * o).sum()) + 1e-9))


@dataclass
class Registration:
    H: np.ndarray                     # layout frame -> original image pixels
    quality: float                    # correlation between rendered scene and photo evidence (0.92-0.98 when right)
    method: str                       # which initial guess won
    present: set = field(default_factory=set)
    waste_dy: int = 0
    margins: dict = field(default_factory=dict)
    trusted: bool = True
    checks: dict = field(default_factory=dict)


class _Work:
    def __init__(self, image, reference=None):
        w, h = image.size; self.f = min(1.0, WORK / max(w, h))
        small = image.resize((max(1, round(w * self.f)), max(1, round(h * self.f))), resample=3) if self.f < 1 else image
        self.size = small.size
        self.face, self.back, self.parch = evidence(np.asarray(small.convert("RGB")), reference)
        self.obs = _observed(cardness(self.face, self.back), self.parch)

    def to_orig(self, H): return np.diag([1 / self.f, 1 / self.f, 1.0]) @ H


def _scorer(work, H, res=0.25):
    sigma = max(0.0, 0.5 / res * _scale(H) - 0.5)        # pre-blur to avoid aliasing when the screen is large
    fb = [cv2.GaussianBlur(m, (0, 0), sigma) if sigma > 0.6 else m for m in (work.face, work.back)]
    return scene.Scorer(*(warp_to_layout(m, H, res, scene.ROI) for m in fb), res=res)


def _scale(H):
    """Approximate image pixels per layout unit around the frame centre."""
    c = scene.project(H, [(512, 400), (612, 400), (512, 500)])
    return float((np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[0])) / 200.0)


def _roi_mask(res):
    """Template pixels that take part in alignment: the whole frame except regions that change between frames."""
    m = np.full((int(round(768 * res)), int(round(1024 * res))), 255, np.uint8)
    for a, b, c, d in VARIABLE_BOXES: m[int(b * res):int(d * res), int(a * res):int(c * res)] = 0
    return m


_AFFINE, _HOMOGRAPHY = cv2.MOTION_AFFINE, cv2.MOTION_HOMOGRAPHY
# (template resolution, motion model, iterations, mask large mismatches, use cards). The first round aligns only the
# screen-level structure (parchment band, dark band, screen edges) with every card area masked, so a wrong guess about
# which cards are present cannot steer it; then affine and perspective rounds on the full scene.
MAX_REFINEMENTS, MIN_REFINE_QUALITY = 6, .40
SCHEDULE = ((0.125, _AFFINE, 50, False, False), (0.125, _AFFINE, 40, False, True), (0.25, _AFFINE, 40, False, True),
            (0.25, _HOMOGRAPHY, 40, True, True), (0.25, _HOMOGRAPHY, 30, True, True))


def _warped_obs(work, H, res):
    sigma = max(0.0, 0.5 / res * _scale(H) - 0.5)            # pre-blur so large screens do not alias
    obs = cv2.GaussianBlur(work.obs, (0, 0), sigma) if sigma > 0.6 else work.obs
    return warp_to_layout(obs, H, res)


def _outliers(tpl, obs, res):
    """Large regions where the photo disagrees with every card hypothesis (banners, glare): left out of ECC."""
    resid = np.abs(cv2.GaussianBlur(obs, (0, 0), 1.5) - cv2.GaussianBlur(tpl, (0, 0), 1.5)) > 0.6
    k = max(3, int(round(45 * res)))
    return cv2.morphologyEx(resid.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8)) > 0


def _card_area(res):
    """Everywhere a card can be (all 28 slots, the waste's range of positions), padded: unknown before presence."""
    m = np.zeros((int(round(768 * res)), int(round(1024 * res))), bool)
    rects = list(scene.RECTS.values()) + [scene.waste_rect(dy) for dy in (min(scene.WASTE_SEARCH), max(scene.WASTE_SEARCH))]
    for x0, y0, x1, y1 in rects: m[int((y0 - 6) * res):int((y1 + 6) * res), int((x0 - 6) * res):int((x1 + 6) * res)] = True
    return m


def _correction(tpl, obs, mask, motion, iters):
    """ECC correction in template coordinates (identity = no change), or None if ECC fails."""
    W = np.eye(3, dtype=np.float32) if motion == _HOMOGRAPHY else np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-5)
    try:
        if hasattr(cv2, "findTransformECCWithMask"):
            _, W = cv2.findTransformECCWithMask(tpl, obs, mask, np.full(obs.shape, 255, np.uint8), W, motion, crit, 3)
        else:
            _, W = cv2.findTransformECC(tpl, obs, W, motion, crit, mask, 3)
    except cv2.error:
        return None
    return W.astype(np.float64) if W.shape == (3, 3) else np.vstack([W.astype(np.float64), [0, 0, 1]])


def _evaluate(work, H, start=None):
    sc = _scorer(work, H, 0.25)
    present, dy, margins = scene.infer_presence(sc, start=start)
    quality = _corr(silhouette(present, 0.25, dy), _warped_obs(work, H, 0.25), _roi_mask(0.25) > 0)
    return quality, present, dy, margins


def _refine(work, H0):
    """Alternate presence inference and ECC refinement from H0. Returns (H, quality, present, dy, margins)."""
    H = H0.copy(); present = set(range(1, 29)); dy = 0
    for res, motion, iters, robust, cards in SCHEDULE:
        if cards: present, dy, _ = scene.infer_presence(_scorer(work, H, 0.25), start=present)
        tpl = cv2.GaussianBlur(silhouette(present if cards else set(), res, dy), (0, 0), 1.0)
        obs = _warped_obs(work, H, res)
        mask = _roi_mask(res) if cards else np.where(_card_area(res), 0, _roi_mask(res)).astype(np.uint8)
        if robust: mask = np.where(_outliers(tpl, obs, res), 0, mask).astype(np.uint8)
        A = _correction(tpl, obs, mask, motion, iters)
        if A is None: continue
        M = _tpl_matrix(res)
        Hn = H @ np.linalg.inv(M) @ A @ M                     # obs_warped(A u) = obs(W A u)
        if _sane(Hn, work.size): H = Hn / Hn[2, 2]
    refined = (H, *_evaluate(work, H, present))
    unrefined = (H0, *_evaluate(work, H0))
    return max(refined, unrefined, key=lambda t: t[1])


def _bright_quads(work, image):
    """Screen guesses from the largest bright region (the lit screen against a darker cabinet).

    The region's top, left and right edges follow the screen; its bottom may be the screen edge or, in this skin, the
    top of the dark band under the parchment, so both are offered and ECC quality decides."""
    rgb = np.asarray(image.resize(work.size).convert("RGB"))
    v = cv2.GaussianBlur(rgb.max(axis=2), (0, 0), 2)
    thr, m = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(3, int(round(0.02 * max(work.size))))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2: return []
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); W, Hh = work.size
    if not 0.08 * W * Hh <= st[i, cv2.CC_STAT_AREA] <= 0.95 * W * Hh: return []
    ys, xs = np.nonzero(lab == i)
    hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))
    quad = None
    for eps in np.linspace(0.01, 0.1, 10):
        a = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(a) == 4: quad = a.reshape(4, 2).astype(np.float32); break
    if quad is None: quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
    try:
        from .image import order_corners
        quad = order_corners(quad)
    except Exception:
        return []
    out = []
    for name, bottom in (("bright_quad", 768), ("bright_quad_band", 640)):
        src = np.float32([[0, 0], [1024, 0], [1024, bottom], [0, bottom]])
        out.append((name, cv2.getPerspectiveTransform(src, quad).astype(np.float64)))
    return out


def _sane(H, size):
    """Reject degenerate warps: the layout must stay convex, not tiny, and mostly inside the image."""
    c = scene.project(H, [(0, 120), (1024, 120), (1024, 768), (0, 768)]).astype(np.float32)
    if not np.isfinite(c).all() or not cv2.isContourConvex(c.reshape(-1, 1, 2)): return False
    area = abs(cv2.contourArea(c)); w, h = size
    if area < 0.04 * w * h: return False
    inside = ((c[:, 0] > -0.25 * w) & (c[:, 0] < 1.25 * w) & (c[:, 1] > -0.25 * h) & (c[:, 1] < 1.25 * h)).sum()
    return inside >= 3


def _rect_H(x0, y0, x1, y1):
    """Layout frame onto an axis-aligned image rectangle."""
    return cv2.getPerspectiveTransform(np.float32([[0, 0], [1024, 0], [1024, 768], [0, 768]]),
                                       np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])).astype(np.float64)


def _search(work, top=3):
    """Coarse similarity search: the start-of-game silhouette slid over the evidence at several scales and angles."""
    W, Hh = work.size; g = min(1.0, 320 / max(W, Hh))
    obs = cv2.resize(work.obs, (max(1, round(W * g)), max(1, round(Hh * g))), interpolation=cv2.INTER_AREA)
    full = silhouette(set(range(1, 29)), 1.0, 0)
    found = []
    for frac in np.geomspace(0.35, 1.05, 11):
        lw = frac * obs.shape[1]; s = lw / 1024.0
        if 768 * s > obs.shape[0] * 1.05: continue
        base = cv2.resize(full, (max(8, round(1024 * s)), max(6, round(768 * s))), interpolation=cv2.INTER_AREA)
        for ang in (-8.0, 0.0, 8.0):
            tpl = base if ang == 0 else cv2.warpAffine(base, cv2.getRotationMatrix2D((base.shape[1] / 2, base.shape[0] / 2), ang, 1.0), (base.shape[1], base.shape[0]))
            if tpl.shape[0] > obs.shape[0] or tpl.shape[1] > obs.shape[1]: continue
            r = cv2.matchTemplate(obs, tpl, cv2.TM_CCOEFF_NORMED)
            _, mv, _, (mx, my) = cv2.minMaxLoc(r)
            # template pixel (u, v) of the rotated 1024x768 frame -> obs pixel (u + mx, v + my)
            A = np.vstack([cv2.getRotationMatrix2D((base.shape[1] / 2, base.shape[0] / 2), ang, 1.0), [0, 0, 1]]) @ np.diag([s, s, 1.0])
            T = np.array([[1, 0, mx], [0, 1, my], [0, 0, 1]], np.float64) @ A
            found.append((mv, np.diag([1 / g, 1 / g, 1.0]) @ T))
    found.sort(key=lambda t: -t[0]); out = []
    for mv, H in found:
        c = scene.project(H, [(512, 384)])[0]
        if all(np.linalg.norm(c - scene.project(o, [(512, 384)])[0]) > 0.05 * max(W, Hh) for _, o in out): out.append((mv, H))
        if len(out) == top: break
    return [H for _, H in out]


def register(image, manual_corners=None):
    """Best layout alignment for a PIL image. manual_corners: four screen corners in image pixels (any order)."""
    work = _Work(image)
    W, Hh = work.size
    cands = []
    if manual_corners is None:
        frame = _rect_H(0, 0, W, Hh)
        initial = _evaluate(work, frame)
        if initial[0] >= .94:
            H, q, present, dy, margins = _refine(work, frame)
            trusted, checks = acceptance(work, H, q, present, dy)
            if trusted:
                return Registration(work.to_orig(H), q, "full_frame", present, dy, margins, trusted, checks)
    if manual_corners is not None:
        from .image import order_corners
        q = order_corners(manual_corners) * work.f
        cands.append(("manual", cv2.getPerspectiveTransform(np.float32([[0, 0], [1024, 0], [1024, 768], [0, 768]]), q).astype(np.float64)))
    else:
        cands.append(("full_frame", _rect_H(0, 0, W, Hh)))
        try:
            from .image import detect_screen
            q, _ = detect_screen(image.resize(work.size))
            if q is not None:
                cands.append(("screen_quad", cv2.getPerspectiveTransform(np.float32([[0, 0], [1024, 0], [1024, 768], [0, 768]]), q).astype(np.float64)))
        except Exception:
            pass
        from .automatic import candidates as anchor_candidates
        cands += anchor_candidates(work)
        cands += _bright_quads(work, image)
        cands += [("search", H) for H in _search(work, top=4)]
    cands = [(m, H0, work) for m, H0 in cands if _sane(H0, work.size)]
    if manual_corners is None:
        from .automatic import photographic_candidates
        cands += photographic_candidates(image, work)
    scored = [(m, H, w, _evaluate(w, H)) for m, H, w in cands]
    scored.sort(key=lambda candidate: -candidate[3][0])
    best = None
    for k, (method, H0, candidate_work, initial) in enumerate(scored):
        # Refinement is the costly step when no layout is supported. Accepted
        # layouts started at quality >= .52 in every local and generated check
        # (126 scenes), so weak starts are not refined, and at most six are.
        if k and (k >= MAX_REFINEMENTS or initial[0] < MIN_REFINE_QUALITY):
            break
        # Refining many equivalent starts cannot improve a supported alignment;
        # stop when both image evidence and scene-state checks pass strongly.
        H, q, present, dy, margins = _refine(candidate_work, H0)
        trusted, checks = acceptance(candidate_work, H, q, present, dy)
        candidate = (H, q, method, present, dy, margins, trusted, checks, candidate_work)
        if best is None or (trusted, q) > (best[6], best[1]):
            best = candidate
        if trusted:
            break
    if best is None:
        H = _rect_H(0, 0, W, Hh)
        return Registration(work.to_orig(H), 0.0, "none", set(), 0, {}, False)
    H, q, method, present, dy, margins, trusted, checks, candidate_work = best
    return Registration(candidate_work.to_orig(H), q, method, present, dy, margins, trusted, checks)


def acceptance(work, H, quality, present, waste_dy):
    """Accept only a supported scene, independently of rank matching scores.

    The full-screen correlation can be reduced by lighting outside the cards.
    An alternative path therefore requires substantially overlapping silhouettes,
    the right face/back class inside cards, and no unexplained card-sized face.
    Empty hypotheses face the same checks, preventing a missed last card from
    silently turning into an empty board. None of these checks proves a rank.
    """
    res = .5
    face, back = [warp_to_layout(m, H, res) for m in (work.face, work.back)]
    actual = cardness(face, back) > .5
    labels = scene.render(present, res, waste_dy, roi=FULL)
    keep = _roi_mask(res) > 0
    expected = (labels > 0) & keep
    actual &= keep
    intersection = np.count_nonzero(actual & expected)
    union = max(1, np.count_nonzero(actual | expected))
    iou = intersection / union
    interior = cv2.erode((labels > 0).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & keep
    recall = float(actual[interior].mean()) if interior.any() else 0.0
    class_map = np.where(labels == scene.FACE, face, np.where(labels == scene.BACK, back, 0))
    class_fit = float(class_map[interior].mean()) if interior.any() else 0.0
    # Exclude ordinary edge mismatch; only a coherent bright interior, well away
    # from predicted cards, can veto an empty slot. Parchment/red map ink cannot.
    padding = max(1, round(12 * res))
    explained = cv2.dilate(expected.astype(np.uint8), np.ones((padding * 2 + 1,) * 2, np.uint8)) > 0
    residual = ((face > .7) & ~explained & keep).astype(np.uint8)
    residual[:round(130 * res)] = 0
    residual[round(550 * res):] = 0
    residual = cv2.morphologyEx(residual, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(residual)
    unexplained = max(stats[1:, cv2.CC_STAT_AREA], default=0) / (98 * 150 * res * res)
    slot_fits, exposed_fits, covered_fits = [], [], []
    for pos in sorted(present):
        x0, y0, x1, _ = scene.RECTS[pos]
        # The top 52 interior units are visible for either card orientation and
        # exclude neighbouring cards, rounded corners, and lower court art.
        region = np.s_[round((y0 + 8) * res):round((y0 + 60) * res),
                       round((x0 + 8) * res):round((x1 - 8) * res)]
        ev = face if scene.exposed(present, pos) else back
        fit = float(ev[region].mean())
        slot_fits.append(fit)
        (exposed_fits if scene.exposed(present, pos) else covered_fits).append(fit)
    minimum_slot = min(slot_fits, default=0.0)
    supported = quality >= MIN_QUALITY and iou >= .78 and recall >= .88 and class_fit >= .46
    independently_supported = quality >= .80 and iou >= .85 and recall >= .95 and class_fit >= .52
    # Dense boards provide 19+ independent top-band observations. This path
    # tolerates glare/colour spill on the surrounding map only when *every*
    # exposed slot has face evidence and the covered pattern is independently supported.
    # Covered occupancy also follows from the observed blockers; a small glare
    # patch on one back must not erase a card that still has blockers in front.
    dense_supported = (len(present) >= 19 and quality >= .70 and iou >= .74 and recall >= .985
                       and class_fit >= .60 and min(exposed_fits, default=0.) >= .50
                       and len(covered_fits) >= 8 and np.median(covered_fits) >= .65
                       and np.mean(np.asarray(covered_fits) >= .48) >= .75 and unexplained < .08)
    # An animated banner can add red/brown ink to otherwise empty parchment.
    # Sparse boards remain supported when every actual card is locally verified
    # and no unexplained bright card remains anywhere in the tableau. A missed
    # last card fails that residual check even if screen correlation is high.
    sparse_supported = (1 <= len(present) <= 18 and quality >= .80 and iou >= .65 and recall >= .985
                        and class_fit >= .58 and minimum_slot >= .60 and unexplained < .05)
    trusted = bool((supported or independently_supported or dense_supported or sparse_supported) and unexplained < .22)
    checks = {"silhouette_iou": round(iou, 3), "card_interior": round(recall, 3),
              "face_back_fit": round(class_fit, 3), "min_slot_fit": round(minimum_slot, 3), "unexplained_face": round(float(unexplained), 3)}
    return trusted, checks

"""Automatic geometry regressions use generated screens, never private photos."""
import random

import cv2
import numpy as np
import pytest
from PIL import Image

from tools import vision_synth as synth
from tritowers_vision import glyphs, rank2, reader, scene
from tritowers_vision.register import _Work, _rect_H, acceptance


@pytest.mark.parametrize("seed,removed,mode", [(244, 27, "screenshot"), (344, 27, "crop"),
                                                (444, 27, "photo"), (245, 28, "crop"), (239, 22, "crop")])
def test_sparse_board_without_stock_keeps_every_visible_card(seed, removed, mode):
    rng = random.Random(seed)
    deal = synth.random_deal(rng, removed)
    deal.stock_count = 0
    image, _ = synth.capture(synth.render_screen(deal, synth.Skin(), rng), rng, mode)
    result = reader.read_photo(image)
    expected = {i for i, token in enumerate(deal.tableau, 1) if token != "--"}
    assert result.registration.trusted, result.draft["registration"]
    assert result.registration.present == expected
    tokens, _ = reader.board_tokens(result.draft)
    assert {i for i, token in enumerate(tokens, 1) if token != "--"} == expected


def test_missed_last_card_vetoes_high_global_score():
    rng = random.Random(244)
    deal = synth.random_deal(rng, 27)
    deal.stock_count = 0
    image = synth.render_screen(deal, synth.Skin(), rng).resize((1024, 768))
    work = _Work(image)
    H = _rect_H(0, 0, *work.size)
    trusted, checks = acceptance(work, H, .99, set(), deal.waste_dy)
    assert not trusted
    assert checks["unexplained_face"] > .22


def test_dense_acceptance_requires_observed_face_back_pattern():
    rng = random.Random(40)
    deal = synth.random_deal(rng, 0)
    image = synth.render_screen(deal, synth.Skin(), rng).resize((1024, 768))
    work = _Work(image)
    H = _rect_H(0, 0, *work.size)
    present = set(range(1, 29))
    assert acceptance(work, H, .76, present, deal.waste_dy)[0]
    # Face rectangles in the covered rows are not a Tri Towers starting board.
    deal.tableau = ["8" if token == "?" else token for token in deal.tableau]
    bad = _Work(synth.render_screen(deal, synth.Skin(), rng).resize((1024, 768)))
    assert not acceptance(bad, H, .76, present, deal.waste_dy)[0]


@pytest.mark.parametrize("color", ["black", "white", "#b87837"])
def test_blank_images_never_become_empty_games(color):
    result = reader.read_photo(Image.new("RGB", (800, 600), color))
    assert not result.registration.trusted
    assert reader.board_tokens(result.draft) == (["?"] * 28, "")
    assert len(result.draft["needs_human_review"]) == 29


def _scalar_ranked(query, templates):
    blurred = cv2.GaussianBlur(query, (0, 0), .8)
    holes = glyphs.holes(query)
    best = {}
    for rank, template in templates:
        if template.shape != (rank2.GH, rank2.GW):
            continue
        score = max(rank2._ncc(blurred, shifted) for shifted in rank2._shifts(cv2.GaussianBlur(template, (0, 0), .8)))
        score -= glyphs.HOLE_PENALTY if holes not in glyphs.HOLES[rank] else 0
        best[rank] = max(best.get(rank, -1.), score)
    return best


def test_vectorized_font_scores_match_scalar_reference():
    bank = glyphs.font_bank()
    rng = np.random.default_rng(19)
    queries = [next(g for r, g in bank if r == rank) for rank in glyphs.RANKS]
    queries += [np.clip(g + rng.normal(0, .07, g.shape), 0, 1).astype(np.float32) for g in queries]
    for query in queries:
        expected = _scalar_ranked(query, bank)
        got = dict(glyphs._ranked(query, bank))
        assert got.keys() == expected.keys()
        np.testing.assert_allclose(list(got[r] for r in glyphs.RANKS), list(expected[r] for r in glyphs.RANKS), atol=2e-6)
        assert max(got, key=got.get) == max(expected, key=expected.get)
        order = sorted(expected, key=expected.get, reverse=True)
        want = order[0] if (expected[order[0]] >= glyphs.FONT_SCORE and
                           expected[order[0]] - expected[order[1]] >= glyphs.FONT_MARGIN) else None
        assert glyphs.match(query)[0] == want


def test_private_templates_are_not_cached_between_reads():
    eight = next(g for r, g in glyphs.font_bank() if r == "8")
    assert glyphs._ranked(eight, [("8", eight)])[0][0] == "8"
    assert glyphs._ranked(eight, [("3", eight)])[0][0] == "3"


def test_rank_crop_excludes_separate_card_edge_fragment():
    # The narrow side fragment starts below a clean index. It must neither
    # become part of its normalized shape nor force a perfectly good A to abstain.
    patch = np.full((189, 147, 3), 238, np.uint8)
    cv2.line(patch, (39, 129), (76, 40), (15, 15, 15), 9)
    cv2.line(patch, (76, 40), (113, 129), (15, 15, 15), 9)
    cv2.line(patch, (54, 96), (98, 96), (15, 15, 15), 8)
    clean, details = glyphs.extract(patch)
    assert details["valid"]
    patch[75:106, 6:9] = 25
    fragmented, details = glyphs.extract(patch)
    assert details["valid"]
    np.testing.assert_array_equal(fragmented, clean)


def test_suit_below_missing_index_is_not_a_valid_rank_crop():
    patch = np.full((189, 147, 3), 238, np.uint8)
    cv2.fillConvexPoly(patch, np.array([[65, 112], [88, 146], [65, 175], [42, 146]]), (10, 10, 10))
    glyph, details = glyphs.extract(patch)
    assert glyph is None and not details["valid"]


def test_short_border_ink_cannot_displace_the_rank_band():
    patch = np.full((189, 171, 3), 238, np.uint8)
    cv2.line(patch, (45, 137), (82, 48), (15, 15, 15), 9)
    cv2.line(patch, (82, 48), (119, 137), (15, 15, 15), 9)
    cv2.line(patch, (60, 104), (104, 104), (15, 15, 15), 8)
    clean, details = glyphs.extract(patch)
    assert details["valid"]
    patch[4:27, 161:165] = 25
    found, details = glyphs.extract(patch)
    assert details["valid"]
    np.testing.assert_array_equal(found, clean)


def test_untrusted_geometry_cannot_supply_training_templates(monkeypatch):
    from tritowers_vision.register import Registration
    monkeypatch.setattr(reader, "register", lambda image: Registration(np.eye(3), .1, "none", trusted=False))
    assert reader.labelled_glyphs(Image.new("RGB", (800, 600), "white"), ["8"] * 28, "8") == []


def test_sparse_board_with_coloured_banner_does_not_invent_cards():
    from PIL import ImageDraw
    present = {1, 4, 5, 10, 11}
    tokens = [("7" if scene.exposed(present, p) else "?") if p in present else "--" for p in range(1, 29)]
    deal = synth.Deal(tokens, "2", stock_count=0)
    image = synth.render_screen(deal, synth.Skin(), random.Random(5)).resize((1024, 768))
    ImageDraw.Draw(image).rectangle((420, 320, 720, 415), fill=(210, 120, 50))
    result = reader.read_photo(image)
    assert result.registration.trusted, result.draft["registration"]
    assert result.registration.present == present
    assert sum(token == "--" for token in reader.board_tokens(result.draft)[0]) == 23


def test_two_digit_ten_keeps_tall_one_left_of_zero():
    patch = np.full((189, 171, 3), 238, np.uint8)
    cv2.rectangle(patch, (35, 43), (43, 135), (10, 10, 10), -1)
    cv2.ellipse(patch, (86, 88), (24, 48), 0, 0, 360, (10, 10, 10), 8)
    raw = glyphs.raw_glyph(patch)
    assert raw is not None and raw.shape[1] > 65


def test_index_registered_slightly_low_is_read_but_a_displaced_one_is_not():
    # A browser-resampled phone photo registered the leftmost cards ~10 layout
    # units low; the index is still whole and well above the suit pip.
    from PIL import Image, ImageDraw
    from test_full_deal import _public_font
    from tritowers_vision import glyphs
    unit = glyphs.PX
    width, height = (glyphs.TABLEAU_CORNER[2] - glyphs.TABLEAU_CORNER[0]) * unit, (glyphs.TABLEAU_CORNER[3] - glyphs.TABLEAU_CORNER[1]) * unit
    font = _public_font(33 * unit)
    def corner(top):
        image = Image.new("RGB", (width, height), (247, 241, 224))
        box = font.getbbox("5")
        ImageDraw.Draw(image).text((15 * unit - box[0], round(top * unit) - box[1]), "5", font=font, fill=(20, 20, 25))
        return np.asarray(image)
    glyph, detail = glyphs.extract(corner(24.5))
    assert detail["valid"] and glyphs.match(glyph, counters=detail["counters"])[0] == "5"
    assert not glyphs.extract(corner(28))[1]["valid"]


def test_rejected_photos_refine_only_plausible_candidates(monkeypatch):
    from PIL import Image
    from tritowers_vision import register
    refined = []
    original = register._refine
    monkeypatch.setattr(register, "_refine", lambda work, H0: refined.append(1) or original(work, H0))
    noise = Image.fromarray((np.random.default_rng(1).random((600, 800, 3)) * 255).astype(np.uint8))
    assert not register.register(noise).trusted
    assert 1 <= len(refined) <= register.MAX_REFINEMENTS

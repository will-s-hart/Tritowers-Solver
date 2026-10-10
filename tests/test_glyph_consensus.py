"""Same-image consensus and exclusion: when several identical crops, or this photo's own
copies of the rivals, can name a rank that no single font comparison proves.

Keys and similarities are synthetic; no photo data is involved.
"""
import cv2
import numpy as np

from tritowers_vision import glyphs

UNREAD = (None, 0.0, 0.0, "abstain")


def run(rule, ranking, similar, read=()):
    keys = sorted(ranking) + [k for k, _ in read]
    reads = {k: UNREAD for k in ranking}
    reads.update({k: (rank, .9, .2, "font") for k, rank in read})
    def look(a, b):
        return similar.get((a, b), similar.get((b, a), .2))
    return rule(keys, reads, look, lambda k: ranking[k])


THREE = [("3", .82), ("5", .75), ("8", .67)]


def test_identical_crops_are_named_when_the_rival_looks_different_here():
    named = run(glyphs.consensus, {"a": THREE, "b": THREE}, {("a", "b"): .98, ("a", "five"): .68, ("b", "five"): .69},
                read=[("five", "5")])
    assert {k: v[0] for k, v in named.items()} == {"a": "3", "b": "3"}


def test_no_consensus_without_seeing_the_rival_or_when_it_looks_alike():
    assert not run(glyphs.consensus, {"a": THREE, "b": THREE}, {("a", "b"): .98})
    assert not run(glyphs.consensus, {"a": THREE, "b": THREE},
                   {("a", "b"): .98, ("a", "five"): .92, ("b", "five"): .9}, read=[("five", "5")])


def test_a_single_crop_or_two_unlike_groups_are_not_a_consensus():
    assert not run(glyphs.consensus, {"a": THREE}, {}, read=[("five", "5")])
    groups = {("a", "b"): .97, ("c", "d"): .97, ("a", "five"): .6, ("b", "five"): .6, ("c", "five"): .6, ("d", "five"): .6}
    assert not run(glyphs.consensus, {k: THREE for k in "abcd"}, groups, read=[("five", "5")])


def test_a_rank_already_read_here_is_left_to_the_photo_tier():
    assert not run(glyphs.consensus, {"a": THREE, "b": THREE}, {("a", "b"): .98, ("a", "five"): .6, ("b", "five"): .6},
                   read=[("five", "5"), ("three", "3")])


def test_exclusion_names_a_lone_crop_only_when_every_close_rival_is_ruled_out():
    assert run(glyphs.exclusion, {"a": THREE}, {("a", "five"): .68}, read=[("five", "5")])["a"][0] == "3"
    assert not run(glyphs.exclusion, {"a": THREE}, {})                                   # rival never seen here
    assert not run(glyphs.exclusion, {"a": THREE}, {("a", "five"): .8}, read=[("five", "5")])  # looks like this photo's 5
    weak = [("3", .7), ("5", .6)]
    assert not run(glyphs.exclusion, {"a": weak}, {("a", "five"): .5}, read=[("five", "5")])   # below the font score


def test_holes_are_counted_before_a_narrow_opening_closes_by_scaling():
    ring = np.zeros((120, 90), np.uint8)
    cv2.ellipse(ring, (45, 60), (36, 52), 0, 0, 360, 1, 14)
    ring[0:30, 45:46] = 0                                    # a hairline opening, like the curl of this skin's 3
    assert glyphs.raw_holes(ring.astype(bool)) == 0
    assert glyphs.holes(glyphs.normalize(ring.astype(np.float32))) == 1

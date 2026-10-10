"""Browser Python transport: real shared solver, no server or private photos."""
import random
from types import SimpleNamespace

from PIL import Image
import pytest

import solver
import solver_ui as ui
import web_core as core


@pytest.fixture(autouse=True)
def fresh_worker():
    core._sessions.clear()
    core._replies.clear()
    yield
    core._sessions.clear()
    core._replies.clear()


def new(**overrides):
    return core.api_new({"board": " ".join(["6"] + ["--"] * 27),
                         "waste": "5", "stock": 3, **overrides})


def act(sid, op, **extra):
    return core.api_act({"sid": sid, "op": op, **extra})


def test_retrying_start_and_action_preserves_one_session_one_draw():
    first = new(req_id="new-once")
    assert new(req_id="new-once") == first
    assert len(core._sessions) == 1
    sid = first["sid"]
    result = act(sid, "draw", rank="9", req_id="draw-once")
    assert act(sid, "draw", rank="9", req_id="draw-once") == result
    assert result["stock"] == 2 and result["log"] == ["Drew 9."]


def test_late_mutation_retry_survives_poll_and_reply_cache_pressure():
    sid = new()["sid"]
    act(sid, "draw", rank="9", req_id="draw-one")
    act(sid, "draw", rank="7", req_id="draw-two")
    for i in range(300):
        assert act(sid, "state", req_id=f"poll-{i}")["ok"]
        core.api_solve({"board": "? " * 28, "req_id": f"solve-{i}"})
    retry = act(sid, "draw", rank="9", req_id="draw-one")
    assert retry["stock"] == 1 and retry["waste"] == "7"
    assert retry["log"] == ["Drew 9.", "Drew 7."]


def test_refresh_replays_all_actions_including_undo_and_preserves_history():
    setup = {"board": " ".join(["?", "4"] + ["--"] * 26),
             "waste": "A", "stock": 3, "sid": "saved-game", "req_id": "saved-new"}
    sid = core.api_new(setup)["sid"]
    journal = [dict(op="reveal", pos=1, rank="2"), dict(op="play", pos=1),
               dict(op="draw", rank="3"), dict(op="play", pos=2), dict(op="undo")]
    for i, action in enumerate(journal):
        before = act(sid, req_id=str(i), **action)
        assert before["ok"], before
    assert before["can_undo"] and before["remaining"] == 1
    core._sessions.clear(); core._replies.clear()  # new browser worker after refresh
    assert act(sid, "state", req_id="restore-state")["gone"]
    assert core.api_new(setup)["sid"] == sid
    for action in journal: assert act(sid, **action)["ok"]
    restored = act(sid, "state", req_id="restore-state")
    assert {k: v for k, v in restored.items() if k != "message"} == {
        k: v for k, v in before.items() if k != "message"}
    # Undo again after refresh: remove the preceding draw, not the replay itself.
    undo = act(sid, "undo", req_id="undo-after-refresh")
    assert undo["stock"] == 3 and undo["waste"] == "2" and undo["remaining"] == 1


def test_cached_new_does_not_prevent_recreation_of_an_evicted_session(monkeypatch):
    monkeypatch.setattr(core, "MAX_SESSIONS", 1)
    first = new(sid="replayed-game", req_id="restore-new")
    new(sid="another-game", req_id="other-new")
    assert act(first["sid"], "state")["gone"]
    restored = new(sid="replayed-game", req_id="restore-new")
    assert restored["ok"] and restored["sid"] == first["sid"]
    assert act(first["sid"], "state")["ok"]


def test_action_touches_session_for_local_eviction_order(monkeypatch):
    monkeypatch.setattr(core, "MAX_SESSIONS", 2)
    a, b = new()["sid"], new()["sid"]
    assert act(a, "state")["ok"]
    new()
    assert act(a, "state")["ok"] and act(b, "state")["gone"]


@pytest.mark.parametrize("body", [None, [], "x", {"sid": []}, {"req_id": []}, {"req_id": "x" * 129}])
def test_malformed_local_messages_return_errors(body):
    for fn in (core.api_new, core.api_act, core.api_solve):
        assert fn(body)["ok"] is False


@pytest.mark.parametrize("rank", [None, [], "<img/src/onerror=alert(1)>", "?", "--", "Z"])
def test_invalid_draw_rank_never_mutates_or_adds_undo(rank):
    sid = new()["sid"]
    before = act(sid, "state")
    invalid = act(sid, "draw", rank=rank)
    assert invalid["ok"] is False and invalid["message"]
    after = act(sid, "state")
    assert after == before and not after["can_undo"]


def test_joker_tail_is_automatic_and_undo_restores_its_reservation():
    sid = new(stock=2, joker=True)["sid"]
    assert not act(sid, "draw", rank="*")["ok"]
    assert act(sid, "draw", rank="A")["stock"] == 1
    final = act(sid, "draw", req_id="joker-once")
    assert final["waste"] == "*" and final["stock"] == 0 and not final["joker"]
    assert act(sid, "draw", req_id="joker-once") == final
    undo = act(sid, "undo")
    assert undo["waste"] == "A" and undo["stock"] == 1 and undo["joker"]
    assert act(sid, "draw")["waste"] == "*"


def test_recommend_uses_server_simulation_cap_and_action_clears_advice(monkeypatch):
    calls = []
    def recommend(session, sims, seed):
        calls.append(sims)
        return "Play position 01.", 1, True, 1.0, 0, {"type": "play", "pos": 1}
    monkeypatch.setattr(ui, "recommend_action", recommend)
    sid = new()["sid"]
    result = act(sid, "recommend", sims=100000, req_id="recommend")
    assert calls == [ui.MAX_SIMULATIONS]
    assert "Play the 6" in result["advice"]["text"]
    assert result["next_action"] == result["advice"]["action"] == {"type": "play", "pos": 1}
    assert act(sid, "recommend", sims=100000, req_id="recommend") == result
    assert calls == [ui.MAX_SIMULATIONS]
    assert act(sid, "draw", rank="9")["advice"] is None


def test_corrected_photo_draft_stays_unknown_and_follows_play_along(monkeypatch):
    from tritowers_vision import intake
    bottom = ["9", "3", "4", "5", "6", "7", "8", "9", "10", "J"]
    cards = {f"tableau-{p:02d}": {"state": "covered" if p <= 18 else "face_up",
             "rank": None if p <= 18 else bottom[p-19]} for p in range(1, 29)}
    cards["waste"] = {"state": "face_up", "rank": "A"}
    draft = {"cards": cards, "needs_human_review": ["tableau-19"],
             "registration": {"trusted": True}}
    def read(data, templates, corners):
        assert templates == [] and corners is None
        return SimpleNamespace(draft=draft, overlay=Image.new("RGB", (20, 20), "green"))
    monkeypatch.setattr(intake, "read_photo", read)
    photo = core.api_photo(b"synthetic adapter fixture")
    assert photo["ok"] and photo["trusted"] and photo["board"][:18] == ["?"] * 18
    assert photo["review"] == ["19"] and photo["overlay"].startswith("data:image/jpeg;base64,")
    corrected = photo["board"][:]; corrected[18] = "2"
    session = core.api_new({"board": " ".join(corrected), "waste": photo["waste"], "stock": 2, "joker": True})
    sid = session["sid"]
    assert not core._sessions[sid]["s"].game.stock_known
    assert core.api_solve({"board": " ".join(corrected), "waste": "A", "stock_count": 2})["status"] == "incomplete"
    assert act(sid, "play", pos=19)["waste"] == "2"
    exposed = act(sid, "play", pos=20)
    assert exposed["pending"] == [10] and exposed["cells"][9]["card"] == "?"
    assert not act(sid, "draw", rank="4")["ok"]
    assert act(sid, "reveal", pos=10, rank="A")["pending"] == []
    assert act(sid, "draw", rank="5")["stock"] == 1
    assert act(sid, "draw")["waste"] == "*"


def solve(**extra):
    return core.api_solve({"board": " ".join(["6"] + ["--"] * 27),
                           "waste": "5", "stock": [], "stock_count": 0, **extra})


@pytest.mark.parametrize("extra,missing", [({"board": "? " * 28}, "tableau position"),
    ({"stock_count": 2}, "stock order"), ({"waste": ""}, "waste card"),
    ({"stock": ["?", "*"], "stock_count": 2}, "stock card 1"), ({"waste": "?"}, "waste card")])
def test_incomplete_exact_input_is_never_filled_in(extra, missing):
    result = solve(**extra)
    assert result["ok"] and result["status"] == "incomplete" and missing in result["message"]
    assert "steps" not in result


def test_partial_known_position_solves_and_frames_replay():
    result = solve()
    assert result["status"] == "solved" and result["steps"] == ["Play the 6 (position 1)"]
    assert result["frames"][0]["remaining"] == 1 and result["frames"][-1]["remaining"] == 0
    assert result["policy"] == "stuck"


def test_full_known_deal_solves_and_finishes_empty():
    deck = [r for r in solver.RANKS for _ in range(4)]
    random.Random(3).shuffle(deck)
    result = solve(board=" ".join(deck[:28]), waste=deck[28], stock=deck[29:], stock_count=23)
    assert result["ok"] and result["status"] == "solved"
    assert result["frames"][0]["remaining"] == 28 and result["frames"][-1]["remaining"] == 0
    assert len(result["frames"]) == len(result["steps"]) + 1


def test_known_joker_endgame_is_verified_by_replay():
    result = solve(board=" ".join(["K"] + ["--"] * 27), stock=["*"], stock_count=1)
    assert result["status"] == "solved"
    assert result["steps"] == ["Draw from the stock: *", "Play the K (position 1)"]
    assert result["frames"][1]["waste"] == "*" and result["frames"][-1]["remaining"] == 0


def test_unsolvable_and_budget_exhaustion_remain_distinct():
    assert solve(waste="A")["status"] == "unsolvable"
    timed = solve(time_budget=0)
    assert timed["status"] == "unknown" and "NOT a proof" in timed["message"]


def test_inconsistent_cleared_geometry_and_impossible_ranks_are_rejected():
    invalid = solve(board=" ".join(["--"] * 27 + ["6"]))
    assert not invalid["ok"] and "still covered" in invalid["message"]
    invalid = solve(stock=["Q"] * 5, stock_count=5)
    assert not invalid["ok"] and "Impossible deck" in invalid["message"]


def test_solve_retries_reuse_the_verified_response(monkeypatch):
    original = core.solve_deal
    calls = []
    def wrapped(body):
        calls.append(body)
        return original(body)
    monkeypatch.setattr(core, "solve_deal", wrapped)
    first = solve(req_id="solve-once")
    assert solve(req_id="solve-once") == first
    assert len(calls) == 1

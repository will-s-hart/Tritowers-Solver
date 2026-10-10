"""The explicit next-action and known-stock contract shared by HTTP and the browser worker.

A null position never means "draw": every state says what comes next, and only
known stock ranks are exposed. Deals are seeded and public.
"""
import random

from fastapi.testclient import TestClient
import pytest

import solver
import solver_ui as ui
import web_app
import web_core as core
import web_shared

RANKS = "A 2 3 4 5 6 7 8 9 10 J Q K".split()
LONE_NINE = ["9"] + ["--"] * 27


@pytest.fixture(autouse=True)
def fresh_worker():
    core._sessions.clear(); core._replies.clear()
    yield
    core._sessions.clear(); core._replies.clear()


def worker(board, waste, stock, joker=False):
    sid = core.api_new({"board": " ".join(board), "waste": waste, "stock": stock, "joker": joker})["sid"]
    return lambda op, **extra: core.api_act({"sid": sid, "op": op, **extra})


def seeded_deal(seed=7):
    rng = random.Random(seed)
    deck = [rank for rank in RANKS for _ in range(4)]
    rng.shuffle(deck)
    return deck[:28], deck[28], deck[29:] + ["*"]


def test_forced_draw_then_play_then_completion():
    act = worker(LONE_NINE, "5", 3)
    state = act("state")
    assert state["next_action"] == {"type": "draw", "forced": True}
    assert state["can_draw"] and state["stock_next"] is None and not state["stock_known"]
    drawn = act("draw", rank="8")
    # A play is available now, but nothing has been recommended yet: no cue.
    assert drawn["next_action"] == {"type": "none", "reason": "awaiting_advice"}
    advice = act("recommend", sims=50, seed=1)
    assert advice["next_action"] == advice["advice"]["action"] == {"type": "play", "pos": 1}
    done = act("play", pos=1)
    assert done["over"] and done["next_action"] == {"type": "done", "reason": "won"} and done["advice"] is None
    undone = act("undo")
    assert undone["next_action"] == {"type": "none", "reason": "awaiting_advice"} and undone["waste"] == "8"


def test_pending_reveal_and_empty_stock_clear_the_draw():
    act = worker(["?", "?"] + ["--"] * 26, "5", 0)
    assert act("state")["next_action"] == {"type": "reveal", "positions": [1, 2]}
    act = worker(LONE_NINE, "5", 0)
    state = act("state")
    assert state["next_action"] == {"type": "none", "reason": "no_moves"} and not state["can_draw"]


def test_final_joker_is_the_known_next_card():
    act = worker(LONE_NINE, "5", 2, joker=True)
    assert act("state")["stock_next"] is None
    last = act("draw", rank="2")
    assert last["stock"] == 1 and last["stock_next"] == "*" and last["next_action"]["type"] == "draw"
    joker = act("draw")
    assert joker["waste"] == "*" and joker["stock"] == 0 and joker["stock_next"] is None
    assert joker["next_action"] == {"type": "none", "reason": "awaiting_advice"}
    assert act("undo")["stock_next"] == "*"


def test_exact_recommendation_names_the_draw_explicitly():
    session = ui.Session(solver.Game(LONE_NINE, "5", True, ["8", "*"]))
    text, pos, proven, _, _, action = ui.recommend_action(session, 10)
    assert pos is None and proven and action == {"type": "draw", "forced": True} and "Draw" in text
    assert ui.recommend_detail(session, 10)[:3] == (text, None, True)


def test_replay_frames_state_each_step_and_the_known_stock():
    board, waste, stock = seeded_deal()
    result = web_shared.solve_deal({"board": " ".join(board), "waste": waste, "stock": stock, "stock_count": 24})
    assert result["status"] == "solved"
    frames, steps = result["frames"], result["steps"]
    assert len(frames) == len(steps) + 1 and frames[-1]["next_action"] == {"type": "done", "reason": "won"}
    for frame, step in zip(frames, steps):
        left = frame["stock"]
        assert frame["stock_known"] and frame["stock_next"] == (stock[len(stock) - left] if left else None)
        action = frame["next_action"]
        if step.startswith("Draw"):
            assert action == {"type": "draw", "card": frame["stock_next"]} and frame["next"] is None
            assert step.endswith(": " + frame["stock_next"])
        else:
            assert action == {"type": "play", "pos": frame["next"]} and f"position {frame['next']}" in step
    assert any(f["next_action"]["type"] == "draw" for f in frames)


def test_http_and_worker_report_the_same_state(monkeypatch):
    # HTTP also precomputes advice in the background; the worker computes it on
    # request. With advice requested explicitly the contract must be identical.
    monkeypatch.setattr(web_app, "_precompute", lambda entry: None)
    keys = ("next_action", "stock_next", "stock_known", "stock", "waste", "can_draw", "over", "pending")
    act = worker(LONE_NINE, "5", 2, joker=True)
    with TestClient(web_app.app) as client:
        sid = client.post("/api/new", json={"board": " ".join(LONE_NINE), "waste": "5", "stock": 2, "joker": True}).json()["sid"]
        http = lambda n, op, **extra: client.post("/api/act", json={"sid": sid, "op": op, "req_id": f"step-{n}", **extra}).json()
        steps = (("state", {}), ("draw", {"rank": "8"}), ("recommend", {"sims": 50, "seed": 1}), ("undo", {}),
                 ("draw", {"rank": "2"}), ("draw", {}), ("undo", {}))
        for n, (op, extra) in enumerate(steps):
            ours, theirs = act(op, **extra), http(n, op, **extra)
            assert {k: ours[k] for k in keys} == {k: theirs[k] for k in keys}, op

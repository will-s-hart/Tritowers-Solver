"""In-browser adapter. Uses exactly the same rules and responses as the HTTP app.

Worker messages execute serially, so no threads or server are needed. Request
IDs and replayable session actions make retries and refresh safe.
"""
from collections import OrderedDict
import uuid
import solver_ui as ui
from web_shared import GEO, ASPECT, view, solve_deal, photo_response, advice_for

_sessions = OrderedDict()
_replies = OrderedDict()
MAX_SESSIONS = 50
_MUTATIONS = ("play", "reveal", "draw", "undo")


def api_geo():
    return {"geo": {str(k): v for k, v in GEO.items()}, "aspect": ASPECT}


def warm_vision():
    """Build the public font caches while idle, so the first photo read does not pay for them."""
    from tritowers_vision import glyphs
    glyphs.warm()
    return {"ok": True}


def health():
    return {"ok": True, "engine": "pyodide", "templates": 0}


def _once(route, body, fn):
    if not isinstance(body, dict): return {"ok": False, "message": "Expected a JSON object."}
    rid = body.get("req_id")
    if rid is not None and (not isinstance(rid, str) or len(rid) > 128):
        return {"ok": False, "message": "Invalid request ID."}
    key = (route, body.get("sid", ""), rid)
    if not isinstance(key[1], str): return {"ok": False, "message": "Invalid session ID."}
    entry = _sessions.get(key[1]) if route == "act" else None
    mutation = route == "act" and body.get("op") in _MUTATIONS
    if entry and rid and rid in entry["applied"]:
        # Return the current board, not a historical board that could make a
        # late retry appear to undo later actions. Never perform it twice.
        return view(entry, entry["applied"][rid])
    cacheable = route != "act" or (entry is not None and body.get("op") == "recommend")
    if route == "act" and entry is not None:
        key += (len(entry["s"].log),)
    if rid and cacheable and key in _replies:
        cached = _replies[key]
        if route != "new" or cached.get("sid") in _sessions:
            return cached
        del _replies[key]  # an evicted session must be recreated for replay
    try: result = fn()
    except (ValueError, TypeError) as e: result = {"ok": False, "message": str(e)}
    if rid and mutation and entry is not None and result.get("ok"):
        # This small request-ID ledger survives polling/cache pressure for the
        # lifetime of the session. Full response objects remain bounded below.
        entry["applied"][rid] = result.get("message", "")
    elif rid and cacheable:
        _replies[key] = result
        while len(_replies) > 256: _replies.popitem(last=False)
    return result


def api_new(body):
    def create():
        waste = body.get("waste", "")
        if not isinstance(waste, str) or not waste.strip(): raise ValueError("Pick the waste card first.")
        s = ui.new_session(body.get("board", ""), waste, body.get("stock", 23), joker=body.get("joker", False))
        # A saved ID is allowed only in this private, browser-local replay adapter.
        sid = body.get("sid") or uuid.uuid4().hex
        if not isinstance(sid, str) or len(sid) > 128: raise ValueError("Invalid session ID.")
        if sid in _sessions: return view(_sessions[sid], extra={"sid": sid})
        entry = {"s": s, "advice": None, "applied": {}}
        _sessions[sid] = entry
        while len(_sessions) > MAX_SESSIONS: _sessions.popitem(last=False)
        return view(entry, extra={"sid": sid})
    return _once("new", body, create)


def api_act(body):
    def action():
        entry = _sessions.get(body.get("sid", ""))
        if not entry: return {"ok": False, "gone": True, "message": "This game expired. Start again."}
        _sessions.move_to_end(body["sid"])
        s = entry["s"]; op = body.get("op"); message = ""
        try:
            if op == "play": ui.do_play(s, body.get("pos"))
            elif op == "reveal": ui.do_reveal(s, body.get("pos"), body.get("rank", ""))
            elif op == "draw": ui.do_draw(s, "*" if s.game.joker_in_stock and s.game.stock_remaining == 1 else body.get("rank", ""))
            elif op == "undo": s.undo()
            elif op == "recommend":
                sims = min(ui.MAX_SIMULATIONS, ui.to_int(body.get("sims", 1200), "Simulations", 1))
                entry["advice"] = advice_for(s, sims, body.get("seed"))
            elif op != "state": raise ValueError("Unknown action.")
            if op in ("play", "reveal", "draw", "undo"):
                entry["advice"] = None; message = s.log[-1]
            return view(entry, message)
        except ValueError as error: return view(entry, str(error), ok=False)
    # Missing-session replies are transient: restoring the session must allow retry.
    result = _once("act", body, action)
    if result.get("gone") and isinstance(body, dict):
        _replies.pop(("act", body.get("sid", ""), body.get("req_id")), None)
    return result


def api_solve(body):
    return _once("solve", body, lambda: solve_deal(body))


def api_photo(data, corners=""):
    return photo_response(bytes(data), corners)

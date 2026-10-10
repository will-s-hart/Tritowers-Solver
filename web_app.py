"""Plain HTTP front end (no Gradio queue, no SSE). Every action is a JSON POST that the page can safely retry:
the server caches the response per request id, so a retry after a dropped connection never applies an action twice.
Run: python web_app.py   (PORT env, default 7860)."""
import os, threading, time, uuid, logging
from collections import OrderedDict
from pathlib import Path
from fastapi import FastAPI, File, Form, UploadFile, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import solver, solver_ui as ui
from web_shared import GEO, ASPECT, view, solve_deal, photo_response, advice_for

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
MAX_SESSIONS = 300
MAX_SESSIONS_PER_CLIENT = 20
SESSION_TTL = 3600
MAX_REPLY_CACHE = 128
MAX_SESSION_ACTIONS = 512
MUTATING_ACTIONS = {"play", "reveal", "draw", "undo"}
MAX_REQUEST_CACHE = 600
MAX_SIMULATIONS = 2000
log = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
app = FastAPI(title="TriTowers")
_sessions = OrderedDict()   # sid -> {"s": Session, "lock": Lock, "replies": {req_id: reply|Event}, "advice": dict|None}
_glock = threading.Lock()
_TEMPLATES = None
_requests = OrderedDict()
_cpu_slots = threading.BoundedSemaphore(2)
_photo_slots = threading.BoundedSemaphore(1)
_pre_jobs = OrderedDict()
_pre_condition = threading.Condition()
_pre_worker = None

class RetryLater(Exception):
    pass


def _retry(message="The server is busy. Please retry shortly.", *, pending=False):
    return {"ok": False, "pending" if pending else "busy": True,
            "retry_after": 1, "message": message}


def _request_id(body):
    rid = body.get("req_id")
    if rid is None: return uuid.uuid4().hex
    if not isinstance(rid, str) or not rid or len(rid) > 128:
        raise ValueError("Request ID must be a nonempty string of at most 128 characters.")
    return rid


def _client(request):
    # Never trust a client-supplied forwarded-for header for admission limits.
    return request.client.host if request.client else "local"


def _text(value, name, maximum=1024):
    if not isinstance(value, str): raise ValueError(f"{name} must be text.")
    if len(value) > maximum: raise ValueError(f"{name} is too long.")
    return value


def _trim(cache, limit):
    # A running request must remain reserved even under cache pressure.
    for key in list(cache):
        if len(cache) <= limit: break
        if not isinstance(cache[key], threading.Event): del cache[key]


def _claim(cache, key, limit):
    with _glock:
        cached = cache.get(key)
        if isinstance(cached, threading.Event): return None, _retry("That request is still running.", pending=True)
        if cached is not None: return None, cached
        if sum(isinstance(value, threading.Event) for value in cache.values()) >= limit:
            return None, _retry()
        event = threading.Event(); cache[key] = event
        _trim(cache, limit)
        return event, None


def _finish(cache, key, event, reply, limit):
    with _glock:
        if reply.get("busy") is True or reply.get("pending") is True:
            cache.pop(key, None)
        else:
            cache[key] = reply
            _trim(cache, limit)
        event.set()
    return reply


def _expire_sessions():
    now = time.monotonic()
    for sid, entry in list(_sessions.items()):
        if now - entry["touched"] > SESSION_TTL and not entry["lock"].locked():
            entry["active"] = False
            _sessions.pop(sid, None)
            with _pre_condition: _pre_jobs.pop(sid, None)


async def _too_large(request, error):
    return JSONResponse({"ok": False, "message": "Upload or request is too large."}, status_code=413)


class RequestBodyLimit:
    """Bound streamed requests before JSON/multipart parsers buffer or spool them."""
    def __init__(self, app): self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/api/"):
            return await self.app(scope, receive, send)
        from tritowers_vision.image import MAX_BYTES
        limit = MAX_BYTES + 1024 * 1024 if scope["path"] == "/api/photo" else 16384
        length = dict(scope["headers"]).get(b"content-length")
        if length is not None:
            try:
                size = int(length)
                if size < 0: raise ValueError()
            except ValueError:
                response = JSONResponse({"ok": False, "message": "Invalid request length."}, status_code=400)
                return await response(scope, receive, send)
            if size > limit:
                response = await _too_large(None, None)
                return await response(scope, receive, send)
        received = 0
        async def bounded_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit: raise HTTPException(status_code=413)
            return message
        await self.app(scope, bounded_receive, send)


app.add_exception_handler(413, _too_large)
app.add_middleware(RequestBodyLimit)


def _templates():
    global _TEMPLATES
    if _TEMPLATES is None:
        path = os.environ.get("TT_TEMPLATES")
        if path and os.path.exists(path):
            from tritowers_vision.rank import load_templates
            _TEMPLATES = load_templates(path)
        else: _TEMPLATES = []
    return _TEMPLATES

def _get(sid):
    if not isinstance(sid, str) or len(sid) > 128: return None
    with _glock:
        _expire_sessions()
        entry = _sessions.get(sid)
        if entry: entry["touched"] = time.monotonic()
        return entry

@app.get("/", response_class=HTMLResponse)
def index(): return (HERE / "web" / "index.html").read_text(encoding="utf-8")

app.mount("/web", StaticFiles(directory=HERE / "web"), name="web")

@app.get("/api/geo")
def api_geo(): return {"geo": {str(k): v for k, v in GEO.items()}, "aspect": ASPECT}

@app.get("/health")
async def health(): return {"ok": True}

@app.post("/api/new")
def api_new(body: dict, request: Request):
    try:
        rid = _request_id(body)
        board = _text(body.get("board", ""), "Board")
        waste = _text(body.get("waste", ""), "Waste", 16).strip()
        if not waste: raise ValueError("Pick the waste card first.")
        joker = body.get("joker", False)
        if not isinstance(joker, bool): raise ValueError("Joker mode must be true or false.")
        session = ui.new_session(board, waste, body.get("stock", 23), joker=joker)
    except ValueError as error: return {"ok": False, "message": str(error)}
    owner = _client(request); key = (owner, "new", rid)
    # Creation IDs live with the session, so polling/solve cache churn cannot
    # create a second game. A retry resumes its current state, not its first view.
    with _glock:
        _expire_sessions()
        existing = next((entry for entry in _sessions.values() if entry.get("new_key") == key), None)
        if existing:
            existing["touched"] = time.monotonic()
        else:
            cached = _requests.get(key)
            if isinstance(cached, dict) and cached.get("sid"):
                _requests.pop(key, None)  # stale successful reply for an expired game
    if existing:
        if not existing["lock"].acquire(blocking=False): return _retry()
        try:
            reply = view(existing); reply["sid"] = existing["sid"]
            return reply
        finally: existing["lock"].release()
    event, cached = _claim(_requests, key, MAX_REQUEST_CACHE)
    if cached is not None: return cached
    try:
        with _glock:
            _expire_sessions()
            if len(_sessions) >= MAX_SESSIONS or sum(e["owner"] == owner for e in _sessions.values()) >= MAX_SESSIONS_PER_CLIENT:
                return _finish_unlocked_new_limit(key, event)
            sid = uuid.uuid4().hex
            entry = {"s": session, "sid": sid, "owner": owner, "new_key": key, "active": True,
                     "touched": time.monotonic(), "lock": threading.Lock(),
                     "replies": OrderedDict(), "applied": {}, "advice": None}
            _sessions[sid] = entry
        _precompute(entry)
        reply = view(entry); reply["sid"] = sid
    except Exception:
        log.exception("Could not start game")
        reply = {"ok": False, "message": "Could not start this game. Please try again."}
    return _finish(_requests, key, event, reply, MAX_REQUEST_CACHE)


def _finish_unlocked_new_limit(key, event):
    # Caller already holds _glock; do not recursively acquire it.
    reply = {"ok": False, "message": "Too many active games. Resume an existing game or wait for an old game to expire."}
    _requests[key] = reply; event.set(); _trim(_requests, MAX_REQUEST_CACHE)
    return reply


def _key(game):
    stock = tuple(game.stock) if game.stock_known else game.stock
    return (tuple(game.board), game.waste, game.stock_known, stock,
            tuple(sorted(game.removed)), bool(game.joker_in_stock))


def _advice(game, sims=1200, seed=None):
    return advice_for(ui.Session(game), sims, seed, foresight=True)


def _background_work():
    while True:
        with _pre_condition:
            _pre_condition.wait_for(lambda: bool(_pre_jobs))
            _, (entry, pre, game) = _pre_jobs.popitem(last=False)
        try:
            with _cpu_slots:
                if not entry["active"] or entry.get("pre") is not pre: continue
                pre["res"] = _advice(game)
        except Exception:
            log.exception("Background advice failed")
        finally:
            pre["done"].set()


def _precompute(entry):
    """One worker, with at most one queued (latest) state for each live session."""
    global _pre_worker
    game = entry["s"].game
    if game.remaining() == 0 or ui.pending_reveals(game):
        entry["pre"] = None
        with _pre_condition: _pre_jobs.pop(entry["sid"], None)
        return
    key = _key(game); current = entry.get("pre")
    if current and current["key"] == key: return
    pre = {"key": key, "done": threading.Event(), "res": None}
    entry["pre"] = pre
    with _pre_condition:
        _pre_jobs[entry["sid"]] = (entry, pre, game.copy())
        if _pre_worker is None or not _pre_worker.is_alive():
            _pre_worker = threading.Thread(target=_background_work, name="tritowers-advice", daemon=True)
            _pre_worker.start()
        _pre_condition.notify()


def _adopt(entry):
    pre = entry.get("pre")
    if pre and pre["done"].is_set() and pre["res"] and pre["key"] == _key(entry["s"].game):
        entry["advice"] = pre["res"]


def _do(entry, op, body):
    session = entry["s"]
    if op == "play": ui.do_play(session, body.get("pos"))
    elif op == "reveal": ui.do_reveal(session, body.get("pos"), _text(body.get("rank", ""), "Rank", 16))
    elif op == "draw":
        rank = "*" if session.game.joker_in_stock and session.game.stock_remaining == 1 else _text(body.get("rank", ""), "Rank", 16)
        ui.do_draw(session, rank)
    elif op == "undo": session.undo()
    elif op == "state": _adopt(entry); return ""
    elif op == "recommend":
        _adopt(entry)
        if entry.get("advice"): return ""
        sims = min(MAX_SIMULATIONS, ui.to_int(body.get("sims", 1200), "Simulations", 1))
        seed = body.get("seed")
        if seed not in (None, ""): seed = ui.to_int(seed, "Seed")
        pre = entry.get("pre")
        if pre and pre["key"] == _key(session.game) and not pre["done"].is_set():
            with _pre_condition:
                if entry["sid"] in _pre_jobs:
                    _pre_jobs.move_to_end(entry["sid"], last=False)
            raise RetryLater("Advice is being calculated. Please retry shortly.")
        if not _cpu_slots.acquire(blocking=False): raise RetryLater()
        try: entry["advice"] = _advice(session.game.copy(), sims, seed)
        finally: _cpu_slots.release()
        return ""
    else: raise ValueError("Unknown action.")
    entry["advice"] = None
    _precompute(entry)
    return session.log[-1]

@app.post("/api/act")
def api_act(body: dict):
    entry = _get(body.get("sid", ""))
    if not entry: return {"ok": False, "gone": True, "message": "This game expired on the server. Start again."}
    try: rid = _request_id(body)
    except ValueError as error: return {"ok": False, "message": str(error)}
    op = body.get("op")
    if not isinstance(op, str): return {"ok": False, "message": "Action must be text."}
    if op == "state":
        if not entry["lock"].acquire(blocking=False): return _retry()
        try:
            _adopt(entry)
            return view(entry)
        finally: entry["lock"].release()
    event, cached = _claim(entry["replies"], rid, MAX_REPLY_CACHE)
    if cached is not None:
        if "cells" in cached:
            if not entry["lock"].acquire(blocking=False): return _retry()
            try: return view(entry, cached.get("message", ""), ok=cached.get("ok") is True)
            finally: entry["lock"].release()
        return cached
    if not entry["lock"].acquire(blocking=False):
        return _finish(entry["replies"], rid, event, _retry(), MAX_REPLY_CACHE)
    try:
        try:
            if rid in entry["applied"]:
                reply = view(entry, entry["applied"][rid])
            else:
                if op in MUTATING_ACTIONS and len(entry["applied"]) >= MAX_SESSION_ACTIONS:
                    raise ValueError("This game has reached its action limit. Start a new game using the current board.")
                message = _do(entry, op, body)
                if op in MUTATING_ACTIONS: entry["applied"][rid] = message
                reply = view(entry, message)
        except RetryLater as error: reply = _retry(str(error) or "The server is busy. Please retry shortly.")
        except ValueError as error: reply = view(entry, str(error), ok=False)
        except Exception:
            log.exception("Game action failed")
            reply = view(entry, "Could not finish that action. Refresh the game and try again.", ok=False)
    finally: entry["lock"].release()
    return _finish(entry["replies"], rid, event, reply, MAX_REPLY_CACHE)

@app.post("/api/solve")
def api_solve(body: dict, request: Request):
    try: rid = _request_id(body)
    except ValueError as error: return {"ok": False, "message": str(error)}
    key = (_client(request), "solve", rid)
    event, cached = _claim(_requests, key, MAX_REQUEST_CACHE)
    if cached is not None: return cached
    if not _cpu_slots.acquire(blocking=False):
        return _finish(_requests, key, event, _retry(), MAX_REQUEST_CACHE)
    try:
        try: reply = solve_deal(body)
        except ValueError as error: reply = {"ok": False, "message": str(error)}
        except Exception:
            log.exception("Exact solve failed")
            reply = {"ok": False, "message": "Could not solve this deal. Check the entries and try again."}
    finally: _cpu_slots.release()
    return _finish(_requests, key, event, reply, MAX_REQUEST_CACHE)


@app.post("/api/photo")
def api_photo(file: UploadFile = File(...), corners: str = Form("")):
    """Bounded upload, decoded off the event loop; one photo worker per process."""
    from tritowers_vision.image import MAX_BYTES
    if not _photo_slots.acquire(blocking=False): return _retry("Another photo is being read. Please retry shortly.")
    try:
        data = file.file.read(MAX_BYTES + 1)
        return photo_response(data, corners, _templates())
    finally: _photo_slots.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "7860")), log_level="warning")

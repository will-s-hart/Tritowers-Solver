"""Transport-independent web responses for the HTTP app and offline browser worker."""
import logging
import math
import solver
import solver_ui as ui

log = logging.getLogger(__name__)
MAX_SOLVE_SECONDS = 10.0

def _text(value, name, maximum=1024):
    if not isinstance(value, str): raise ValueError(f"{name} must be text.")
    if len(value) > maximum: raise ValueError(f"{name} is too long.")
    return value


def geometry():
    xs = {p: float(p - 19) for p in range(19, 29)}
    for p in sorted(solver.BLOCKERS, reverse=True): xs[p] = sum(xs[b] for b in solver.BLOCKERS[p]) / 2
    ys = {p: next(i for i, row in enumerate(ui.ROWS) if p in row) for p in range(1, 29)}
    XU, YU, W, H = 58, 76, 52, 68
    CW, CH = 9 * XU + W, 3 * YU + H
    return {p: (round(xs[p] * XU / CW * 100, 3), round(ys[p] * YU / CH * 100, 3)) for p in range(1, 29)}, round(CW / CH, 4)
GEO, ASPECT = geometry()

def next_action(game, advice=None):
    """The next step the app recommends, stated explicitly; never inferred from null fields.

    {"type": "done"} when the tableau is clear, {"type": "reveal", "positions": [...]}
    while uncovered cards await their ranks, the current advice's action when there is
    one, a forced {"type": "draw"} when no card can be played, otherwise {"type": "none"}.
    can_draw in the view states permission only; this states the recommendation.
    """
    if game.remaining() == 0: return {"type": "done", "reason": "won"}
    pending = ui.pending_reveals(game)
    if pending: return {"type": "reveal", "positions": pending}
    if advice and advice.get("action"): return dict(advice["action"])
    if not game.legal_moves():
        return {"type": "draw", "forced": True} if game.stock_remaining > 0 else {"type": "none", "reason": "no_moves"}
    return {"type": "none", "reason": "awaiting_advice"}

def stock_next(game):
    """The next stock card when its rank is known (a known-order stock, or the machine joker left last), else None."""
    if game.stock_known: return game.stock[0] if game.stock else None
    return solver.JOKER if game.joker_in_stock and game.stock_remaining == 1 else None

def advice_for(session, sims, seed=None, foresight=False):
    """Advice for the current state, with its action stated explicitly (HTTP and browser worker)."""
    text, pos, proven, rate, runs, action = ui.recommend_action(session, sims, seed)
    if pos: text = text.replace(f"Play position {pos:02d}.", f"Play the {session.game.board[pos - 1]} marked with the blue star.")
    return {"text": text, "pos": pos, "proven": proven, "rate": rate, "sims": runs, "action": action,
            "foresight": ui.foresight_line(session, pos, seed) if foresight and pos else None}

def view(entry, message="", ok=True, extra=None):
    s = entry["s"]; g = s.game; snap = g.state_snapshot(); exposed = set(g.exposed()); legal = set(g.legal_moves()); pend = set(ui.pending_reveals(g))
    advice = entry.get("advice")
    cells = []
    for p in range(1, 29):
        if p in snap["removed"]: kind, card = "gone", ""
        elif snap["board"][p - 1] == "?": kind, card = ("ask" if p in pend else "back"), "?"
        else: kind, card = ("play" if p in legal else "up" if p in exposed else "blocked"), snap["board"][p - 1]
        cells.append({"p": p, "card": card, "kind": kind, "x": GEO[p][0], "y": GEO[p][1]})
    over = g.remaining() == 0
    out = {"ok": ok, "message": message, "cells": cells, "aspect": ASPECT, "waste": snap["waste"], "remaining": snap["remaining"],
           "stock": snap["stock_remaining"], "joker": bool(snap.get("joker_in_stock")), "status": ui.status(s), "pending": sorted(pend), "over": over,
           "can_undo": bool(s.history), "can_draw": (not over) and (not pend) and snap["stock_remaining"] > 0,
           "stock_known": bool(g.stock_known), "stock_next": stock_next(g), "next_action": next_action(g, advice),
           "log": s.log[-12:], "advice": advice, "rev": len(s.log)}
    if extra: out.update(extra)
    return out

def _replay_frames(game, moves):
    """Replay a returned line on a fresh copy and return (frames, texts). Raises ValueError if any step is illegal."""
    g = game.copy(); frames = []; texts = []
    def frame(g, nxt, action):
        f = view({"s": ui.Session(g.copy()), "advice": None}); f.pop("log", None); f.pop("advice", None)
        # next (position or None) is kept for older clients; next_action is explicit.
        f["next"] = nxt; f["next_action"] = action; return f
    for mv in moves:
        nxt = mv[1] if mv[0] == "play" else None
        if mv[0] == "play": action = {"type": "play", "pos": int(mv[1])}
        elif mv[0] == "draw": action = {"type": "draw", "card": g.stock[0] if g.stock else None}
        else: action = None
        frames.append(frame(g, nxt, action))
        if mv[0] == "play":
            card = g.play(int(mv[1])); texts.append(f"Play the {card} (position {int(mv[1])})")
        elif mv[0] == "draw":
            if not g.stock: raise ValueError("Line draws from an empty stock.")
            card = g.stock.pop(0); g.waste = card; texts.append(f"Draw from the stock: {card}")
        else: raise ValueError("Unknown step.")
    frames.append(frame(g, None, {"type": "done", "reason": "won"}))
    if g.remaining() != 0: raise ValueError("Line does not clear the tableau.")
    return frames, texts

def solve_deal(body):
    """Complete-deal mode: every card known, stock order known. Never guesses missing cards."""
    board = _text(body.get("board", ""), "Board").replace(",", " ").split(); waste = _text(body.get("waste", ""), "Waste", 16).strip()
    raw_stock = body.get("stock")
    if raw_stock is None: raw_stock = []
    if not isinstance(raw_stock, (list, tuple)): return {"ok": False, "message": "Stock must be a list of cards in draw order."}
    if len(raw_stock) > 24: return {"ok": False, "message": "Stock can contain at most 24 cards."}
    stock = [_text(x, "Stock rank", 16).strip() for x in raw_stock]
    try:
        stock_count = None if body.get("stock_count") is None else ui.to_int(body.get("stock_count"), "Stock count", 0, 24)
        budget_in = float(body.get("time_budget", MAX_SOLVE_SECONDS))
        if not math.isfinite(budget_in): raise ValueError("Time budget must be finite.")
    except (ValueError, TypeError) as e: return {"ok": False, "message": str(e) if "Stock count" in str(e) else "Time budget must be a finite number."}
    missing = []
    if len(board) != 28: return {"ok": False, "message": f"Need 28 board entries, got {len(board)}."}
    missing += [f"tableau position {i}" for i, c in enumerate(board, 1) if c == "?"]
    if not waste or waste == "?": missing.append("waste card")
    missing += [f"stock card {i}" for i, c in enumerate(stock, 1) if c == "?"]
    if stock_count is not None and stock_count != len(stock): missing.append(f"stock order ({len(stock)} of {stock_count} cards entered)")
    if missing:
        return {"ok": True, "status": "incomplete", "message": "The deal is not complete, so I will not guess. Missing: " + ", ".join(missing[:12]) + (" ..." if len(missing) > 12 else "") + ".", "missing": missing}
    try: game = solver.Game(board, waste, True, stock)
    except ValueError as e: return {"ok": False, "message": str(e)}
    fn = getattr(solver, "solve_complete", None)
    if fn is None: return {"ok": True, "status": "unavailable", "message": "The exact solver is not installed in this build yet."}
    budget = max(0.0, min(budget_in, MAX_SOLVE_SECONDS))
    searched = []
    def run(stuck, share):
        result = fn(game.copy(), time_budget=budget * share, require_full_deal=False, draw_only_when_stuck=stuck)
        searched.append(result)
        return result
    try:
        res = run(True, 0.5); policy = "stuck"
        if res.status in ("unsolvable", "unknown"):
            res2 = run(False, 0.5)
            if res2.status == "solved": res, policy = res2, "voluntary"
            elif res.status == "unsolvable" and res2.status == "unsolvable": policy = "both"
            elif res.status == "unsolvable": res = res2; policy = "voluntary"   # voluntary undecided: say so below
            elif res2.status == "unsolvable": res = res2; policy = "both"                    # voluntary unsolvable implies stuck-only unsolvable
    except Exception:
        log.exception("Exact solver failed")
        return {"ok": False, "message": "The solver could not finish this deal. Please try again."}
    out = {"ok": True, "status": res.status, "policy": policy, "nodes": sum(getattr(r, "nodes", 0) or 0 for r in searched), "seconds": sum(getattr(r, "seconds", 0) or 0 for r in searched)}
    if res.status == "solved":
        try: frames, texts = _replay_frames(game, list(res.moves))
        except Exception:
            log.exception("Exact line verification failed")
            return {"ok": False, "message": "Solver returned a line that failed verification. Not shown."}
        note = ("Valid whether or not the machine lets you draw while a play is available." if policy == "stuck" else
                "Only works if the machine lets you draw while a play is available (not verified).")
        out.update(message=f"Solved in {len(texts)} steps (line verified by replay). {note}", steps=texts, frames=frames)
    elif res.status == "unsolvable" and policy == "both": out["message"] = "Proven unsolvable under both draw rules: the search was exhaustive and no winning line exists."
    elif res.status == "unsolvable": out["message"] = "No line wins if you may only draw when no play is available. Whether voluntary draws would help was not decided, so this is NOT a full proof."; out["status"] = "unknown"
    elif res.status == "unknown": out["message"] = f"Could not decide in the time limit ({getattr(res, 'reason', 'timeout')}). This is NOT a proof that it is unsolvable."
    else: out["message"] = "The deal is not valid or complete for the solver: " + str(getattr(res, "reason", ""))
    return out

def photo_response(data, corners="", templates=None):
    """Read a photo into the same draft response in HTTP and the browser worker."""
    import base64, io
    from tritowers_vision.image import MAX_BYTES, ImageInputError
    from tritowers_vision.reader import board_tokens
    from tritowers_vision.intake import read_photo
    if len(data) > MAX_BYTES: return {"ok": False, "message": f"Image is too large ({MAX_BYTES // (1024 * 1024)} MB max)."}
    manual = None
    try: corners = _text(corners, "Corners", 256)
    except ValueError as error: return {"ok": False, "message": str(error)}
    if corners.strip():
        try: nums = [float(x) for x in corners.replace(";", ",").split(",") if x.strip()]
        except ValueError: return {"ok": False, "message": "Corners need 8 numbers."}
        if len(nums) != 8: return {"ok": False, "message": "Corners need 8 numbers."}
        manual = [(nums[i], nums[i + 1]) for i in range(0, 8, 2)]
    try: r = read_photo(data, templates or [], manual)
    except ImageInputError as error:
        return {"ok": False, "message": f"Could not read that image: {error}"}
    except Exception:
        log.exception("Photo recognition failed")
        return {"ok": False, "message": "Could not read that image. Try a clear JPEG, PNG or HEIC photo of the whole screen."}
    d = r.draft; tokens, waste = board_tokens(d)
    def review_id(slot):
        if slot.startswith("tableau-"): return str(int(slot.split("-")[1]))
        if slot.startswith("stock-"): return "stock-" + str(int(slot.split("-")[1]))
        return slot
    review = [review_id(s) for s in d["needs_human_review"]]
    full_deal = d.get("photo_kind") == "full_deal"
    if not d["registration"]["trusted"]:
        note = "The card layout was not found reliably, so check every card (a straighter photo of the whole screen helps). "
    else:
        note = ("Check: " + ", ".join(review) + ". " if review else "")
    if full_deal:
        note += "Full-deal grid: the first two rows fill the 28 board positions. The rightmost bottom card is the waste. Stock runs right to left, with the joker last. Check every rank and this draw order before solving."
    else:
        note += "Suit and the stock counter are not read: set the stock yourself. Check the picture before you start."
    view = r.overlay.copy(); view.thumbnail((900, 900))
    buf = io.BytesIO(); view.save(buf, format="JPEG", quality=80)
    response = {"ok": True, "mode": "full_deal" if full_deal else "tableau",
                "board": tokens, "waste": waste, "note": note, "review": review,
                "trusted": d["registration"]["trusted"], "overlay": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}
    if full_deal:
        response.update(stock=list(d["stock"]), stock_count=len(d["stock"]),
                        joker=True, draw_direction="right_to_left")
    return response

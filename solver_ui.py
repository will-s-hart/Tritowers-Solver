"""Friendly Gradio front end for the rank-only solver. Rules stay in solver.py; this only presents them."""
import random, html, time
import solver
ROWS = ((1, 2, 3), tuple(range(4, 10)), tuple(range(10, 19)), tuple(range(19, 29)))

class Session:
    def __init__(self, game): self.game, self.history, self.log = game, [], []
    def checkpoint(self): self.history.append(self.game.copy())
    def undo(self):
        if not self.history: raise ValueError("Nothing to undo.")
        self.game = self.history.pop(); self.log.append("Undid last action.")

def to_int(value, name, lo=None, hi=None):
    """Strict integer: rejects fractions, bools, blanks, text. 23.0 is accepted, 23.9 is not."""
    if isinstance(value, bool) or value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{name} must be a whole number.")
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a whole number.")
    if f != f or f in (float("inf"), float("-inf")) or f != int(f):
        raise ValueError(f"{name} must be a whole number, got {value}.")
    n = int(f)
    if lo is not None and n < lo or hi is not None and n > hi:
        raise ValueError(f"{name} must be between {lo} and {hi}, got {n}.")
    return n

MAX_STOCK = 23
MAX_SIMULATIONS = 2000
RECOMMEND_TIME_BUDGET = 2.0

def guard_open(session, need_no_reveals=True):
    g = session.game
    if g.remaining() == 0: raise ValueError("Tableau is cleared: the game is over.")
    if need_no_reveals and pending_reveals(g): raise ValueError("Reveal the yellow ? cards first.")

def parse_board(text):
    if not isinstance(text, str): raise ValueError("Board must be text containing 28 card entries.")
    tokens = [t for t in text.replace(",", " ").split() if t]
    if len(tokens) != 28: raise ValueError(f"Need 28 board entries (positions 1-28), got {len(tokens)}. Use ? for a covered card and -- for an empty slot.")
    return tokens

def new_session(board_text, waste, stock, joker=False):
    """Create a play-along session; stock includes the fixed last joker when opted in."""
    if not isinstance(joker, bool):
        raise ValueError("Joker mode must be true or false.")
    board = parse_board(board_text)
    if not isinstance(waste, str) or not waste.strip(): raise ValueError("Choose the current waste rank.")
    stock = to_int(stock, "Stock", 0, MAX_STOCK + int(joker))
    return Session(solver.Game(board, waste, False, stock, joker_in_stock=joker))

def pending_reveals(game):
    return [p for p in game.exposed() if game.board[p - 1] == "?"]

def render_board(session):
    g = session.game; s = g.state_snapshot(); exposed = set(g.exposed()); legal = set(g.legal_moves()); reveals = set(pending_reveals(g))
    def cell(p):
        card = "" if p in s["removed"] else s["board"][p - 1]
        if p in s["removed"]: cls, label = "gone", ""
        elif card == "?": cls, label = ("ask" if p in reveals else "back"), ("?" if p in reveals else "")
        else: cls, label = ("play" if p in legal else "up" if p in exposed else "blocked"), html.escape(card)
        return f'<div class="c {cls}"><small>{p}</small>{label}</div>'
    W, H, XU, YU = 52, 68, 58, 76
    CW, CH = 9 * XU + W, 3 * YU + H
    xs = {p: float(p - 19) for p in range(19, 29)}
    for p in sorted(solver.BLOCKERS, reverse=True): xs[p] = sum(xs[b] for b in solver.BLOCKERS[p]) / 2
    ys = {p: next(i for i, row in enumerate(ROWS) if p in row) for p in range(1, 29)}
    cells = "".join(cell(p).replace('<div class="c ', f'<div style="left:{xs[p]*XU/CW*100:.3f}%;top:{ys[p]*YU/CH*100:.3f}%" class="c ', 1) for p in range(1, 29))
    rows = f'<div class="tw"><div class="towers" style="aspect-ratio:{CW}/{CH}">{cells}</div></div>'
    css = ("<style>.tw{container-type:inline-size;width:100%;max-width:574px;margin:0 auto}.towers{position:relative;width:100%}.c{position:absolute;width:9.06%;height:22.97%;border-radius:6px;border:1px solid #888;display:flex;"
           "flex-direction:column;align-items:center;justify-content:center;font:700 clamp(11px,3.6cqw,20px) sans-serif}.c small{position:absolute;top:2px;left:4px;font:clamp(7px,1.8cqw,10px) sans-serif;opacity:.6}"
           ".gone{border-style:dashed;opacity:.25}.back{background:#a33;color:#fff}.ask{background:#fc6;color:#000}.up{background:#f6f1e3;color:#222}.play{background:#cfe9c8;color:#111;border:2px solid #2a7}.blocked{background:#ddd;color:#555}</style>")
    foot = (f'<div style="text-align:center;margin-top:8px">Waste <b>{s["waste"]}</b> &nbsp;|&nbsp; Tableau left <b>{s["remaining"]}</b> &nbsp;|&nbsp; Stock left <b>{s["stock_remaining"]}</b></div>'
            '<div style="text-align:center;font-size:12px;opacity:.7">green = playable now, yellow ? = tell me this card, red = still covered</div>')
    return css + rows + foot

def status(session):
    g = session.game; need = pending_reveals(g)
    if need: return "Reveal needed: " + ", ".join(f"{p:02d}" for p in need)
    if g.remaining() == 0: return "Tableau cleared."
    if not g.legal_moves(): return "No playable card. Draw from the stock." if g.stock_remaining > 0 else "No moves and the stock is empty."
    return "Ready for a recommendation."

ESTIMATE_SAMPLES = 400
ESTIMATE_BUDGET = 8.0   # seconds of wall clock for the whole estimate
SOLVE_BUDGET = 0.25     # per exact solve; an undecided sample counts as NOT winnable

def _completion(game, pool, hidden, rng):
    """One random full completion of the hidden tableau cards and the unknown stock order."""
    cards = pool[:]; rng.shuffle(cards)
    g = game.copy(); k = 0
    for i in hidden: g.board[i] = cards[k]; k += 1
    if not g.stock_known:
        # The joker is a fixed stock tail, never a shuffled tableau/rank-pool card.
        g.stock = cards[k:] + ([solver.JOKER] if g.joker_in_stock else [])
        g.stock_known = True
        g.joker_in_stock = False
    return g

def estimate_moves(game, moves, samples=ESTIMATE_SAMPLES, budget=ESTIMATE_BUDGET, rng=None):
    """Sample random completions of the hidden cards and exactly solve each one after each candidate move.

    Returns None when the card counts are inconsistent. Otherwise {pos: {"won","lost","unknown","n"}}.
    Each completion is a full-information game, so the result assumes perfect foresight: an estimate of
    the share of completions that are winnable after the move, not a bound and not a playable plan.
    Draws only when stuck (valid under both draw rules). Undecided solves are counted as not winnable.
    """
    rng = rng or random.Random()
    hidden = [i for i, c in enumerate(game.board) if c == "?" and (i + 1) not in game.removed]
    pool = game.unknown_card_pool()
    stock_n = 0 if game.stock_known else game.stock_remaining - int(game.joker_in_stock)
    if not game.stock_known and len(pool) != len(hidden) + stock_n: return None
    if game.stock_known and hidden and len(pool) != len(hidden): return None
    stats = {p: {"won": 0, "lost": 0, "unknown": 0, "n": 0} for p in moves}
    deadline = time.monotonic() + budget; done = 0
    single = not hidden and game.stock_known
    while done < (1 if single else samples) and time.monotonic() < deadline:
        comp = _completion(game, pool, hidden, rng); done += 1
        for p in moves:
            g = comp.copy()
            try: g.play(p)
            except Exception: continue
            r = solver.solve_complete(g, time_budget=min(SOLVE_BUDGET, max(0, deadline - time.monotonic())), draw_only_when_stuck=True)
            st = stats[p]; st["n"] += 1
            if r.status == "solved": st["won"] += 1
            elif r.status == "unsolvable": st["lost"] += 1
            else: st["unknown"] += 1
    return stats

def foresight_line(session, position, seed=None):
    """Separate, clearly labelled perfect-information estimate for one candidate move (never a bound or a plan)."""
    g = session.game
    rng = random.Random(to_int(seed, "Seed")) if seed not in (None, "") else None
    stats = estimate_moves(g, [position], rng=rng)
    if stats is None: return None
    st = stats[position]; n = st["n"]
    if not n: return None
    known = not any(c == "?" for c in g.board) and g.stock_known
    if known:
        if st["won"]: return "Exact: with every card known, a winning line exists after this move (draws only when stuck)."
        if st["unknown"]: return "Could not decide this position in the time limit; not a proof either way."
        return "Exact: with every card known, no winning line exists after this move (draws only when stuck)."
    extra = f" {st['unknown']} undecided samples are counted as not winnable." if st["unknown"] else ""
    return (f"Perfect-information view: winnable in {st['won'] / n:.0%} of {n} sampled completions of the hidden cards, assuming perfect foresight "
            f"and draws only when stuck. An estimate, not a bound or a plan, and optimistic compared with real play.{extra}")

def recommend_detail(session, simulations=solver.SIMULATIONS, seed=None):
    """Return (text, position|None, proven|None, rate|None, sims)."""
    return recommend_action(session, simulations, seed)[:5]

def recommend_action(session, simulations=solver.SIMULATIONS, seed=None):
    """Return (text, position|None, proven|None, rate|None, sims, action).

    action states the recommendation explicitly, so callers never infer it
    from the text or from a missing position: {"type": "play", "pos": p},
    {"type": "draw"} (with "forced": True when no card can be played), or
    {"type": "none", "reason": ...} when nothing is recommended.
    """
    guard_open(session)
    sims = to_int(simulations, "Simulations", 1, MAX_SIMULATIONS)
    rng = random.Random(to_int(seed, "Seed")) if seed not in (None, "") else None
    game = session.game
    if game.stock_known and not pending_reveals(game) and all(
            c != "?" for p, c in enumerate(game.board, 1) if p not in game.removed):
        exact = solver.solve_complete(game, time_budget=None if rng is not None else RECOMMEND_TIME_BUDGET,
                                      max_nodes=200_000)
        if exact.status == "solved" and exact.moves:
            step = exact.moves[0]
            if step[0] == "draw":
                return ("Draw from the stock. Proven: a verified winning line starts with this draw.", None, True, 1.0, 0,
                        {"type": "draw", "forced": not game.legal_moves()})
            return (f"Play position {step[1]:02d}. Proven: a verified winning line starts with this move.", step[1], True, 1.0, 0,
                    {"type": "play", "pos": step[1]})
        if exact.status == "unsolvable":
            return ("No winning line exists with these known cards (draw only when stuck).", None, False, 0.0, 0,
                    {"type": "none", "reason": "no_winning_line"})
    rec = solver.best_move(game, simulations=sims, rng=rng,
                           time_budget=None if rng is not None else RECOMMEND_TIME_BUDGET)
    if rec is None:
        if session.game.stock_remaining:
            return "No legal move: draw from the stock.", None, None, None, 0, {"type": "draw", "forced": True}
        return "No legal move and stock empty.", None, None, None, 0, {"type": "none", "reason": "no_moves"}
    action = {"type": "play", "pos": rec.position}
    if rec.is_proven: return f"Play position {rec.position:02d}. Proven: this move is guaranteed by the known cards.", rec.position, True, 1.0, 0, action
    return (f"Play position {rec.position:02d}. Sampled estimate {rec.success_rate:.0%} over {rec.simulations} simulations. This is an estimate, not a proof."), rec.position, False, rec.success_rate, rec.simulations, action

def recommend(session, simulations=solver.SIMULATIONS, seed=None):
    return recommend_detail(session, simulations, seed)[0]

def do_play(session, position):
    position = to_int(position, "Position", 1, 28); guard_open(session)
    session.checkpoint()
    try: card = session.game.play(int(position))
    except Exception: session.history.pop(); raise
    session.log.append(f"Played {int(position):02d} ({card}).")

def do_reveal(session, position, rank):
    position = to_int(position, "Position", 1, 28)
    if position not in pending_reveals(session.game): raise ValueError(f"Position {position:02d} is not waiting for a reveal.")
    session.checkpoint()
    try: card = session.game.observe_rank(rank); session.game.board[position - 1] = card
    except Exception: session.history.pop(); raise
    session.log.append(f"Revealed {position:02d} = {card}.")

def do_draw(session, rank):
    guard_open(session)
    if session.game.stock_empty: raise ValueError("Stock is empty.")
    if not session.game.stock_known and (not isinstance(rank, str) or not rank.strip()):
        raise ValueError("Choose the rank of the drawn card.")
    session.checkpoint()
    try:
        if session.game.stock_known: session.game.draw_known()
        else: session.game.observe_draw(rank)
    except Exception: session.history.pop(); raise
    session.log.append(f"Drew {session.game.waste}.")

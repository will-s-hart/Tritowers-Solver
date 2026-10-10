# Tritowers Solver

Recommends the next move in the Tri Towers pub-quiz-machine card game. Only card ranks matter, suits are ignored.

## Run

    python3 solver.py                 # with the startup tutorial
    python3 solver.py --skip-tutorial
    python3 solver.py --seed 1 --simulations 300
    python3 solver.py --time-budget 5

Options: `--seed` seeds the sampling, so the same entries and seed give the same output (tested for one scripted game; not a guarantee across Python versions), `--simulations` sets runs per candidate move (default 1200), `--seed` and `--time-budget` are mutually exclusive: use a seed for repeatable fixed-work sampling, or a time budget for a responsive timed recommendation. `--time-budget` is a soft cap in seconds on sampling per recommendation: it is shared between candidates and candidates get equal numbers of runs; an in-flight round can exceed it, `--skip-tutorial` hides the intro. Python 3.10+, no dependencies.

## Play

1. Say whether you know all 28 tableau cards and whether you know the stock order.
2. Enter ranks (`A 2 ... 10 J Q K`), use `--` for already-cleared positions.
3. The solver prints the board each turn, then `PLAY <card> @ <position>` with `[GUARANTEED]` or a verified `PROVEN` exact route for a known deal, or the sampled win rate and run count. Enter any newly exposed or drawn card when asked.
4. After setup the board is shown once more. Press Enter to start, or correct a mistyped entry first: `fix 07 K` (position, rank; `--` cleared, `?` unknown), `waste Q`, or `stock 5 9` (known stock only, 1 = next card drawn), or `stock 12` to correct an unknown-stock count. A correction that breaks the deck rules is rejected and nothing changes.
5. Type `undo` (or `u`) at a card prompt to return to the previous input prompt, including across automatic plays and known-stock draws. Re-enter the mistaken card there. The sampler state is restored too, so after undo the session replays as if the step had not happened (same seed, same entries). Undo does not apply during setup, use the correction step above.
6. Ctrl-D ends the session cleanly.

## Test

    python3 -m pip install -r requirements.txt pytest httpx
    python3 -m pytest -q -m "not browser"

For browser tests, install `playwright` and run `python -m playwright install chromium`, then `python -m pytest -q -m browser`. CI also builds and exercises the real Pyodide worker.

`test_e2e_cli.py` drives the real `solver.py` through stdin. `python tools/heuristic_baseline.py --deals 10000 --seed 11` compares the move heuristic with simple policies.

If setup cards cannot be a real deck (a rank entered more than four times), the solver says where, for example `K was entered 5 times: position 01, position 07 ...`, and asks for a correction with the same `fix` / `waste` / `stock` commands instead of making you start over. Ctrl-D still ends the session.

## Photo reader

`app.py` (Gradio, "Photo to board" tab) and `web_app.py` (`/api/photo`, the "Read from a photo" button) turn a
screenshot or phone photo of the machine screen into an editable draft. Photo intake automatically distinguishes
the normal three-tower board from the full-deal grid. Both readers return an overlay and mark unreadable ranks `?`.
They are calibrated on one skin (see `MODEL_CARD.md`); hidden ranks are never guessed.

The automatic path needs no bank upload or row calibration. Browser photos are oriented and resized to a 1600px long edge before recognition, with an optional software HEIC decoder running locally in its own worker. The HTTP path also accepts HEIC through Pillow. Uncertain geometry stays unknown, and the board highlights corrections before one **Confirm board & start** action enters play-along. Set the stock counter yourself; a tableau photo cannot reveal the stock order.

For the **grid showing all cards**, choose the photo in the same upload control. Its first two rows (14 cards each)
fill the 28 tableau positions. The bottom row's rightmost card is the waste; the other 23 normal cards, read right
to left, form the stock with the leftmost joker last. This direction is inferred from the machine layout: check
it against your game. The app switches to **I know the whole deal**, shows numbered stock cards in draw order,
and lets you correct every rank before **Confirm deal & solve**. Unknown board, waste or stock ranks block solving
until corrected. In Gradio, use **Copy reviewed draft to Known deal tab** after reviewing the photo output.

The recognizer ships a bank rendered from open fonts, never owner photos or photo-derived templates. `TT_FONT_TIER=0` disables font matching. `TT_TEMPLATES` remains an optional local research setting on the HTTP adapter; the static app has no private-bank loader.

Measure no-bank recognition with `python tools/vision_eval.py --labels local/labels.json --readers v3 --regimes none`. The labels format is in `DATASET.md`; private captures and metadata belong under ignored `local/`. See `PHOTO-VALIDATION.md` for the issue crosswalk, measured results and remaining validation limits.

## Play view

Below the towers, a large face-up waste card sits beside the stock pile and its remaining count. The next stock card
is shown only when known (a known deal, or the joker as the last card); otherwise the pile stays face down. When the
next recommended action is a draw, the pile is outlined with a "Draw next" label (no animation under reduced motion),
and tapping it draws, or steps the solution replay over that draw. API views and replay frames state this as
`next_action` (`play`, `draw`, `reveal`, `done` or `none`) with `stock_next`; `can_draw` only means drawing is allowed.

## Browser-only app

The static app runs the same Python solver and photo response adapter in a Pyodide worker. It starts loading immediately and never probes the retired HTTP origin. Build without deploying:

    python tools/prepare_static.py /tmp/tritowers-site --download-runtime
    python -m http.server 7861 --directory /tmp/tritowers-site

Open `http://localhost:7861`. The build pins Pyodide and the HEIC codec, verifies package checksums and includes only allowlisted public runtime files. The first visit downloads the runtime; selected photos and game state remain in the browser. Existing static hosting can serve this directory without restarting an origin. No hosting or paid resources are created by this command.

    TT_STATIC_SITE=/tmp/tritowers-site python -m pytest -q tests/test_static_browser.py

For the alternate local HTTP/Gradio app, run `python app.py` and open port 7860; `/` is the mobile app and `/gradio` is Gradio. `HF-DEPLOY.md` describes the optional Docker Space package. Neither command deploys anything.

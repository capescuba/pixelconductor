# Handoff: song-repeat sequence extension for PixelConductor

Copy this file, `xsq_extend.py`, and the two plan JSONs into `E:\Code\PixelConductor\` and
start Claude Code there. Suggested first message is at the bottom.

## What this feature does

Given an MP3 and a partially hand-sequenced xLights `.xsq` (written by PixelConductor),
extend the sequence by repeating the hand-sequenced material wherever the song repeats.
Prototyped on "Wizards in Winter" against `back-yard-master.xsq` (8 snowflakes, 8 strings,
4 icicles, submodels Hub / Drop 1 / Drop 2 / "/ Lines" / "\ Lines"). 648 effects in → ~2,600 out,
zero changes to existing effects, verified no overlaps.

## Engine: `xsq_extend.py` (done, tested)

Three pure functions plus a JSON plan the UI edits:

    a    = analyze(mp3)                       # ~20 s; beats, similarity matrix, downbeat phase, accents. Cache per song.
    plan = propose_plan(a, xsq, cutoff_s=None) # Viterbi bar-level lag mapping; returns sections with mode/src/lag/confidence
    stats = apply_plan(xsq, a, plan, out)     # beat-relative copy; skips collisions; renumbers ids; adds timing tracks

Plan section modes: `copy` (src_s, src_e, lag_beats), `tile` (src block + cell_beats),
`extend` (one model's effects → all models of its class), `dark`, `ending` (generated from accents).
Optional `models` / `exclude` lists per section. Times in seconds; engine works in beats internally
so copies stay locked through tempo drift.

Deps: numpy, librosa (+ ffmpeg on PATH for MP3). Effect times use xLights ms convention;
first effect in a layer has no `id`, rest are 1..n. Timing tracks are written as
`<Element type="timing">` in both DisplayElements and ElementEffects.

## Lessons from the prototype (drive the UI design)

- Automatic matcher nails true repeats (conf 0.87–0.92) but needs human decisions for:
  full-band riff sections with no timbral analog (chose to *tile* the intro pattern),
  sliding a build/chase so it lands on the drop (manual lag), sketches on one fence that
  should be *extended*, and endings/breakdowns (generated or left dark).
- `sequenced_cutoff()` heuristic (last beat with ≥3 models sequenced in an 8-beat window)
  guessed 65 s; the real "where I got to" was 50.4 s. Must be visible and overridable.
- Sections in a plan may overlap (several ops on one range) — timing marks are merged.
- Compare `plan-wizards-auto-proposed.json` (unaided) with `plan-wizards-in-winter.json`
  (hand-edited, reproduces the accepted output) to see the kind of edits the UI must support.

## Integration plan for PixelConductor (app.py + PixelConductor.html)

1. Backend endpoints (or whatever app.py's pattern is):
   - `POST /extend/analyze {mp3, xsq}` → caches `analyze()` result keyed by MP3 hash; returns plan from `propose_plan()`.
   - `POST /extend/apply {plan}` → runs `apply_plan()`, writes `<name>-extended.xsq` next to the original, returns stats.
   - Decide: if app.py owns effects in memory, have `apply_plan` return effect dicts instead of writing XML
     (Writer.added is already a dict keyed by (model, submodel)).
2. Review screen in PixelConductor.html:
   - waveform strip with beat grid, cutoff marker (draggable), section rows below.
   - each row: start/end (draggable), mode dropdown, source range picker (click-drag on the waveform
     before the cutoff), lag readout in beats, confidence badge, models/exclude chips.
   - "Apply" posts the plan; show added/skipped counts per section.
3. Keep original file untouched; always write a new filename.
4. Nice-to-have: reuse an existing timing track from the .xsq as the beat grid instead of re-detecting.

## Suggested first message to Claude Code

> Read HANDOFF-xsq-extend.md and xsq_extend.py. Then read app.py and PixelConductor.html and tell me
> how app.py currently builds .xsq files (in-memory effect model vs direct XML) and propose where the
> analyze/propose/apply endpoints and the review screen fit. Don't write code until I confirm the plan.

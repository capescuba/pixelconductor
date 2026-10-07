# PixelConductor

A browser-based light-show sequencer that works alongside [xLights](https://xlights.org).
Load a song, see it split into stems with beats, sections and onsets marked, place effects
on a timeline against your real display layout, and save straight back to an xLights
`.xsq` file. Open that file in xLights to fine-tune, render and run the show.

PixelConductor is a Flask server (`app.py`) plus a single-file web app
(`PixelConductor.html`). Everything runs locally at `http://localhost:7842`.

## Features

- **xLights round-trip.** Opens and saves xLights `.xsq` sequences. Each effect's full
  xLights settings string is preserved, so parameters you tune in xLights survive a trip
  through PixelConductor unchanged.
- **Deep audio analysis.** Beat and downbeat tracking, section detection, and 12 stems
  with per-stem onsets and spectrograms:
  - drums, bass, vocals, guitar, piano and other, from demucs `htdemucs_6s`
  - kick/snare and cymbals, from a harmonic/percussive split of the drums
  - bells, choir and strings, from a second demucs pass on the "other" stem
  - sub-bass, from a low-pass of the full mix
- **Timeline editor.** Snap to beat, multi-select, copy and paste, undo and redo, effects
  on whole models or on submodels, and model groups as collapsible track headers.
- **Live stage preview.** Your actual xLights layout drawn over its background image,
  animating whatever is under the playhead. It can pop out into its own window.
- **17 effects with an xLights property editor.** Each effect type has curated controls
  (sliders, dropdowns, checkboxes) whose ranges and option lists come from the xLights
  source, plus a live thumbnail preview of the selected effect and color.
- **Auto-arrange.** Sends the audio analysis and your prop list to Claude and lays out a
  first-pass sequence.
- **Extend.** Repeats the material you've already sequenced wherever the song repeats.
  You review and edit the proposed plan before anything is written.
- **Save safety.** Atomic writes, timestamped backups, server-side drafts and an optional
  30-second auto-save.

## Requirements

- **Windows.** The launcher and the default paths are Windows-specific. The server
  itself is plain Flask.
- **Python 3.13.** That's the version it's developed on; 3.11 or later should work.
- **[FFmpeg](https://ffmpeg.org/)** on your `PATH`, for MP3 decoding.
- **An NVIDIA GPU (optional).** demucs uses CUDA when it's available and falls back to
  the CPU, which is much slower.
- **An xLights show folder** containing `xlights_rgbeffects.xml` (your layout) and your
  sequences.

## Install

```bash
git clone https://github.com/capescuba/pixelconductor.git
cd pixelconductor
```

For GPU stem separation, install the CUDA build of PyTorch first. Pick the command for
your CUDA version at <https://pytorch.org/get-started/locally/>. For example:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then install everything else:

```bash
pip install -r requirements.txt
```

## Configure

PixelConductor reads two folders. Point them at your xLights setup with environment
variables:

| Variable | Default | What it is |
|---|---|---|
| `PC_SHOW_DIR` | `E:\Code\xlights` | xLights show folder (layout and sequences) |
| `PC_AUDIO_DIR` | `E:\Code\xlights\audio` | Folder of song audio files |

The server listens on port `7842`.

## Run

Start the server in a console and open <http://localhost:7842>:

```bash
python app.py
```

Or use the launcher, which starts the server in the background with no console window
(if it isn't already running) and opens your browser:

```bash
pythonw launch.pyw
```

```bash
pythonw launch.pyw --stop
```

The launcher writes server output to `logs/server.log`.

## Workflow

1. **Open XSQ** and pick a sequence. The layout loads from `xlights_rgbeffects.xml`
   and the song from the sequence's media file.
2. Wait for the first analysis. Stem separation takes a few minutes per song on a GPU,
   and the results are cached, so later loads are instant. **Re-analyze** clears the
   cache and starts over.
3. Choose an effect and color in the **Effects** panel, then drag on a model's track to
   place it. Double-click an effect to change its type, timing, color and xLights
   properties.
4. Press **Save** (Ctrl+S) to write the `.xsq` back in place, or **Export .xsq** to
   download a copy.
5. Open the sequence in xLights and render it before playing the show. The `.fseq`
   output comes from xLights, not PixelConductor.

> If you edit `xlights_rgbeffects.xml` outside xLights, fully restart xLights to pick up
> the change. A running xLights can overwrite external edits when it saves.

## Effects

PixelConductor effect names map to xLights effects as follows:

| PixelConductor | xLights effect | Notes |
|---|---|---|
| On | On | |
| Chase | SingleStrand | Chase tab |
| Rainbow | SingleStrand | FX tab, `Rainbow` mode |
| Pulse | On | Brightness ramp plus fade-in and fade-out |
| Color Wash | Color Wash | |
| Twinkle | Twinkle | |
| Fire | Fire | |
| Snowflakes | Snowflakes | |
| Shimmer | Shimmer | |
| Strobe | Strobe | |
| Bars | Bars | |
| Meteors | Meteors | |
| Spirals | Spirals | |
| Butterfly | Butterfly | |
| Plasma | Plasma | |
| Fireworks | Fireworks | |
| Marquee | Marquee | |

Because Chase and Rainbow share one xLights effect, as do On and Pulse, PixelConductor
tells them apart by their settings when it imports a sequence.

The stage and palette previews are PixelConductor's own approximations of each effect.
The rendered look in xLights is what plays on your lights.

## Layout support

The stage draws each model's real geometry for these xLights model types:

- Arches
- Single Line
- Icicles, including drop submodels
- Custom models

Any other model type shows as a single dot on the stage. You can still sequence it
normally.

## Extend: repeat a sequence across the song

Most songs repeat. **Extend** takes the part you've sequenced by hand and copies it onto
the later sections that match it musically.

1. It analyzes the song's repeat structure and proposes a plan. Each section in the plan
   has a mode and a confidence score:
   - `copy`: copy an earlier range, shifted by some number of beats
   - `tile`: repeat a short block across a range
   - `extend`: spread one model's effects to every model of the same kind
   - `dark`: leave the section empty
   - `ending`: generate hits from the song's accents
2. You adjust the plan: move the "sequenced up to here" cutoff, change modes and source
   ranges, and limit sections to certain models.
3. **Apply** writes a new numbered version next to the original, for example
   `back-yard-master.0.1.xsq`. Your original file is never touched, and existing effects
   are never modified. Copies are beat-relative, so they stay in time through tempo drift.
4. **Promote** makes a version the new master. The old master is backed up first and
   superseded versions move to `.pc_versions/`. Nothing is deleted.

The engine also runs from the command line:

```bash
python xsq_extend.py song.mp3 sequence.xsq
```

```bash
python xsq_extend.py song.mp3 sequence.xsq --plan plan.json --out sequence-extended.xsq
```

The first command prints a proposed plan as JSON. The second applies a plan and writes
the result. `plan-wizards-auto-proposed.json` and `plan-wizards-in-winter.json` show an
automatic plan next to a hand-edited one.

## Saving and safety

- **Atomic writes.** A save never leaves a half-written `.xsq`.
- **Backups.** Before each overwrite, the previous file is copied to `.pc_backups/` in
  the show folder. PixelConductor keeps the 20 most recent backups, plus one per hour for
  7 days and one per day for 90 days.
- **Auto-save.** Tick **Auto** to save every 30 seconds.
- **Drafts.** Unsaved work is mirrored to `drafts/` on the server. The next time you open
  that sequence, PixelConductor offers the draft instead. Discarded drafts move to
  `drafts/discarded/`.

## Keyboard shortcuts

| Keys | Action |
|---|---|
| Space | Play or pause |
| Home / End | Jump to the start or end |
| Ctrl+← / Ctrl+→ | Previous or next beat |
| `[` / `]` | Previous or next section |
| Ctrl+Z | Undo |
| Ctrl+Shift+Z or Ctrl+Y | Redo |
| Ctrl+C / Ctrl+V | Copy or paste effects |
| Delete or Backspace | Delete selected effects |
| Esc | Clear selection |
| Ctrl+S | Save |
| Ctrl+0 | Reset zoom |
| F | Full screen |

## API keys

**Auto-arrange** needs an Anthropic API key, which you enter under **API Keys**. The key
is stored only in your browser's local storage and is sent directly from the browser to
`api.anthropic.com`. It never touches the PixelConductor server or any file. The Moises
key field is not used yet.

## Files and caches

| Path | What it is |
|---|---|
| `app.py` | Flask server: file I/O, layout parsing, audio analysis, Extend endpoints |
| `PixelConductor.html` | The entire web app |
| `xsq_extend.py` | Extend engine: analyze, propose plan, apply plan |
| `launch.pyw` | Background launcher for Windows |
| `HANDOFF-xsq-extend.md` | Design notes for the Extend feature |
| `<song>.analysis.json` | Cached analysis, saved next to the audio file |
| `.stems/<song>/` | Cached stem audio, in the audio folder |
| `<song>.extend.npz` | Cached Extend analysis, next to the audio file |
| `<sequence>.extend-plan.json` | Your last Extend plan for that sequence |
| `drafts/` | Server-side drafts (not committed) |
| `logs/` | Launcher logs (not committed) |

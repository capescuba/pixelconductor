"""
xsq_extend — extend a partially sequenced xLights .xsq by repeating the
hand-sequenced material wherever the song repeats.

Pipeline (each stage is independently callable from app.py):

    a = analyze(mp3_path)                      # beats + per-beat features (cacheable)
    plan = propose_plan(a, xsq_path)           # list of sections: what to copy where, with confidence
    ... user edits plan in the UI (JSON) ...
    apply_plan(xsq_path, a, plan, out_path)    # writes the extended .xsq, never touching existing effects

Plan section modes:
    copy      – copy ref [src_s, src_e) shifted by lag_beats onto [s, e)
    tile      – tile a reference block (cells of `cell_beats`) across [s, e)
    extend    – replicate the effects one model already has in [s, e) onto all models of that class
    dark      – leave empty (recorded so the UI can show it)
    ending    – generated hit/hold/stab pattern from onset accents
All times in the plan are seconds; the engine works in beats internally.
"""
import json, copy
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np

# ----------------------------------------------------------------------------- analysis
def analyze(mp3_path, sr=22050, hop=512, beats=None):
    """Beat grid + per-beat features for the repeat matcher.

    `beats`: optional pre-computed beat times in seconds. Pass PixelConductor's cached
    /api/analyze grid here so the plan lines up with the beat lines the editor already
    draws; leave it None to let librosa track beats (what the prototype was verified on).
    """
    import librosa
    y, _ = librosa.load(mp3_path, sr=sr, mono=True)
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    if beats is None:
        tempo, bframes = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=hop, units='frames')
        bt = librosa.frames_to_time(bframes, sr=sr, hop_length=hop)
        tempo = float(np.atleast_1d(tempo)[0])
    else:
        bt = np.asarray(sorted(float(t) for t in beats), dtype=float)
        bt = bt[(bt >= 0) & (bt <= len(y) / sr)]
        if len(bt) < 4:
            raise ValueError(f'supplied beat grid has only {len(bt)} usable beats')
        tempo = float(60.0 / np.median(np.diff(bt)))
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=13)[1:]
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    # A supplied grid is not derived from these frames, so it can run past the last one
    # and can put two beats in the same frame; clamp so every per-beat slice is non-empty.
    nfr = min(oenv.shape[-1], chroma.shape[-1], mfcc.shape[-1], len(rms))
    bf = np.clip(librosa.time_to_frames(bt, sr=sr, hop_length=hop), 0, nfr - 1)
    feat = []
    for i in range(len(bt) - 1):
        a = int(bf[i]); b = max(int(bf[i + 1]), a + 1)
        c = np.median(chroma[:, a:b], axis=1); c /= (np.linalg.norm(c) + 1e-9)
        m = mfcc[:, a:b].mean(1)
        sub = np.array_split(oenv[a:b], 4); op = np.array([s.mean() if len(s) else 0.0 for s in sub]); op /= (op.sum() + 1e-9)
        feat.append(np.concatenate([c, m, op, [rms[a:b].mean()]]))
    feat = np.array(feat)
    feat[:, 12:24] = (feat[:, 12:24] - feat[:, 12:24].mean(0)) / (feat[:, 12:24].std(0) + 1e-9)
    feat[:, -1] = (feat[:, -1] - feat[:, -1].mean()) / (feat[:, -1].std() + 1e-9)
    W = np.concatenate([np.full(12, 1.5), np.full(12, 0.35), np.full(4, 1.0), [0.6]])
    F = feat * W; Fn = F / np.linalg.norm(F, axis=1, keepdims=True)
    D = 1 - Fn @ Fn.T
    # downbeat phase: which of the 4 beat positions carries the most onset energy
    beat_os = np.array([oenv[int(bf[i]):int(bf[i]) + 3].max() for i in range(len(bt) - 1)])
    phase = int(np.argmax([beat_os[p::4].mean() for p in range(4)]))
    # accents per beat (for the generated ending)
    span = lambda i: (max(int(bf[i]) - 2, 0), max(int(bf[i + 1]), int(bf[i]) + 1))
    accent = np.array([oenv[slice(*span(i))].max() for i in range(len(bt) - 1)])
    rms_beat = np.array([rms[int(bf[i]):max(int(bf[i + 1]), int(bf[i]) + 1)].mean() for i in range(len(bt) - 1)])
    return dict(tempo=tempo, beats=bt, D=D, phase=phase, beat_source='supplied' if beats is not None else 'librosa',
                accent=accent, rms_beat=rms_beat, duration=len(y) / sr)


class BeatClock:
    """seconds <-> fractional beat index, linear extrapolation at both ends."""
    def __init__(self, bt): self.bt = np.asarray(bt); self.idx = np.arange(len(bt))
    def tb(self, t):
        bt = self.bt
        if t <= bt[0]: return (t - bt[0]) / (bt[1] - bt[0])
        if t >= bt[-1]: return len(bt) - 1 + (t - bt[-1]) / (bt[-1] - bt[-2])
        return float(np.interp(t, bt, self.idx))
    def tt(self, b):
        bt = self.bt
        if b <= 0: return float(bt[0] + b * (bt[1] - bt[0]))
        if b >= len(bt) - 1: return float(bt[-1] + (b - (len(bt) - 1)) * (bt[-1] - bt[-2]))
        return float(np.interp(b, self.idx, bt))
    def shift_ms(self, ms, lag_beats): return int(round(self.tt(self.tb(ms / 1000) + lag_beats) * 1000))


# ----------------------------------------------------------------------------- xsq reading
def read_xsq(xsq_path):
    tree = ET.parse(xsq_path); root = tree.getroot()
    effects = {}  # (model, submodel|None) -> [(start_ms, end_ms, attrs)]
    for el in root.find('ElementEffects'):
        if el.get('type') != 'model': continue
        for layer in el.findall('EffectLayer'):
            effects[(el.get('name'), None)] = [(int(x.get('startTime')), int(x.get('endTime')), dict(x.attrib)) for x in layer]
        for sl in el.findall('SubModelEffectLayer'):
            effects[(el.get('name'), sl.get('name'))] = [(int(x.get('startTime')), int(x.get('endTime')), dict(x.attrib)) for x in sl]
    return tree, effects


def sequenced_cutoff(effects, beats, min_models=3, window_beats=8):
    """Last beat where at least `min_models` distinct models have an effect starting within the previous
    `window_beats` beats. Returns seconds. This is the 'where I got to' heuristic; the UI should show it and allow override."""
    clock = BeatClock(beats)
    starts_by_beat = defaultdict(set)
    for (model, _), effs in effects.items():
        for s, e, a in effs: starts_by_beat[int(clock.tb(s / 1000))].add(model)
    last = 0
    for b in range(len(beats)):
        models = set().union(*[starts_by_beat[k] for k in range(b - window_beats, b + 1)])
        if len(models) >= min_models: last = b
    return clock.tt(last + 1)


# ----------------------------------------------------------------------------- plan proposal
def propose_plan(a, xsq_path, cutoff_s=None, none_cost=0.25, switch=0.25, min_run_bars=2):
    bt = a['beats']; D = a['D']; phase = a['phase']; clock = BeatClock(bt)
    _, effects = read_xsq(xsq_path)
    cutoff_s = cutoff_s or sequenced_cutoff(effects, bt)
    nb = len(bt) - 1
    bars = [(i, min(i + 4, nb)) for i in range(phase, nb, 4)]
    barstart = np.array([bt[s] for s, e in bars]); nbar = len(bars)
    def bar_d(k, r): return float(np.mean([D[i, i - (bars[k][0] - bars[r][0])] for i in range(*bars[k])]))
    tgt = [k for k in range(nbar) if barstart[k] >= cutoff_s]
    refs = [k for k in range(nbar) if barstart[k] < cutoff_s]
    if not tgt:
        return dict(cutoff_s=round(cutoff_s, 3), tempo=round(a['tempo'], 2), sections=[],
                    note='nothing after the cutoff - this sequence is already fully sequenced')
    lags = list(range(1, nbar)); INF = 1e9
    def local(k, li):
        if li == len(lags): return none_cost
        r = k - lags[li]
        if r < 0 or barstart[r] >= cutoff_s: return INF
        return bar_d(k, r)
    T = len(tgt); cost = np.full((T, len(lags) + 1), INF); back = np.zeros((T, len(lags) + 1), int)
    prev = np.array([local(tgt[0], li) for li in range(len(lags) + 1)]); cost[0] = prev
    for ti in range(1, T):
        best = prev.min(); bi = int(prev.argmin())
        for li in range(len(lags) + 1):
            lc = local(tgt[ti], li)
            if lc >= INF: continue
            if prev[li] <= best + switch: cost[ti, li] = prev[li] + lc; back[ti, li] = li
            else: cost[ti, li] = best + switch + lc; back[ti, li] = bi
        prev = cost[ti]
    path = [int(prev.argmin())]
    for ti in range(T - 1, 0, -1): path.append(back[ti, path[-1]])
    path = path[::-1]
    runs = []
    for ti, li in enumerate(path):
        k = tgt[ti]; lag = None if li == len(lags) else lags[li] * 4
        d = None if lag is None else bar_d(k, k - lags[li])
        s, e = bt[bars[k][0]], bt[bars[k][1]]
        if runs and runs[-1]['lag_beats'] == lag: runs[-1]['e'] = float(e); runs[-1]['_ds'].append(d)
        else: runs.append(dict(s=float(s), e=float(e), lag_beats=lag, _ds=[d]))
    plan = []
    for r in runs:
        ds = [d for d in r['_ds'] if d is not None]
        sec = dict(s=round(r['s'], 3), e=round(r['e'], 3))
        if r['lag_beats'] is None:
            sec.update(mode='dark', note='no confident repeat found - review')
        else:
            src_s = clock.tt(clock.tb(r['s']) - r['lag_beats']); src_e = clock.tt(clock.tb(r['e']) - r['lag_beats'])
            sec.update(mode='copy', src_s=round(src_s, 3), src_e=round(src_e, 3), lag_beats=int(r['lag_beats']),
                       confidence=round(1 - float(np.mean(ds)), 3))
        plan.append(sec)
    # tail after the last beat: mark as ending candidate
    if plan and plan[-1]['e'] < a['duration'] - 1.0:
        plan.append(dict(s=plan[-1]['e'], e=round(a['duration'], 3), mode='ending', note='generated from accents - review'))
    return dict(cutoff_s=round(cutoff_s, 3), tempo=round(a['tempo'], 2), sections=plan)


# ----------------------------------------------------------------------------- plan application
class Writer:
    def __init__(self, tree, effects, clock, groups=()):
        """`groups`: names of xLights model GROUPS. PixelConductor writes groups as
        <Element type="model">, so without this list a group lands in a model class and
        `extend` mode fans effects out onto it."""
        self.tree = tree; self.root = tree.getroot(); self.clock = clock
        self.orig = effects
        self.occ = {k: [(s, e) for s, e, _ in v] for k, v in effects.items()}
        self.added = defaultdict(list); self.stats = defaultdict(lambda: [0, 0])
        self.manifest = []   # [model, submodel|None, startTime_ms, endTime_ms, tag] per added effect
        groups = set(groups)
        self.elements = {el.get('name'): el for el in self.root.find('ElementEffects')
                         if el.get('type') == 'model'}
        # Groups stay in `elements` (a group the user sequenced directly still gets copied),
        # but never in `classes` -- that is what `extend` mode fans out over, and group names
        # collide with model classes ("Fence 1" the group vs "Fence 5" the model).
        self.classes = defaultdict(list)  # model class ("Snow Flake") -> models
        for m in self.elements:
            if m not in groups: self.classes[m.rstrip('0123456789 ')].append(m)

    def overlaps(self, k, s, e): return any(s < b and e > a for a, b in self.occ.get(k, []))

    def add(self, model, sub, s, e, attrs, tag):
        if e - s < 40 or model not in self.elements: return
        k = (model, sub)
        if self.overlaps(k, s, e): self.stats[tag][1] += 1; return
        a = dict(attrs); a['startTime'] = str(s); a['endTime'] = str(e); a.pop('id', None); a.pop('protected', None)
        self.added[k].append(a); self.occ.setdefault(k, []).append((s, e)); self.stats[tag][0] += 1
        self.manifest.append([model, sub, s, e, tag])

    def copy_range(self, src_s, src_e, lag_beats, clip_s=None, clip_e=None, tag='copy', models=None, exclude=()):
        for (model, sub), effs in self.orig.items():
            if models and model not in models: continue
            if model in exclude: continue
            for s, e, attrs in effs:
                if not (src_s * 1000 <= s < src_e * 1000): continue
                ns, ne = self.clock.shift_ms(s, lag_beats), self.clock.shift_ms(e, lag_beats)
                if clip_s is not None: ns = max(ns, clip_s)
                if clip_e is not None: ne = min(ne, clip_e)
                self.add(model, sub, ns, ne, attrs, tag)

    def layer_elem(self, model, sub):
        el = self.elements[model]
        if sub is None: return el.find('EffectLayer')
        for sl in el.findall('SubModelEffectLayer'):
            if sl.get('name') == sub: return sl
        return ET.SubElement(el, 'SubModelEffectLayer', {'name': sub, 'layer': '0'})

    def flush(self):
        for (model, sub), effs in self.added.items():
            layer = self.layer_elem(model, sub)
            for a in effs: ET.SubElement(layer, 'Effect', a)
        for el in self.elements.values():
            for layer in el.findall('EffectLayer') + el.findall('SubModelEffectLayer'):
                effs = sorted(list(layer), key=lambda x: int(x.get('startTime')))
                for x in list(layer): layer.remove(x)
                for i, x in enumerate(effs):
                    a = dict(x.attrib); a.pop('id', None); x.attrib.clear()
                    for key in ('ref', 'name'):
                        if key in a: x.set(key, a.pop(key))
                    if i > 0: x.set('id', str(i))
                    for key, v in a.items(): x.set(key, v)
                    layer.append(x)

    def add_timing(self, name, marks):
        de = self.root.find('DisplayElements'); ee = self.root.find('ElementEffects')
        # Re-applying to an already-extended file must refresh these tracks, not stack
        # a second copy of them.
        for parent in (de, ee):
            for el in [x for x in parent if x.get('type') == 'timing' and x.get('name') == name]:
                parent.remove(el)
        ET.SubElement(de, 'Element', {'collapsed': 'false', 'type': 'timing', 'name': name, 'visible': 'true', 'active': 'true'})
        layer = ET.SubElement(ET.SubElement(ee, 'Element', {'type': 'timing', 'name': name}), 'EffectLayer')
        for s, e, label in marks: ET.SubElement(layer, 'Effect', {'label': label, 'startTime': str(s), 'endTime': str(e)})


def apply_plan(xsq_path, a, plan, out_path=None, timing_tracks=True, groups=(), dry_run=False):
    """Returns {stats, added, xml}. `added` is the provenance manifest
    ([model, submodel|None, startTime_ms, endTime_ms, tag] per generated effect). `xml` is the new
    document as a string (None when dry_run) so the caller can write it atomically;
    out_path is still honoured for the CLI. dry_run skips the write-out entirely, which
    is what the review screen's coverage preview uses.
    """
    bt = a['beats']; clock = BeatClock(bt); ms = lambda t: int(round(t * 1000))
    tree, effects = read_xsq(xsq_path); w = Writer(tree, effects, clock, groups)
    sec_marks = []
    for sec in plan['sections']:
        s, e, mode = sec['s'], sec['e'], sec['mode']
        if mode == 'copy':
            w.copy_range(sec['src_s'], sec['src_e'], sec['lag_beats'], clip_s=ms(s), clip_e=ms(e),
                         tag=f"copy {sec['src_s']:.1f}-{sec['src_e']:.1f}", models=sec.get('models'), exclude=sec.get('exclude', ()))
            label = f"copy of {sec['src_s']:.1f}-{sec['src_e']:.1f} (conf {sec.get('confidence', '?')})"
        elif mode == 'tile':
            cell = sec['cell_beats']; blk_s, blk_e = sec['src_s'], sec['src_e']
            ncell = max(1, int(round((clock.tb(blk_e) - clock.tb(blk_s)) / cell)))
            b = clock.tb(s); k = 0
            while b < clock.tb(e):
                ci = k % ncell
                cs = clock.tt(clock.tb(blk_s) + ci * cell); ce = clock.tt(clock.tb(blk_s) + (ci + 1) * cell)
                w.copy_range(cs - 0.06, ce, b - clock.tb(cs), clip_e=ms(e), tag='tile', models=sec.get('models'), exclude=sec.get('exclude', ()))
                b += cell; k += 1
            label = f"tiled {blk_s:.1f}-{blk_e:.1f}"
        elif mode == 'extend':
            src = sec['model']; cls = src.rstrip('0123456789 ')
            for ms_, me_, attrs in effects[(src, None)]:
                if ms_ >= ms(s) and me_ <= ms(e) + 1:
                    for m in w.classes[cls]:
                        if m != src: w.add(m, None, ms_, me_, attrs, 'extend')
            label = f"{src} effects extended to all {cls}s"
        elif mode == 'ending':
            _generated_ending(w, a, s, e, sec)
            label = 'ending: generated'
        else:
            label = sec.get('note', 'dark')
        sec_marks.append((ms(s), ms(e), label))
    stats = {k: dict(added=v[0], skipped=v[1]) for k, v in w.stats.items()}
    if dry_run:
        return dict(stats=stats, added=w.manifest, xml=None)
    w.flush()
    if timing_tracks:
        w.add_timing('Beats (auto)', [(ms(bt[i]), ms(bt[i + 1]), str((i - a['phase']) % 4 + 1)) for i in range(len(bt) - 1)])
        # plan sections may overlap (several operations on one range); timing marks may not
        edges = sorted({0, ms(a['duration'])} | {x for s, e, _ in sec_marks for x in (s, e)})
        marks = []
        for s, e in zip(edges[:-1], edges[1:]):
            labels = [l for ss, ee, l in sec_marks if ss <= s and ee >= e and l]
            marks.append((s, e, ' | '.join(dict.fromkeys(labels))))
        w.add_timing('Sections (auto)', marks)
    ET.indent(tree, space='  ')
    xml = '<?xml version="1.0" encoding="UTF-8"?>' + chr(10) + ET.tostring(tree.getroot(), encoding='unicode')
    if out_path:
        with open(out_path, 'w', encoding='utf-8') as f: f.write(xml)
    return dict(stats=stats, added=w.manifest, xml=xml)


def _generated_ending(w, a, s, e, sec):
    """White hits on accented beats, a hold once accents stop, full stab on the last accent.

    Which models play which part comes from sec['roles'] = {hit, hold, wash: [model names]}:
      hit  - short flashes only, plus a pulse on their `pulse_sub` submodel through the tail
      hold - flash, then sustain through the quiet tail
      wash - same as hold on a dimmer palette
    Omitted roles fall back to the back-yard show's classes, and a layout with none of
    those just sustains everything.
    """
    bt = a['beats']; ms = lambda t: int(round(t * 1000))
    on = {'ref': '0', 'name': 'On'}
    pal = sec.get('palettes', {'hit': '0', 'hold': '0', 'pulse': '2', 'dim': '4'})
    roles = sec.get('roles') or {}
    cls = w.classes
    def pick(key, default_class):
        if key in roles: return [m for m in roles[key] if m in w.elements]
        return list(cls.get(default_class, []))
    hit_models  = pick('hit',  'Snow Flake')
    hold_models = pick('hold', 'String')
    wash_models = pick('wash', 'Icicles')
    if not (hit_models or hold_models or wash_models):
        hold_models = [m for ml in cls.values() for m in ml]
    pulse_sub = sec.get('pulse_sub', 'Hub')
    i0, i1 = int(np.searchsorted(bt, s)), int(np.searchsorted(bt, e)) - 1
    acc = a['accent'][i0:i1]; thr = np.percentile(acc, 60) if len(acc) else 0
    hits = [bt[i] for i in range(i0, i1) if a['accent'][i] >= thr]
    if not hits: return
    last_hit = bt[i1] if i1 < len(bt) else e
    # find where accents stop: first run of 3 quiet beats after the last strong one
    quiet_from = None
    for i in range(i0, i1 - 2):
        if all(a['accent'][j] < thr for j in range(i, i + 3)): quiet_from = bt[i]; break
    for t in hits:
        if quiet_from and t >= quiet_from: continue
        for m in hit_models:  w.add(m, None, ms(t), ms(t) + 100, {**on, 'palette': pal['hit']}, 'ending hits')
        for m in hold_models: w.add(m, None, ms(t), ms(t) + 200, {**on, 'palette': pal['hit']}, 'ending hits')
        for m in wash_models: w.add(m, None, ms(t), ms(t) + 150, {**on, 'palette': pal['dim']}, 'ending hits')
    if quiet_from:
        for m in hold_models: w.add(m, None, ms(quiet_from), ms(last_hit), {**on, 'palette': pal['hold']}, 'ending hold')
        for m in wash_models: w.add(m, None, ms(quiet_from), ms(last_hit), {**on, 'palette': pal['dim']}, 'ending hold')
        for i in range(int(np.searchsorted(bt, quiet_from)), i1):
            for m in hit_models: w.add(m, pulse_sub, ms(bt[i]), ms(bt[i]) + 100, {**on, 'palette': pal['pulse']}, 'ending pulse')
    for m in hold_models + hit_models + wash_models:
        w.add(m, None, ms(last_hit), ms(last_hit) + 480, {**on, 'palette': pal['hit']}, 'ending stab')


# ----------------------------------------------------------------------------- CLI
if __name__ == '__main__':
    import argparse, pickle
    p = argparse.ArgumentParser()
    p.add_argument('mp3'); p.add_argument('xsq'); p.add_argument('--plan'); p.add_argument('--out'); p.add_argument('--cutoff', type=float)
    args = p.parse_args()
    a = analyze(args.mp3)
    if args.plan and args.out:
        plan = json.load(open(args.plan))
        r = apply_plan(args.xsq, a, plan, args.out)
        print(json.dumps(r['stats'], indent=1))
        print(f"  {len(r['added'])} effects added")
    else:
        print(json.dumps(propose_plan(a, args.xsq, args.cutoff), indent=1))

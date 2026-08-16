"""HTML for the diagnostic page.

Kept apart from `visualise.py` so the markup can be iterated on without
re-rendering galleries, which cost a couple of minutes of GL time.

Design notes, since they are decisions and not defaults:

*Ground.* Near-black with a blue bias (#0B0F14) rather than neutral grey,
because the page's primary content is greyscale sensor frames on black with
cyan/orange vector ink. The chrome extends the same instrument language instead
of framing it in something unrelated.

*Type.* Monospace for every number, axis label and eyebrow -- that is the
subject's own vernacular, since all of this came out of a terminal, and it gives
tabular alignment for free. System sans for prose. No webfont: the CSP blocks
font CDNs and a silent fallback is worse than a deliberate system stack.

*Series colours* are the four validated by `dataviz/scripts/validate_palette.js`
against both surfaces -- CVD separation dE 21.8, normal-vision floor 24.8,
contrast >=3:1 everywhere. Red is reserved for failure rate, which is a status,
not a fourth series. Every series is also direct-labelled, so identity never
rests on colour alone.
"""

from __future__ import annotations

import json

# Validated on both surfaces. Order matters: this is the fixed assignment, and
# swapping two entries breaks the adjacent-pair separation the validator checked.
S1, S2, S3, CRIT = "#2A8FC4", "#CE7412", "#8465E4", "#D23D58"

CSS = """
*, *::before, *::after { box-sizing: border-box; }

:root {
  --ground: #F7F8FA;
  --panel: #FFFFFF;
  --panel-2: #EEF1F5;
  --line: #D6DCE4;
  --ink: #10161C;
  --ink-2: #47535F;
  --ink-3: #7A8896;
  --s1: __S1__; --s2: __S2__; --s3: __S3__; --crit: __CRIT__;
  --grid: #E3E8EE;
  --shadow: 0 1px 2px rgba(16,22,28,.06), 0 8px 24px rgba(16,22,28,.05);
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root {
    --ground: #0B0F14; --panel: #121821; --panel-2: #0F141B; --line: #232D3A;
    --ink: #E6EDF3; --ink-2: #9FB0C0; --ink-3: #6C7C8C; --grid: #1B242F;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 28px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"] {
  --ground: #0B0F14; --panel: #121821; --panel-2: #0F141B; --line: #232D3A;
  --ink: #E6EDF3; --ink-2: #9FB0C0; --ink-3: #6C7C8C; --grid: #1B242F;
  --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 28px rgba(0,0,0,.35);
}
:root[data-theme="light"] {
  --ground: #F7F8FA; --panel: #FFFFFF; --panel-2: #EEF1F5; --line: #D6DCE4;
  --ink: #10161C; --ink-2: #47535F; --ink-3: #7A8896; --grid: #E3E8EE;
  --shadow: 0 1px 2px rgba(16,22,28,.06), 0 8px 24px rgba(16,22,28,.05);
}

body {
  margin: 0; background: var(--ground); color: var(--ink);
  font-family: var(--sans); font-size: 15px; line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 40px 24px 96px; }

h1, h2, h3 { text-wrap: balance; margin: 0; font-weight: 600; letter-spacing: -0.015em; }
h1 { font-size: 30px; line-height: 1.2; }
h2 { font-size: 20px; margin-bottom: 6px; }
h3 { font-size: 15px; }
p { margin: 0 0 12px; max-width: 68ch; color: var(--ink-2); }
p strong, li strong { color: var(--ink); font-weight: 600; }
a { color: var(--s1); }

.eyebrow {
  font-family: var(--mono); font-size: 11px; letter-spacing: .13em;
  text-transform: uppercase; color: var(--ink-3); margin-bottom: 10px;
}
header { border-bottom: 1px solid var(--line); padding-bottom: 28px; margin-bottom: 36px; }
header .sub { font-size: 16px; color: var(--ink-2); max-width: 74ch; margin-top: 10px; }
.meta {
  font-family: var(--mono); font-size: 12px; color: var(--ink-3);
  margin-top: 16px; display: flex; flex-wrap: wrap; gap: 6px 20px;
}

section { margin-bottom: 52px; scroll-margin-top: 20px; }
.lede { margin-bottom: 20px; }

/* ---- stat strip ---- */
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(168px, 1fr)); gap: 12px; }
.stat {
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 16px 18px; box-shadow: var(--shadow);
}
.stat .k {
  font-family: var(--mono); font-size: 10.5px; letter-spacing: .1em;
  text-transform: uppercase; color: var(--ink-3);
}
.stat .v {
  font-family: var(--mono); font-size: 27px; font-weight: 600; margin-top: 8px;
  font-variant-numeric: tabular-nums; letter-spacing: -0.02em;
}
.stat .u { font-size: 14px; color: var(--ink-2); font-weight: 400; }
.stat .n { font-size: 12px; color: var(--ink-3); margin-top: 4px; font-family: var(--mono); }

/* ---- galleries ---- */
.strip { display: flex; gap: 12px; overflow-x: auto; padding-bottom: 8px; }
.strip figure { flex: 0 0 232px; margin: 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(228px, 1fr)); gap: 14px; }
figure { margin: 0; }
figure img {
  width: 100%; display: block; border-radius: 8px; border: 1px solid var(--line);
  background: #05080B;
}
figcaption { font-size: 12.5px; color: var(--ink-2); margin-top: 8px; }
figcaption b { color: var(--ink); font-weight: 600; display: block; font-size: 13px; }
figcaption .num {
  font-family: var(--mono); font-size: 11.5px; color: var(--ink-3);
  font-variant-numeric: tabular-nums; display: block; margin-top: 3px;
}
.card {
  background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
  padding: 18px; box-shadow: var(--shadow);
}
.card.bad { border-color: color-mix(in oklab, var(--crit) 45%, var(--line)); }
.ba { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.delta {
  font-family: var(--mono); font-size: 12px; color: var(--s1); margin-top: 10px;
  padding-top: 8px; border-top: 1px solid var(--line); font-variant-numeric: tabular-nums;
}
td.good { color: var(--s1); font-weight: 600; }
td.warn { color: var(--s2); font-weight: 600; }
td.bad  { color: var(--crit); font-weight: 600; }
.tag {
  display: inline-block; font-family: var(--mono); font-size: 10.5px;
  letter-spacing: .08em; text-transform: uppercase; padding: 2px 7px;
  border-radius: 4px; background: var(--panel-2); color: var(--ink-2);
  border: 1px solid var(--line);
}
.tag.crit { color: var(--crit); border-color: color-mix(in oklab, var(--crit) 40%, var(--line)); }

/* ---- charts ---- */
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(430px, 1fr)); gap: 18px; }
.chart { background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
         padding: 18px 18px 12px; box-shadow: var(--shadow); }
.chart h3 { margin-bottom: 2px; }
.chart .why { font-size: 13px; color: var(--ink-2); margin: 4px 0 12px; max-width: none; }
.chart svg { width: 100%; height: auto; display: block; overflow: visible; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; margin-top: 10px;
          font-family: var(--mono); font-size: 11.5px; color: var(--ink-2); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
.s1 { stroke: var(--s1); } .s2 { stroke: var(--s2); }
.s3 { stroke: var(--s3); } .scrit { stroke: var(--crit); }
.f1 { fill: var(--s1); } .f2 { fill: var(--s2); }
.f3 { fill: var(--s3); } .fcrit { fill: var(--crit); }
.gridline { stroke: var(--grid); stroke-width: 1; }
.axisline { stroke: var(--line); stroke-width: 1; }
.tick { fill: var(--ink-3); font-family: var(--mono); font-size: 10.5px; }
.axlabel { fill: var(--ink-2); font-family: var(--mono); font-size: 11px; }
.dlabel { font-family: var(--mono); font-size: 11px; font-weight: 600; }
.mark { stroke-width: 2; fill: none; stroke-linecap: round; stroke-linejoin: round; }
.dot { r: 3.6; stroke: var(--panel); stroke-width: 1.5; }
.zone { fill: var(--crit); opacity: .07; }
.zoneline { stroke: var(--crit); stroke-dasharray: 3 3; stroke-width: 1; opacity: .55; }
.note { fill: var(--ink-3); font-family: var(--mono); font-size: 10.5px; }
.hit { fill: transparent; cursor: crosshair; }
.cross { stroke: var(--ink-3); stroke-width: 1; stroke-dasharray: 3 3; opacity: 0; }

.tip {
  position: fixed; pointer-events: none; opacity: 0; transition: opacity .09s;
  background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  padding: 9px 11px; font-family: var(--mono); font-size: 11.5px; line-height: 1.55;
  box-shadow: var(--shadow); z-index: 50; color: var(--ink);
  font-variant-numeric: tabular-nums; max-width: 260px;
}
.tip b { color: var(--ink); } .tip .k { color: var(--ink-3); }

table { border-collapse: collapse; width: 100%; font-family: var(--mono); font-size: 12px;
        font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--line); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink-3); font-weight: 600; font-size: 10.5px; letter-spacing: .08em;
     text-transform: uppercase; }
.scroll { overflow-x: auto; }
details { margin-top: 14px; }
summary { cursor: pointer; font-family: var(--mono); font-size: 12px; color: var(--ink-2); }
summary:focus-visible, button:focus-visible, a:focus-visible {
  outline: 2px solid var(--s1); outline-offset: 2px; border-radius: 4px;
}
.callout {
  border-left: 3px solid var(--s2); background: var(--panel-2);
  padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 16px 0;
}
.callout p:last-child { margin-bottom: 0; }
footer { border-top: 1px solid var(--line); padding-top: 20px; color: var(--ink-3);
         font-size: 13px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
@media (max-width: 620px) { .wrap { padding: 24px 14px 64px; } h1 { font-size: 24px; } }
"""


JS = r"""
const $ = (s, r) => (r || document).querySelector(s);
const tip = document.createElement('div');
tip.className = 'tip'; document.body.appendChild(tip);
const SVG = 'http://www.w3.org/2000/svg';
const el = (n, a) => { const e = document.createElementNS(SVG, n);
  for (const k in (a || {})) e.setAttribute(k, a[k]); return e; };
const fmt = (v, d) => v == null ? '--' : (+v).toFixed(d == null ? 2 : d);

function showTip(ev, html) {
  tip.innerHTML = html; tip.style.opacity = 1;
  const r = tip.getBoundingClientRect();
  let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - 14;
  if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - 14;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
}
const hideTip = () => { tip.style.opacity = 0; };

// Nice axis ticks: 1/2/5 x 10^n, so labels land on numbers people read easily.
function ticks(lo, hi, n) {
  const span = hi - lo || 1, raw = span / n, p = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].find(m => m * p >= raw) * p;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

function chart(host, spec) {
  const W = 560, H = spec.height || 300, m = { t: 14, r: 18, b: 44, l: 58 };
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img',
                          'aria-label': spec.aria || spec.title || 'chart' });
  const pts = spec.series.flatMap(s => s.points).filter(p => p.y != null);
  if (!pts.length) { host.appendChild(svg); return; }

  const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
  let x0 = spec.x0 != null ? spec.x0 : Math.min(...xs);
  let x1 = spec.x1 != null ? spec.x1 : Math.max(...xs);
  let y0 = spec.y0 != null ? spec.y0 : 0;
  let y1 = spec.y1 != null ? spec.y1 : Math.max(...ys);
  if (x1 === x0) x1 = x0 + 1;
  y1 = y1 * 1.12 || 1;

  const X = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
  const Y = v => H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);

  (spec.zones || []).forEach(z => {
    svg.appendChild(el('rect', { class: 'zone', x: X(z.from), y: m.t,
      width: Math.max(0, X(z.to) - X(z.from)), height: H - m.t - m.b }));
    svg.appendChild(el('line', { class: 'zoneline', x1: X(z.from), x2: X(z.from),
      y1: m.t, y2: H - m.b }));
    if (z.label) { const t = el('text', { class: 'note', x: X(z.from) + 5, y: m.t + 12 });
      t.textContent = z.label; svg.appendChild(t); }
  });

  ticks(y0, y1, 5).forEach(v => {
    svg.appendChild(el('line', { class: 'gridline', x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }));
    const t = el('text', { class: 'tick', x: m.l - 8, y: Y(v) + 3.5, 'text-anchor': 'end' });
    t.textContent = spec.yfmt ? spec.yfmt(v) : fmt(v, v < 10 ? 1 : 0); svg.appendChild(t);
  });
  (spec.xticks || ticks(x0, x1, 6)).forEach(v => {
    const t = el('text', { class: 'tick', x: X(v), y: H - m.b + 16, 'text-anchor': 'middle' });
    t.textContent = spec.xfmt ? spec.xfmt(v) : String(v); svg.appendChild(t);
  });
  svg.appendChild(el('line', { class: 'axisline', x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));

  let lab = el('text', { class: 'axlabel', x: (m.l + W - m.r) / 2, y: H - 6, 'text-anchor': 'middle' });
  lab.textContent = spec.xlabel; svg.appendChild(lab);
  lab = el('text', { class: 'axlabel', x: 12, y: (m.t + H - m.b) / 2,
                     'text-anchor': 'middle', transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})` });
  lab.textContent = spec.ylabel; svg.appendChild(lab);

  (spec.bands || []).forEach(b => {
    const up = b.points.map(p => `${X(p.x)},${Y(p.hi)}`);
    const dn = b.points.slice().reverse().map(p => `${X(p.x)},${Y(p.lo)}`);
    svg.appendChild(el('polygon', { points: up.concat(dn).join(' '),
      fill: `var(--${b.color})`, opacity: .13 }));
  });

  spec.series.forEach(s => {
    const p = s.points.filter(q => q.y != null);
    if (!p.length) return;
    const d = p.map((q, i) => `${i ? 'L' : 'M'}${X(q.x)},${Y(q.y)}`).join(' ');
    svg.appendChild(el('path', { d, class: `mark ${s.cls}`,
      'stroke-dasharray': s.dash || 'none' }));
    p.forEach(q => svg.appendChild(el('circle', { class: `dot ${s.fill}`, cx: X(q.x), cy: Y(q.y) })));
    // Direct label at the last point, so identity is never colour-alone.
    const last = p[p.length - 1];
    const t = el('text', { class: `dlabel ${s.fill}`, x: X(last.x) + 7, y: Y(last.y) + 3.5 });
    t.textContent = s.name; svg.appendChild(t);
  });

  const cross = el('line', { class: 'cross', y1: m.t, y2: H - m.b });
  svg.appendChild(cross);
  const hit = el('rect', { class: 'hit', x: m.l, y: m.t,
    width: W - m.l - m.r, height: H - m.t - m.b });
  svg.appendChild(hit);

  const key = [...new Set(spec.series.flatMap(s => s.points.map(p => p.x)))].sort((a, b) => a - b);
  hit.addEventListener('pointermove', ev => {
    const bb = svg.getBoundingClientRect();
    const vx = x0 + ((ev.clientX - bb.left) / bb.width * W - m.l) / (W - m.l - m.r) * (x1 - x0);
    const near = key.reduce((a, b) => Math.abs(b - vx) < Math.abs(a - vx) ? b : a, key[0]);
    cross.setAttribute('x1', X(near)); cross.setAttribute('x2', X(near));
    cross.style.opacity = .8;
    let html = `<b>${spec.xlabel.split(' (')[0]} ${spec.xfmt ? spec.xfmt(near) : near}</b>`;
    spec.series.forEach(s => {
      const q = s.points.find(p => p.x === near);
      if (q && q.y != null) html += `<br><span class="k">${s.name}</span> ${fmt(q.y, s.dp)}${s.unit || ''}`;
    });
    if (spec.extra) html += spec.extra(near);
    showTip(ev, html);
  });
  hit.addEventListener('pointerleave', () => { cross.style.opacity = 0; hideTip(); });
  host.appendChild(svg);
}

function scatter(host, spec) {
  const W = 560, H = spec.height || 300, m = { t: 14, r: 18, b: 44, l: 58 };
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': spec.aria });
  const d = spec.points;
  const x0 = 0, x1 = Math.max(...d.map(p => p.x)) * 1.05;
  const y1 = spec.y1 || Math.min(Math.max(...d.map(p => p.y)), spec.clip || 1e9) * 1.05;
  const X = v => m.l + v / (x1 - x0) * (W - m.l - m.r);
  const Y = v => H - m.b - Math.min(v, y1) / y1 * (H - m.t - m.b);

  (spec.zones || []).forEach(z => {
    svg.appendChild(el('rect', { class: 'zone', x: X(z.from), y: m.t,
      width: Math.max(0, X(z.to) - X(z.from)), height: H - m.t - m.b }));
    if (z.label) { const t = el('text', { class: 'note', x: X(z.from) + 5, y: m.t + 12 });
      t.textContent = z.label; svg.appendChild(t); }
  });
  ticks(0, y1, 5).forEach(v => {
    svg.appendChild(el('line', { class: 'gridline', x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }));
    const t = el('text', { class: 'tick', x: m.l - 8, y: Y(v) + 3.5, 'text-anchor': 'end' });
    t.textContent = fmt(v, 1); svg.appendChild(t);
  });
  ticks(0, x1, 6).forEach(v => {
    const t = el('text', { class: 'tick', x: X(v), y: H - m.b + 16, 'text-anchor': 'middle' });
    t.textContent = String(v); svg.appendChild(t);
  });
  svg.appendChild(el('line', { class: 'axisline', x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
  let lab = el('text', { class: 'axlabel', x: (m.l + W - m.r) / 2, y: H - 6, 'text-anchor': 'middle' });
  lab.textContent = spec.xlabel; svg.appendChild(lab);
  lab = el('text', { class: 'axlabel', x: 12, y: (m.t + H - m.b) / 2, 'text-anchor': 'middle',
                     transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})` });
  lab.textContent = spec.ylabel; svg.appendChild(lab);

  d.forEach(p => {
    const c = el('circle', { cx: X(p.x), cy: Y(p.y), r: 2.6,
      fill: `var(--${p.y > y1 ? 'crit' : 's1'})`, opacity: .5 });
    c.addEventListener('pointerenter', ev => { c.setAttribute('r', 5); c.setAttribute('opacity', 1);
      showTip(ev, spec.tip(p)); });
    c.addEventListener('pointerleave', () => { c.setAttribute('r', 2.6);
      c.setAttribute('opacity', .5); hideTip(); });
    svg.appendChild(c);
  });
  host.appendChild(svg);
}
"""


def _css():
    """CSS with the validated series colours substituted in.

    Plain token replacement rather than %-formatting: the stylesheet is full of
    literal percent signs (widths, color-mix ratios) and escaping every one of
    them to satisfy the formatter is a standing invitation to miss one.
    """
    return (CSS.replace("__S1__", S1).replace("__S2__", S2)
            .replace("__S3__", S3).replace("__CRIT__", CRIT))


def _legend(items):
    return '<div class="legend">' + "".join(
        f'<span><i style="background:var(--{c})"></i>{n}</span>' for n, c in items
    ) + "</div>"


def _tile(t, extra_class=""):
    nums = []
    if t["detected"] and t["pos"] is not None:
        nums.append(f'|resid| {t["pos"]:.2f} mm (dz {t["dz"]:+.2f})')
        if t["normal"] is not None:
            nums.append(f'normal {t["normal"]:.2f}°')
        if t["rms"] is not None:
            nums.append(f'fit rms {t["rms"]:.2f} px')
    else:
        nums.append("no detection")
    return f"""<figure class="card {extra_class}">
  <img src="{t['img']}" alt="{t['title']}: overlay of fitted and true ellipse with orientation and residual vectors" loading="lazy">
  <figcaption><b>{t['title']}</b>{t['note']}
    <span class="num">{' · '.join(nums)}</span></figcaption>
</figure>"""


def _pair(w):
    """One before/after row: same pose, weighting off then on.

    Every number is guarded: a tile can legitimately have no detection, in which
    case `_tile_json` writes ``None`` rather than a float. Formatting ``None``
    with ``:.2f`` raises, and it would raise *during the page build*, after the
    galleries have already cost minutes of GL time.
    """
    def n(v, fmt="{:.2f}", suffix=""):
        return "&mdash;" if v is None else fmt.format(v) + suffix

    cells = "".join(
        f'<figure><img src="{t["img"]}" alt="{w["title"]}, {t["label"]}" loading="lazy">'
        f'<figcaption><b>{t["label"]}</b>'
        f'<span class="num">|resid| {n(t["pos"])} mm · normal {n(t["normal"])}°</span>'
        f'</figcaption></figure>' for t in w["pair"]
    )
    before, after = w["pair"][0], w["pair"][1]

    def ratio(k):
        a, b = before[k], after[k]
        return a / b if (a is not None and b) else None

    return f"""<div class="card">
  <h3>{w['title']}</h3>
  <p class="why">{w['note'] or '&nbsp;'}</p>
  <div class="ba">{cells}</div>
  <div class="delta">position ×{n(ratio('pos'), '{:.1f}')} ·
    orientation ×{n(ratio('normal'), '{:.1f}')}</div>
</div>"""


def _floor_section(f):
    """The Cramer-Rao floors, and how far above them the pipeline sits."""
    if not f:
        return ""

    b, br, law = f["boundary"], f["branch"], f["law"]

    def num(v, fmt="{:.4f}"):
        return "&mdash;" if v is None else fmt.format(v)

    kind_class = {"bound": "good", "prediction": "warn", "actual": "bad"}
    kind_note = {"bound": "bound", "prediction": "prediction", "actual": "measured"}
    rows = []
    for lv in f["levels"]:
        rows.append(
            f"<tr><td>{lv['label']}</td>"
            f"<td class='{kind_class[lv['kind']]}'>{kind_note[lv['kind']]}</td>"
            f"<td>{num(lv['pos'])}</td><td>{num(lv['depth'])}</td>"
            f"<td>{num(lv['lateral'])}</td><td>{num(lv['angle'], '{:.3f}')}</td></tr>"
        )

    g_rows = "".join(f"<td>{r['value']:.3f}</td>" for r in law["g"])
    g_head = "".join(f"<th>{r['tilt']}&deg;</th>" for r in law["g"])

    return f"""<section>
  <div class="eyebrow">the information floor &middot; what no estimator can beat</div>
  <h2>How much of the error is even removable?</h2>
  <p class="lede">A residual is uninterpretable without a floor to compare it against:
  a millimetre could mean the solver is sloppy, or it could mean the information is not in
  the image, and those point at opposite work. The Cram&eacute;r&ndash;Rao bound settles it.
  Derivations in <code>bounds.py</code>, 60+ Monte-Carlo checks in
  <code>test_bounds.py</code>, this comparison from <code>limits.py</code>
  ({f['n']} frames spanning tilt 0&ndash;70&deg; and both condition tiers, so the
  &ldquo;measured&rdquo; row below is harder than the headline band at the top of this
  page and the two are not directly comparable).</p>

  <div class="callout"><p><b>The solver is already at the bound.</b> End-to-end
  Monte&nbsp;Carlo &mdash; noisy boundary points through the shipped ellipse fit and the
  shipped back-projection &mdash; gives statistical efficiencies of <b>0.94&ndash;1.06</b> on
  every pose component, and 0.96&ndash;1.01 on the five ellipse parameters. The direct
  algebraic fit is famously biased toward small ellipses and the reflex is to replace it
  with an iterative maximum-likelihood fit; at this noise level there is nothing to
  recover. <b>Whatever explains the gap below, it is not the solver.</b></p></div>

  <div class="scroll"><table>
    <thead><tr><th>level</th><th>status</th><th>&#8214;position&#8214; mm</th>
    <th>depth mm</th><th>lateral mm</th><th>angle &deg;</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>
  <p class="why">The first three are <b>bounds</b> &mdash; nothing may beat them. The
  fourth is a <b>prediction</b>: what independent Gaussian boundary error of the measured
  size would produce. The estimator beats it, and that is itself the finding &mdash; real
  boundary errors are correlated, and their common mode is exactly what the radius
  calibration absorbs. Pose error is scored against the <em>oracle</em> ambiguity branch,
  because a local bound cannot describe a likelihood with two equal maxima.</p>

  <div class="charts">
    <div class="chart">
      <h3>Where the boundary error actually lives</h3>
      <p class="why">Signed distance from every hull point to the analytic rim ellipse,
      decomposed. The sensor is nowhere near the limiting factor.</p>
      <div class="scroll"><table>
        <thead><tr><th>quantity</th><th>px</th></tr></thead>
        <tbody>
        <tr><td>photon-limited edge localisation</td><td>{num(b['photon_sigma'])}</td></tr>
        <tr><td>pixel quantisation, 1/&radic;12</td><td>{num(b['quant_sigma'])}</td></tr>
        <tr><td>measured scatter (robust)</td><td><b>{num(b['robust'])}</b></td></tr>
        <tr><td>measured scatter (std)</td><td>{num(b['scatter'])}</td></tr>
        <tr><td>contamination, std/robust</td><td>{num(b['contamination'], '{:.2f}')}&times;</td></tr>
        <tr><td>residual radial bias</td><td>{num(b['residual_bias'])}</td></tr>
        <tr><td>correlation length along contour</td><td>{num(b['corr_len'], '{:.1f}')}</td></tr>
        <tr><td>hull points kept / independent</td>
            <td>{num(b['n_points'], '{:.0f}')} / {num(b['n_effective'], '{:.0f}')}</td></tr>
        </tbody>
      </table></div>
    </div>
    <div class="chart">
      <h3>The depth law, corrected</h3>
      <p class="why">Every earlier section here quotes <code>z/(2R)</code>. That is the
      <em>tilt-known</em> case. Estimating tilt from the same ellipse costs a further
      &radic;3, because the semi-axis must be disentangled from the minor axis and the
      rotation &mdash; the <code>(a,b)</code> Fisher block is
      <code>(N/8&sigma;&sup2;)[[3,1],[1,3]]</code>, giving
      <code>&sigma;<sub>a</sub> = &radic;3&thinsp;&sigma;/&radic;N</code>.</p>
      <div class="scroll"><table>
        <thead><tr><th>tilt</th>{g_head}</tr></thead>
        <tbody><tr><td><i>g</i>(&theta;)</td>{g_rows}</tr></tbody>
      </table></div>
      <p class="why">At 250&nbsp;mm and 30&deg; the depth penalty is
      <b>{law['at_250']:.1f}&times;</b> lateral, not the {law['naive_at_250']:.1f}&times;
      the naive form gives. Noise level, point count, focal length and resolution all
      cancel &mdash; verified invariant to 0.4% across 16 combinations, and to six
      significant figures between <i>N</i>&nbsp;=&nbsp;500 and 8000. A practical form is
      <code>&sigma;<sub>depth</sub> &asymp; (z/R)&thinsp;&sigma;<sub>lateral</sub></code>,
      range over <em>radius</em>, good to 15% at any tilt.</p>
    </div>
  </div>

  <div class="charts">
    <div class="chart">
      <h3>The 71&times;, factored</h3>
      <p class="why">The three bounds multiply out as 6.0 &times; 3.2 &times; 3.7 = 71. It is
      tempting to read that as three budgets to attack. Only one is fully open.</p>
      <div class="scroll"><table>
        <thead><tr><th>step</th><th>factor</th><th>recoverable?</th></tr></thead>
        <tbody>
        <tr><td>photon &rarr; pixel-quantised</td><td>6.0&times;</td>
            <td class="bad">No &mdash; and not for the obvious reason</td></tr>
        <tr><td>quantised &rarr; hulled</td><td>3.2&times;</td>
            <td class="good">Yes &mdash; untouched, this is the lever</td></tr>
        <tr><td>hulled &rarr; measured</td><td>3.7&times;</td>
            <td class="warn">Partly &mdash; see below</td></tr>
        </tbody>
      </table></div>
      <p class="why"><b>Why the first is not recoverable.</b> Sub-pixel refinement does beat
      quantisation &mdash; <code>subpixel_boundary</code> reaches 0.076&nbsp;px on a clean disc
      against an edge bound of 0.052&nbsp;px, a 3.8&times; gain over the 0.289&nbsp;px
      quantisation floor. It buys nothing here because the traced curve sits
      <b>1.087&nbsp;px</b> from the rim whether it is located to 0.289&nbsp;px or
      0.076&nbsp;px. That is the derived form of the earlier measured result: an 18&times;
      improvement in edge localisation bought 2% of the outcome.</p>
      <p class="why"><b>Why the second is.</b> The convex hull keeps
      <b>31 vertices of a ~354&nbsp;px perimeter</b>, and <code>1/&radic;N</code> is not
      negotiable but <i>N</i> is. The hull is there to enforce the outward-only property the
      one-sided loss depends on &mdash; on real captures 42&ndash;62% of dense contour points
      lie <em>inside</em> the fitted ellipse &mdash; so the version that survives is dense
      contour points <em>restricted to</em> the hull.</p>
      <p class="why"><b>Why the third is only partly open.</b> What remains is that the
      silhouette is not the rim: the rod and magnet project into the outline, and which
      non-circular shape is presented changes frame to frame. Mostly that needs a fiducial on
      the rim, illumination that lights only the rim, or a forward model that separates rim
      from mast. But the axial weighting already recovered some of it in software, by
      modelling <em>where</em> the contamination is rather than fitting harder to the
      contaminated boundary &mdash; which is why it works where trimming by residual
      did not.</p>
    </div>
    <div class="chart">
      <h3>Sub-pixel refinement, against its own bound</h3>
      <p class="why">Synthetic soft-edged disc of known radius, C=200, &sigma;<sub>n</sub>=5,
      PSF 1.2&nbsp;px. The refinement works; it is not what is limiting.</p>
      <div class="scroll"><table>
        <thead><tr><th></th><th>bias</th><th>scatter</th></tr></thead>
        <tbody>
        <tr><td>thresholded contour</td><td>&minus;0.461 px</td><td>0.269 px</td></tr>
        <tr><td>after refinement</td><td>&minus;0.038 px</td><td><b>0.076 px</b></td></tr>
        <tr><td>edge Cram&eacute;r&ndash;Rao bound</td><td>&mdash;</td><td>0.052 px</td></tr>
        <tr><td>measured on real renders</td><td>+0.857 px</td><td class="bad">1.087 px</td></tr>
        </tbody>
      </table></div>
      <p class="why">At most <b>1.46&times;</b> remains in edge localisation, ever, by any
      method &mdash; acting on a term that is already 14&times; below the one that matters.
      A <em>perfect</em> sensor moves the total by under 5%.</p>
    </div>
  </div>

  <div class="callout"><p><b>Two things no engineering removes.</b> Rolling the rim about
  its own axis maps the circle onto itself, so the image derivative with respect to roll is
  zero &mdash; measured at 1.6&times;10<sup>&minus;12</sup>&nbsp;px over 64 roll angles. And a
  quadric cone admits exactly two circular cross-sections, so two poses whose normals differ
  by 49&deg; reproject to ellipses that agree to
  4.2&times;10<sup>&minus;12</sup>&nbsp;px. The single-view likelihood has two exactly equal
  maxima; prior-free branch selection is wrong on
  <b>{(br['wrong_fraction'] or 0) * 100:.0f}%</b> of independent frames, which is the size
  of the problem the temporal prior and the second camera exist to solve &mdash; not a
  shipped failure rate.</p></div>
</section>"""


def _modes_table(m):
    """Per-sensor-mode table from the resolution sweep."""
    if not m:
        return ""

    def cell(v, unit):
        return "&mdash;" if v is None else f"{v:.3f}{unit}"

    rows = []
    for r in m["rows"]:
        grade = "good" if r["in_spec"] >= 100 else ("warn" if r["in_spec"] >= 97 else "bad")
        rows.append(
            f"<tr><td>{r['mode']}</td><td>{r['fps']}</td>"
            f"<td>{r['detected']:.1f}%</td><td>{r['n']}</td>"
            f"<td>{cell(r['angle_worst'], '&deg;')}</td>"
            f"<td>{cell(r['pos_worst'], ' mm')}</td>"
            f"<td class='{grade}'>{r['in_spec']:.1f}%</td></tr>"
        )

    caveat = (
        '<div class="callout"><p><b>Read the detection column, not the in-spec column.</b> '
        'The gate is doing its job &mdash; every frame it certifies is in specification '
        '&mdash; but it certifies only a few percent of frames, and that number is '
        'currently <em>understated</em>. The error model it uses was fitted through the '
        'estimator while a default-argument bug forced the axial-weighted fit, which is '
        'what now ships &mdash; so the gate and the fit do at least match. But every '
        'configuration tried certifies ~3% at 1280&times;800 where an earlier sweep '
        'certified 13% on 800 poses, and that gap is identical across the axial A/B, so it '
        'is drift elsewhere in the chain and is not diagnosed. Read this as a lower bound '
        'on achievable coverage.</p></div>')

    gated = "with" if m["gated"] else "without"
    return (
        '<section>\n'
        '  <div class="eyebrow">final evaluation &middot; per sensor mode</div>\n'
        '  <h2>Does it meet &plusmn;1&deg; and &plusmn;0.5 mm on every frame it reports?</h2>\n'
        f'  <p class="lede">Stereo-fused, {m["poses"]} poses, {gated} the certification '
        '  gate. The contract is not the median &mdash; it is the <em>worst</em> frame '
        '  among those the estimator agrees to report. Declining a frame it cannot '
        '  certify is correct behaviour, so detection rate sits beside every accuracy '
        '  figure; without it, &ldquo;100% in spec&rdquo; is trivially achievable by '
        '  reporting nothing.</p>\n'
        f'  {caveat}\n'
        '  <div class="scroll"><table>\n'
        '    <thead><tr><th>mode</th><th>fps</th><th>detected</th><th>n</th>'
        '<th>worst angle</th><th>worst position</th><th>in spec</th></tr></thead>\n'
        f'    <tbody>{"".join(rows)}</tbody>\n'
        '  </table></div>\n'
        '</section>'
    )


def render_page(p):
    h = p["headline"]
    g = p["galleries"]
    sens = p["sensitivity"]
    tilt = p["tilt"]

    stats = [
        ("lateral x,y", f'{h["lateral"]:.3f}', "mm", "0.05% of range"),
        ("depth z", f'{h["depth"]:.3f}', "mm", f'{h["depth_pct"]:.2f}% of range'),
        ("‖position‖", f'{h["pos"]:.3f}', "mm", f'{h["pos_pct"]:.2f}% of range'),
        ("orientation", f'{h["normal"]:.2f}', "°", "normal, best branch"),
        ("tilt θ", f'{h["tilt"]:.2f}', "°", "after calibration"),
        ("ambiguity", f'{h["ambiguity"]:.0f}', "°", "cost of a wrong branch"),
    ]
    stat_html = "".join(
        f'<div class="stat"><div class="k">{k}</div>'
        f'<div class="v">{v}<span class="u"> {u}</span></div>'
        f'<div class="n">{n}</div></div>' for k, v, u, n in stats
    )

    tbl_rows = "".join(
        f"<tr><td>{r['x']:.0f}</td><td>{r['median']:.3f}</td><td>{r['p95']:.3f}</td>"
        f"<td>{r['lateral']:.3f}</td><td>{r['depth']:.3f}</td>"
        f"<td>{r['normal']:.2f}</td><td>{r['n']}</td></tr>" for r in tilt["bins"]
    )

    return f"""<title>Pose estimator diagnostics — where the error comes from</title>
<style>{_css()}</style>
<div class="wrap">

<header>
  <div class="eyebrow">flying-robot vision · synthetic validation</div>
  <h1>Where the pose error actually comes from</h1>
  <p class="sub">A 5-DOF estimator recovers the robot's position and rotor axis by
  back-projecting the image ellipse of its 20.4&nbsp;mm duct ring. It is accurate to
  <b>{h['pos']:.2f}&nbsp;mm</b> — but that number says nothing about <em>which direction</em> the
  error points, <em>which part of the fit</em> produced it, or <em>how much of it is even
  removable</em>. These overlays, curves and Cram&eacute;r&ndash;Rao bounds answer all three.</p>
  <div class="meta">
    <span>{h['n_test']} held-out poses</span><span>rendered with sensor noise + motion blur</span>
    <span>radius {h['radius']} mm</span>
    <span>tilt calibration a={p['calibration']['a']:.4f} b={p['calibration']['b']:.5f}</span>
  </div>
</header>

<section>
  <div class="eyebrow">headline residuals · tilt 10–45°</div>
  <div class="stats">{stat_html}</div>
  <div class="callout"><p><b>Depth carries almost all of it.</b> Lateral position is
  <b>6×</b> better than depth, and that ratio is geometry, not tuning: lateral comes from the
  ellipse's <em>centre</em>, depth from its <em>size</em>. Monocular size-ranging uses the object
  as its own baseline, and this baseline is 20.4&nbsp;mm wide. The exact penalty is
  <code>g(θ)·z/(2R)</code> with <code>g(0) = √3</code> — range over diameter, times a factor
  between 1.73 and 2.18 that is the price of having to estimate the tilt from the same ellipse.
  Derived and verified below.</p></div>
</section>

<section>
  <div class="eyebrow">shipped · axial weighting</div>
  <h2>The rod was being fitted as if it were the rim</h2>
  <p class="lede">The mast and magnet stick out along the rotor axis, so as the robot
  tilts they push the silhouette outward in its <em>short</em> direction — and because
  they sit on the axis, they land near the <b>middle of the major axis</b>. Weighting
  each hull point by how far along that axis it lies, <code>w = |proj|/a</code>,
  suppresses them. Same pose in each pair, weighting off then on.</p>
  <div class="callout"><p><b>An earlier version of this page said the weighting was off
  because it collapses the certification gate from ~13% of frames to ~1–3.5%. That claim is
  withdrawn.</b> It was measured while <code>PoseEstimator.update</code> forced the weighted
  fit regardless of the flag, so both arms ran weighted; the two figures came from unrelated
  runs and a third from a different metric entirely. The pairs below were byte-identical for
  the same reason, and are now genuinely different.</p>
  <p>The controlled A/B — same seed, same 400 poses, same gate, same constants, the flag the
  only difference:</p>
  <div class="scroll"><table>
    <thead><tr><th>core mode</th><th>detect off</th><th>detect on</th>
    <th>pos avg off</th><th>pos avg on</th><th>ang avg off</th><th>ang avg on</th></tr></thead>
    <tbody>
    <tr><td>1280×800</td><td>3.2%</td><td>3.2%</td><td>0.268 mm</td>
        <td class="good">0.186 mm</td><td>0.294°</td><td class="good">0.178°</td></tr>
    <tr><td>800×600</td><td>2.2%</td><td>2.8%</td><td>0.331 mm</td>
        <td class="good">0.271 mm</td><td>0.281°</td><td class="good">0.211°</td></tr>
    <tr><td>640×480</td><td>2.0%</td><td>2.0%</td><td>0.338 mm</td>
        <td class="good">0.296 mm</td><td>0.371°</td><td class="good">0.244°</td></tr>
    </tbody>
  </table></div>
  <p>Certified frames over the six usable modes: <b>61 off, 59 on</b> — the same. Modes at
  100% in spec: <b>5/8 off, 6/8 on</b>. <b>The weighting costs no gate coverage and improves
  both metrics in every mode, so it is now on — the pairs below describe the shipped
  build.</b></p>
  <p>The matched-set objection that kept it off was also wrong. The claim was that the
  weighted fit needs <code>RADIUS_MM = 10.2662</code> against the unweighted 10.2446;
  refitting the radius from a dataset regenerated with the weighting on gives
  <b>10.2418</b> — 0.03% away, an order of magnitude inside the residual it was supposed to
  bias. The two fits do not need different radii.</p>
  <p>Regenerating the <em>whole</em> chain for the weighted fit was tried and
  <b>rejected on measurement</b>, which is worth recording because it is the opposite of
  what "make it self-consistent" predicts:</p>
  <div class="scroll"><table>
    <thead><tr><th>configuration</th><th>certified frames (6 usable core modes)</th>
    <th>modes at 100% in spec</th></tr></thead>
    <tbody>
    <tr><td>axial off, original constants</td><td>61</td><td>5/8</td></tr>
    <tr><td><b>axial on, original constants (shipped)</b></td>
        <td class="good"><b>59</b></td><td class="good"><b>6/8</b></td></tr>
    <tr><td>axial on, fully refit constants</td><td class="bad">12</td><td>7/8</td></tr>
    </tbody>
  </table></div>
  <p>The refit trades <b>59 certified frames for 12</b> to gain one mode, and the refitted
  error model looks unhealthy — its own held-out acceptance fell from 3.5% to 0.9%, and its
  fitted <code>log_inv_margin</code> coefficient went <em>negative</em> (−1.6, from +1.03).
  A sign flip on a feature meaning &ldquo;how far apart the two ambiguity branches
  were&rdquo; is not a physically sensible model. Refitting it properly is open work; that
  coefficient is the lead.</p></div>
  <div class="charts">{"".join(_pair(w) for w in g["weighting"])}</div>
</section>

<section>
  <div class="eyebrow">how to read an overlay</div>
  <h2>Each layer, added one at a time</h2>
  <p class="lede">Same frame throughout. The residual arrow is drawn at
  <b>×{p['gain']:.0f}</b> — at true scale it is under 6&nbsp;px and points mostly along the optical
  axis, so it would be invisible. The gain is written on every frame that uses it.</p>
  <div class="strip">{''.join(_tile(t) for t in g['layers'])}</div>
</section>

{_modes_table(p.get("modes"))}

{_floor_section(p.get("floor"))}

<section>
  <div class="eyebrow">failure modes</div>
  <h2>The frames that go wrong, and why</h2>
  <p class="lede">About 1% of frames under realistic noise fail badly. They are not scattered
  randomly — every one sits in a corner of the condition space that can be named, and you can see
  the cause: the fitted ellipse lands on the blade cross instead of the rim.</p>
  <div class="grid">{''.join(_tile(t, 'bad') for t in g['failures'])}</div>
</section>

<section>
  <div class="eyebrow">conditions</div>
  <h2>What the estimator sees as things get harder</h2>
  <p class="lede">Tilt, lighting, background and opacity. Opacity and background barely move the
  result until background reaches the threshold itself; tilt and ambient light are what matter.</p>
  <div class="grid">{''.join(_tile(t) for t in g['conditions'])}</div>
</section>

<section>
  <div class="eyebrow">measurements</div>
  <h2>Residual against every condition</h2>
  <div class="charts">
    <div class="chart">
      <h3>Residual vs tilt</h3>
      <p class="why">Two different failures bracket the usable band, for opposite reasons.</p>
      <div id="c-tilt"></div>
      {_legend([("median", "s1"), ("p95", "s2"), ("depth", "s3")])}
    </div>
    <div class="chart">
      <h3>Every sample, by tilt</h3>
      <p class="why">Hover any point. The spread below 10° is ill-conditioning, not noise.</p>
      <div id="c-scatter"></div>
    </div>
    <div class="chart">
      <h3>Read noise</h3>
      <p class="why">Median barely moves to σ=40. <b>No frame failed at any level.</b></p>
      <div id="c-noise"></div>
      {_legend([("median", "s1"), ("RMSE", "s2"), ("seg IoU ×10", "s3")])}
    </div>
    <div class="chart">
      <h3>Background grey level</h3>
      <p class="why">Flat, flat, flat — then a cliff exactly where grey meets the threshold.</p>
      <div id="c-bg"></div>
      {_legend([("median", "s1"), ("RMSE", "s2"), ("failure rate %", "crit")])}
    </div>
    <div class="chart">
      <h3>Depth amplification vs range</h3>
      <p class="why">Measured ratio against <code>g(&theta;)&middot;z/(2R)</code>, the tilt-unknown law. The dashed <code>z/(2R)</code> is the tilt-known form this chart used to compare against &mdash; low by &radic;3.</p>
      <div id="c-amp"></div>
      {_legend([("measured", "s1"), ("g(θ)·z/(2R)", "s2"), ("z/(2R), tilt known", "s3")])}
    </div>
    <div class="chart">
      <h3>Lateral and depth, separately</h3>
      <p class="why">Lateral is flat in millimetres; depth grows with range.</p>
      <div id="c-split"></div>
      {_legend([("lateral", "s1"), ("depth", "s3")])}
    </div>
  </div>

  <details>
    <summary>Table view — residual by tilt band</summary>
    <div class="scroll"><table>
      <thead><tr><th>tilt °</th><th>median mm</th><th>p95 mm</th><th>lateral mm</th>
      <th>depth mm</th><th>normal °</th><th>n</th></tr></thead>
      <tbody>{tbl_rows}</tbody>
    </table></div>
  </details>
</section>

<footer>
  <p>All figures are <b>synthetic</b>: the robot mesh rendered through the rig's measured
  intrinsics, so ground truth is exact. Numbers come from a held-out split the calibration never
  saw. Orientation is scored against the better of the two back-projection branches — a single
  view of a circle cannot tell them apart, which is what the ambiguity figure above costs when the
  choice goes wrong. Resolving it needs a second camera.</p>
</footer>
</div>

<script>
{JS}
const D = {json.dumps({'sens': sens, 'tilt': tilt, 'gain': p['gain']})};

chart($('#c-tilt'), {{
  xlabel: 'ground-truth tilt (deg)', ylabel: 'position residual (mm)',
  aria: 'Position residual against ground-truth tilt',
  zones: [{{from: 0, to: 10, label: 'ill-conditioned'}}, {{from: 45, to: 75, label: 'model breaks'}}],
  series: [
    {{name: 'median', cls: 's1', fill: 'f1', unit: ' mm',
      points: D.tilt.bins.map(b => ({{x: b.x, y: b.median}}))}},
    {{name: 'p95', cls: 's2', fill: 'f2', unit: ' mm', dash: '5 4',
      points: D.tilt.bins.map(b => ({{x: b.x, y: b.p95}}))}},
    {{name: 'depth', cls: 's3', fill: 'f3', unit: ' mm',
      points: D.tilt.bins.map(b => ({{x: b.x, y: b.depth}}))}}
  ],
  extra: x => {{ const b = D.tilt.bins.find(b => b.x === x);
    return b ? `<br><span class="k">n</span> ${{b.n}}` : ''; }}
}});

scatter($('#c-scatter'), {{
  xlabel: 'ground-truth tilt (deg)', ylabel: 'position residual (mm)',
  aria: 'Every held-out sample, residual against tilt', clip: 6,
  zones: [{{from: 0, to: 10, label: 'ill-conditioned'}}, {{from: 45, to: 75, label: 'model breaks'}}],
  points: D.tilt.scatter.map(s => ({{x: s.tilt, y: s.pos, d: s}})),
  tip: p => `<b>tilt ${{fmt(p.d.tilt, 1)}}°</b><br>`
    + `<span class="k">|resid|</span> ${{fmt(p.d.pos, 3)}} mm<br>`
    + `<span class="k">depth dz</span> ${{fmt(p.d.dz, 3)}} mm<br>`
    + `<span class="k">lateral</span> ${{fmt(p.d.lat, 3)}} mm<br>`
    + `<span class="k">range</span> ${{fmt(p.d.z, 0)}} mm<br>`
    + `<span class="k">normal</span> ${{fmt(p.d.normal, 2)}}°`
}});

chart($('#c-noise'), {{
  xlabel: 'read noise σ (grey levels)', ylabel: 'position residual (mm)',
  aria: 'Residual against read noise', x0: 0,
  series: [
    {{name: 'median', cls: 's1', fill: 'f1', unit: ' mm', dp: 3,
      points: D.sens.noise.map(r => ({{x: r.x, y: r.median}}))}},
    {{name: 'RMSE', cls: 's2', fill: 'f2', unit: ' mm', dp: 3,
      points: D.sens.noise.map(r => ({{x: r.x, y: r.rmse}}))}},
    {{name: 'IoU ×10', cls: 's3', fill: 'f3', dp: 2,
      points: D.sens.noise.map(r => ({{x: r.x, y: r.iou * 10}}))}}
  ],
  extra: x => {{ const r = D.sens.noise.find(r => r.x === x);
    return r ? `<br><span class="k">failures</span> ${{fmt(r.fail, 1)}}%<br>`
      + `<span class="k">normal</span> ${{fmt(r.normal, 2)}}°` : ''; }}
}});

chart($('#c-bg'), {{
  xlabel: 'background grey level (0–1)', ylabel: 'residual (mm) / failure %',
  aria: 'Residual against background grey level', x0: 0, y1: 100,
  xfmt: v => (+v).toFixed(2),
  zones: [{{from: 0.46, to: 0.52, label: 'grey = threshold'}}],
  series: [
    {{name: 'median', cls: 's1', fill: 'f1', unit: ' mm', dp: 2,
      points: D.sens.background.map(r => ({{x: r.x, y: Math.min(r.median, 100)}}))}},
    {{name: 'RMSE', cls: 's2', fill: 'f2', unit: ' mm', dp: 2, dash: '5 4',
      points: D.sens.background.map(r => ({{x: r.x, y: Math.min(r.rmse, 100)}}))}},
    {{name: 'failure %', cls: 'scrit', fill: 'fcrit', unit: '%', dp: 0,
      points: D.sens.background.map(r => ({{x: r.x, y: r.fail}}))}}
  ]
}});

chart($('#c-amp'), {{
  xlabel: 'range (mm)', ylabel: 'depth error ÷ lateral error',
  aria: 'Depth amplification against range', x0: 140,
  series: [
    {{name: 'measured', cls: 's1', fill: 'f1', unit: '×',
      points: D.tilt.bands.map(b => ({{x: b.x, y: b.measured}}))}},
    {{name: 'g(θ)·z/(2R)', cls: 's2', fill: 'f2', unit: '×', dash: '5 4',
      points: D.tilt.bands.map(b => ({{x: b.x, y: b.predicted}}))}},
    {{name: 'z/(2R), tilt known', cls: 's3', fill: 'f3', unit: '×', dash: '2 4',
      points: D.tilt.bands.map(b => ({{x: b.x, y: b.predicted_naive}}))}}
  ],
  extra: x => {{ const b = D.tilt.bands.find(b => b.x === x);
    return b ? `<br><span class="k">depth</span> ${{fmt(b.rel_depth, 3)}}% of range` : ''; }}
}});

chart($('#c-split'), {{
  xlabel: 'range (mm)', ylabel: 'median residual (mm)',
  aria: 'Lateral and depth residual against range', x0: 140,
  series: [
    {{name: 'lateral', cls: 's1', fill: 'f1', unit: ' mm', dp: 3,
      points: D.tilt.bands.map(b => ({{x: b.x, y: b.lateral}}))}},
    {{name: 'depth', cls: 's3', fill: 'f3', unit: ' mm', dp: 3,
      points: D.tilt.bands.map(b => ({{x: b.x, y: b.depth}}))}}
  ]
}});
</script>
"""

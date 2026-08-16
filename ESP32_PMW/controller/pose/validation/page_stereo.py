"""The stereo diagnostics page: markup only, no measurement.

Deliberately thin.  `page.py` already owns the visual language -- type scale,
colour tokens, card and figure styling, light/dark theming -- so this imports
`_css()` and the series colours rather than restating them, and adds only the
two layouts that are new: a **pair** (two camera frames side by side under one
caption) and a **scene** (a 3-D panel with its own caption).

Two differences from `page.py`, both deliberate:

* It emits a **complete document** -- doctype, charset, viewport.  `page.py`
  returns a fragment because its output is embedded; this page is opened
  directly from disk, and a missing charset turns every `±` and `°` in the
  numbers into mojibake.
* There is **no JavaScript**.  `page.py` needs it to draw charts; here every
  figure is already an image, so the page has nothing to run and nothing that
  can fail to run.

`page.py`'s `render_page` is one enormous f-string, which forces every literal
brace in its embedded JS to be doubled.  This file keeps its CSS in a plain
(non-f) string and its markup in small pieces, so no escaping is needed anywhere.
"""

from __future__ import annotations

import base64
import html
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from page import CRIT, S1, S2, S3, _css  # noqa: E402

# The 3-D panels' own colours, restated here so the page legend and the figures
# cannot drift apart. scene3d owns the authoritative values.
COL_TRUTH = "#000000"
COL_EST = "#C8930A"

EXTRA_CSS = """
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.pair figure { margin: 0; }
.pair img, .scene img {
  width: 100%; height: auto; display: block; border-radius: 8px;
  border: 1px solid var(--line);
}
/* The camera frames are greyscale-on-black; the 3-D panels are line art on
   white and carry their own background, so only the frames get the dark mat. */
.pair img { background: #05080B; }
.scene img { background: #FFFFFF; }
.scenes { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
@media (max-width: 860px) {
  .pair, .scenes { grid-template-columns: 1fr; }
}
.case { margin: 0 0 34px; }
.case > h3 { margin: 0 0 2px; font-size: 15px; }
.case > .sub { color: var(--ink-3); font-size: 12.5px; margin: 0 0 12px; }
.numbers {
  display: flex; flex-wrap: wrap; gap: 0 22px; margin: 10px 0 0;
  font-family: var(--mono); font-size: 12px; color: var(--ink-2);
}
.numbers b { color: var(--ink); font-weight: 600; }
.numbers .k { color: var(--ink-3); }
.numbers .over { color: var(--crit); }
.capt { color: var(--ink-3); font-size: 12px; margin: 6px 0 0; }
.swatch { display: inline-block; width: 11px; height: 11px; border-radius: 2px;
          vertical-align: -1px; margin-right: 5px; border: 1px solid var(--line); }
/* Named `caveat`, not `note`: page.py already owns `.note` as an SVG text
   class, and reusing the name for a block would be a trap for the next reader
   even though nothing in this page draws SVG. */
.caveat { border-left: 2px solid var(--line); padding: 2px 0 2px 12px;
          color: var(--ink-2); font-size: 13px; margin: 18px 0; }
/* page.py's h2 has margin: 0, which is right inside its dense card layout and
   wrong for a page of long sections. */
main.wrap h2 { margin-top: 34px; }
main.wrap p { margin: 10px 0; max-width: 74ch; }
table.rig { border-collapse: collapse; font-size: 12.5px; margin: 6px 0 0; }
table.rig td { padding: 2px 16px 2px 0; }
table.rig td:first-child { color: var(--ink-3); }
table.rig td:nth-child(2) { font-family: var(--mono); color: var(--ink); }
table.m { border-collapse: collapse; font-size: 12px; margin: 10px 0 4px; width: 100%; }
table.m th { text-align: right; font-weight: 600; color: var(--ink-3); font-size: 11px;
             padding: 3px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
table.m th:first-child, table.m td:first-child { text-align: left; }
table.m td { text-align: right; padding: 3px 10px; font-family: var(--mono);
             color: var(--ink); border-bottom: 1px solid var(--panel-2); white-space: nowrap; }
table.m td:first-child { font-family: var(--sans); color: var(--ink-2); }
table.m tr.best td { color: var(--ink); font-weight: 600; }
table.m tr.best td:first-child::after { content: "  \2190 shipped"; color: var(--ink-3);
                                        font-weight: 400; font-size: 10.5px; }
.finding { margin: 26px 0 0; }
.finding h3 { margin: 0 0 2px; font-size: 15px; }
.finding > p { margin: 4px 0 0; }
.primer { background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
          padding: 20px 24px; margin: 16px 0 8px; box-shadow: var(--shadow); }
.primer h3 { margin: 18px 0 4px; font-size: 14px; }
.primer h3:first-of-type { margin-top: 0; }
.primer p { margin: 6px 0; max-width: 70ch; }
.primer dl { margin: 10px 0 0; }
.primer dt { font-weight: 600; color: var(--ink); margin-top: 10px; font-size: 13px; }
.primer dd { margin: 2px 0 0; color: var(--ink-2); font-size: 13px; max-width: 68ch; }
.primer .term { font-style: normal; font-weight: 600; color: var(--ink); }
.ascii { font-family: var(--mono); font-size: 11.5px; line-height: 1.35; color: var(--ink-2);
         background: var(--panel-2); border: 1px solid var(--line); border-radius: 8px;
         padding: 12px 14px; margin: 12px 0; overflow-x: auto; white-space: pre; }
"""


PRIMER = """<section class="primer">
<h3>The problem</h3>
<p>A small flying robot hovers in front of two cameras. We want to know, several
hundred times a second, <b>where it is</b> (three numbers) and <b>which way it is
tilted</b> (two more). Nothing is attached to the robot &mdash; it weighs 21&nbsp;mg
and cannot carry a sensor &mdash; so everything has to come from the pictures.</p>

<h3>The one useful feature</h3>
<p>The robot has a circular <span class="term">rim</span> around its rotor, and we
know its real size: 20.4&nbsp;mm across. That single known dimension is what makes
the problem solvable, because a circle of <em>known</em> size photographed from an
unknown position tells you a lot.</p>
<p>Photograph a circle straight on and you get a circle. Photograph it from an
angle and you get an <b>ellipse</b> &mdash; a squashed circle. Three things about
that ellipse carry the information we want:</p>
<div class="ascii">  how BIG it is        &rarr;  how FAR AWAY the robot is
  how SQUASHED it is   &rarr;  how far it is TILTED
  which way it&rsquo;s squashed &rarr;  which DIRECTION it leans</div>

<h3>Terms used below</h3>
<dl>
<dt>Segmentation</dt>
<dd>Deciding which pixels are robot and which are background. Here: everything
brighter than a threshold, since the robot is lit against a dark backdrop.</dd>

<dt>Fitted ellipse</dt>
<dd>The ellipse that best matches the outline of those bright pixels. This is the
actual <em>measurement</em> &mdash; five numbers (centre x, centre y, long axis,
short axis, angle) standing in for the whole image. Everything downstream works
from these five numbers, which is why so much attention goes to how trustworthy
each of them is.</dd>

<dt>Ray</dt>
<dd>A straight line from the camera&rsquo;s lens out through one pixel and off into
the world. Everything lying along that line lands on the <em>same</em> pixel, so a
single camera can never tell how far along a ray something sits. This is the
fundamental reason one camera struggles with depth.</dd>

<dt>Back-projection</dt>
<dd>Running the camera backwards. Instead of asking &ldquo;where does this 3-D
circle land in the image?&rdquo;, we ask &ldquo;what 3-D circle would have produced
the ellipse I measured?&rdquo; Because we know the rim&rsquo;s real size, this has an
exact answer &mdash; no searching, no guessing.</dd>

<dt>Cone</dt>
<dd>Take a ray through every point around the fitted ellipse. Together they sweep
out a cone with its tip at the lens, widening away from the camera. The real rim
must be a slice through that cone. Slice a cone at the right angle and you get a
circle of the right size &mdash; and it turns out there are exactly <b>two</b> such
slices, which is the ambiguity a second camera resolves.</dd>

<dt>Rig</dt>
<dd>The two cameras and, crucially, <em>where they are relative to each other</em>
&mdash; measured in advance by photographing a printed calibration board. Without
that, two views cannot be compared: you would have two separate answers in two
separate coordinate systems and no way to combine them.</dd>
</dl>

<h3>Why two cameras</h3>
<p>Three reasons, and they are different from each other:</p>
<p><b>1. It breaks the tie.</b> One camera leaves two possible answers (the two
cone slices) &mdash; a disc tilted toward you and one tilted away produce the
<em>identical</em> ellipse. Both cameras see the same real robot, so the answer
they agree on is the real one.</p>
<p><b>2. Each covers the other&rsquo;s blind spot.</b> A camera judges sideways
position very well and distance-along-its-own-line-of-sight poorly &mdash; about
eleven times worse. Point a second camera from a different direction and its
<em>good</em> axis is the first one&rsquo;s <em>bad</em> axis.</p>
<p><b>3. It catches its own mistakes.</b> Two independent answers that disagree
mean something is wrong &mdash; a blocked view, a bad outline &mdash; and that can
be detected without knowing the right answer.</p>

<h3>Reading the pictures</h3>
<p>Each case below shows the two camera images, then the same situation drawn in
3-D from outside. In the 3-D views the <b>rays</b> are drawn leaving each camera
through the outline it measured, so you can see the two cones meeting where the
robot is. Where they meet is the answer.</p>
</section>"""


def _b64_png(data):
    """PNG data URI.

    `visualise._b64` hardcodes JPEG, which is right for its noisy sensor tiles
    and wrong for these: the 3-D panels are thin dark strokes on white, exactly
    the content JPEG rings around. PNG for line art, JPEG for camera frames.
    """
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _b64_jpeg(data):
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def _esc(v):
    return html.escape(str(v))


def _num(label, value, fmt="{:.3f}", unit="", over=False):
    v = "--" if value is None else fmt.format(value)
    cls = ' class="over"' if over else ""
    return f'<span><span class="k">{_esc(label)}</span> <b{cls}>{v}{_esc(unit)}</b></span>'


def _mtable(headers, rows):
    """A measurement table. ``rows`` are ``(label, [cells], is_best)``."""
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = ""
    for label, cells, best in rows:
        tds = "".join(f"<td>{c if c is not None else '--'}</td>" for c in cells)
        body += f'<tr class="{"best" if best else ""}"><td>{_esc(label)}</td>{tds}</tr>'
    return f'<table class="m"><thead><tr><th>{_esc(headers[0])}</th>{head[len(f"<th>{_esc(headers[0])}</th>"):]}</tr></thead><tbody>{body}</tbody></table>'


def _finding(title, lede, body):
    return (f'<section class="finding"><h3>{_esc(title)}</h3>'
            f'<p class="sub">{_esc(lede)}</p>{body}</section>')


def _rig_table(rows):
    body = "".join(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in rows)
    return f'<table class="rig">{body}</table>'


def _case(c):
    """One case: the stereo pair, the two 3-D panels, and the measured numbers."""
    frames = "".join(
        f'<figure><img src="{f["img"]}" alt="camera {_esc(f["name"])}: '
        f'fitted ellipse, true ellipse and the joint estimate reprojected" '
        f'loading="lazy">'
        f'<figcaption class="capt">camera {_esc(f["name"])} &middot; '
        f'IoU {f["iou"]:.3f} &middot; fit rms {f["rms"]:.2f} px</figcaption></figure>'
        for f in c["frames"]
    )
    scenes = "".join(
        f'<figure><img src="{s["img"]}" alt="{_esc(s["alt"])}" loading="lazy">'
        f'<figcaption class="capt">{_esc(s["caption"])}</figcaption></figure>'
        for s in c["scenes"]
    )

    over = c["worst_axis_mm"] > c["gate_mm"]
    nums = "".join([
        _num("dx", c["dx_mm"], "{:+.3f}", " mm"),
        _num("dy", c["dy_mm"], "{:+.3f}", " mm"),
        _num("dz", c["dz_mm"], "{:+.3f}", " mm"),
        _num("|pos|", c["pos_mm"], "{:.3f}", " mm", over=over),
        _num("normal", c["normal_deg"], "{:.3f}", "\N{DEGREE SIGN}"),
        _num("cross-view", c["discrepancy_mm"], "{:.3f}", " mm"),
        _num("branch margin", c["margin"], "{:.0f}", ""),
        _num("refine rms", c["refine_rms_px"], "{:.3f}", " px"),
    ])

    return f"""<section class="case">
  <h3>{_esc(c['title'])}</h3>
  <p class="sub">{_esc(c['subtitle'])}</p>
  <div class="pair">{frames}</div>
  <div class="numbers">{nums}</div>
  <div class="scenes" style="margin-top:12px">{scenes}</div>
</section>"""


def render_page(p):
    """The whole document, as a string. No file I/O -- the caller writes it."""
    cases = "".join(_case(c) for c in p["cases"])
    rig = _rig_table(p["rig_rows"])

    key = (
        f'<p><span class="swatch" style="background:{COL_TRUTH}"></span>'
        f'ground truth &nbsp; '
        f'<span class="swatch" style="background:{COL_EST}"></span>estimate '
        f'&nbsp; <span class="swatch" style="background:{S1}"></span>camera A '
        f'&nbsp; <span class="swatch" style="background:{S3}"></span>camera B</p>'
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(p['title'])}</title>
<style>{_css()}{EXTRA_CSS}</style>
</head>
<body>
<main class="wrap">
<h1>{_esc(p['title'])}</h1>
<p class="lede">{_esc(p['lede'])}</p>

<h2>How this works</h2>
{PRIMER}

<h2>The rig used here</h2>
{rig}

<h2>Cases</h2>
{cases}

<h2>Measurements</h2>
<p>Everything below is computed in the run that produced this page, from the same
poses shown above &mdash; not transcribed from an earlier session. Sample counts
are stated because they are small: this is a diagnostics page, not the sweep.</p>
{p['measurements']}

<h2>How to read this</h2>
{key}
<p>Each camera frame carries three ellipses: the <b>fitted</b> one the segmenter
measured in that view, the <b>true</b> rim projected from ground truth (dashed),
and the single joint 3-D estimate <b>reprojected</b> into that view. The third is
the one worth staring at &mdash; it is the same 3-D pose drawn into both images,
so if it lands on the robot in <em>both</em>, one pose explained two views at
once, which is the entire claim of the stereo solve.</p>

<p>In the wide 3-D panel, each line leaves a camera and passes through one point
on the outline that camera measured, stopping at the plane the estimated disc
lies in. So the ring of points where those lines stop <b>is the rim as that
camera saw it</b>, carried back out into space. How closely that ring sits on the
drawn circle is how much the measurement and the answer disagree, in millimetres.
Two cones meeting on one circle is what the second camera buys.</p>

<div class="caveat"><b>Nothing here is amplified.</b> The estimate sits a few
tenths of a millimetre from truth on a 20.4&nbsp;mm disk, so in the wide panel
the two poses are indistinguishable and in the zoom panel they separate by a
hairline. That is what this error looks like at true scale, and the numbers above
each pair carry the quantitative story. A figure that pushed them apart to make
the difference legible would be drawing an error the estimator does not
make.</div>

<p class="capt">{_esc(p['provenance'])}</p>
</main>
</body>
</html>"""

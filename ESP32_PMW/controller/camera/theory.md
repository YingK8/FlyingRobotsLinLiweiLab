# Chapter 1. The sensor: what the camera actually gives you

*Stage 1 of the pipeline. Produces: timestamped frames, and honest numbers about
the rate they arrive at. Consumed by: [chapter 2, calibration](../calib/theory.md).*

Everything downstream is a function of pixels and the times they were taken. This
chapter is about how much of each you really get, because on both counts the
device will tell you something that is not true.

## Reading order

| # | file | what it does |
|---|---|---|
| 1 | `sources.py` | `Capture`: images, video, or a live camera behind one `read()` |
| 2 | `elp.py` | opens the ELP with the mode *checked*, and the device profile |
| 3 | `elp_camera.json` | the mode table, as data rather than a list retyped per script |
| 4 | `modes.py` | measures what each mode actually delivers, and diagnoses why |
| 5 | `record.py` | the flight recorder: one video per camera plus the `frames.csv` that makes them a timed pair |
| 6 | `test_record.py` | checks that pairing and the take boundaries hold |

Start at `sources.MonoCamera`; everything else exists because of what §1.2 and
§1.3 say about it.

`record.py` is what every flight result downstream was measured on: it writes
`results/flights/<date>_<time>/{A,B}/` plus a `frames.csv` of per-camera capture
times, which is what lets `stereo.fuse` move both views to a common instant
(`pose/theory.md` §17). Without that file the two videos are just two videos.

## 1.1 Two rates, and why they are not the same number

A camera and a program that reads from it run at different speeds, and conflating
them is the single most common way to misdiagnose a pipeline.

`sources.MonoCamera` runs a **dedicated grabber thread** that does nothing but
`cap.read()` into a single-frame slot (`sources.py:_grab_loop`). The consumer takes
whatever is in the slot. So:

$$f_{\text{camera}} = \frac{n_{\text{grabbed}}}{\Delta t}, \qquad
  f_{\text{consumed}} = \frac{n-1}{t_{n-1} - t_0}$$

are independent. Processing can never slow the sensor: it only lowers how often a
*pose* comes out. Frames the consumer never saw are counted in `n_dropped`.

**`n_dropped` is consumer loss, not camera loss** (`sources.py:222-223`). It
increments when the slot was still full, so a slow estimator inflates it and it
says nothing about USB. Reading it backwards is exactly how a Python-loop ceiling
gets blamed on hardware: see the 160×120 anomaly in §1.3.

The slot holds **only the newest frame**, deliberately. For feedback control a
stale frame is worse than no frame, so a queue that buffered backlog would be
actively harmful: it would trade the one thing the loop needs (recency) for the
one thing it does not (completeness).

## 1.2 Why `CAP_PROP_FPS` cannot be trusted, and what to do instead

`cv2.VideoCapture.set` returns success while quietly giving you something else,
and `CAP_PROP_FPS` reports **what was requested, not what arrives**. On macOS
`CAP_PROP_FOURCC` reads back as `0xFFFFFFFF`, unreadable, and AVFoundation
ignores the FOURCC you set and negotiates its own.

So `elp.open_elp` checks the *granted* frame size against the requested one and
raises by default (`strict=True`). This is not fastidiousness. A silent fallback
from 640×400 to 640×480 changes the intrinsics' scale, and every distance the
estimator reports is then wrong by a fixed factor that nothing downstream can
detect: the same class of error as the ChArUco print-scale trap in
[chapter 2](../calib/theory.md#144-the-one-error-the-residual-cannot-see).

The frame *rate* is deliberately not checked the same way, because the property
that would be checked is the one that lies. Rate is measured by timing real reads
: that is what `modes.py` is for.

## 1.3 The frame-time model, and the anomalies it fails to explain

Time $n$ reads after a **time-based** warmup, and fit

$$t_{\text{frame}} = \max\!\left(\frac{1}{f_{\text{req}}},\; a + b\,W H\right)$$

where $a$ is fixed per-frame overhead (USB transaction, MJPEG decode, the Python
round trip) and $b$ is cost per pixel (`modes.fit_frame_time`). Only modes running
*below* their requested rate carry information: one that hits its asked rate is
telling you about the request, not the cost.

The model earns its place mainly by what it **cannot** absorb. A planning sweep
measured this camera at 121.4 fps on its native 1280×800 and 209.9 at 640×480,
matching the datasheet, but three modes disagreed:

| mode | asked | measured | residual |
|---|---|---|---|
| 800×600 | 120 | 98.8 | undershoots with *fewer* pixels than a faster mode |
| 640×400 | 210 | 271.3 | exceeds the asked rate |
| 160×120 | 640 | 285.4 | undershoots badly |

No model of the form above can explain a mode that is slower than one with more
pixels. A large residual there is what promotes "the sensor does not have this
mode" from a guess to the leading hypothesis. `modes.py` separates the four
candidate explanations by construction:

| hypothesis | how it is separated |
|---|---|
| consumer-bound | report both rates of §1.1; `fps_grabbed ≫ fps_consumed` proves a Python ceiling |
| colour conversion inside the measurement | `--grayscale` A/B. The sensor is mono, so MJPG returns replicated grey and `cvtColor` is pure overhead |
| mode is not native | `--probe-formats` asks ffmpeg what AVFoundation reports as real: a path sharing no code with the sweep |
| sample too short | warmup by **time**, not frame count. Ten frames at 640 fps is 16 ms; nothing has settled |

Warmup by frame count is the subtle one, and it is why the original numbers are
untrustworthy rather than merely imprecise. `--frames 100 --warmup-s 0` reproduces
the original conditions: reproducing an anomaly is the regression test for the
measurement itself.

## 1.4 Capture skew of a free-running pair

Two USB cameras without a hardware trigger have no common clock. They free-run and
their frames land wherever they land, so `sources.StereoCamera` **measures** the
skew on every read rather than assuming it, and reports `skew_stats()`.

Model the two phases as independent and uniform on a frame period $T$. Their
difference is triangular on $[-T, T]$, so the expected absolute skew is

$$\mathbb{E}|\Delta| = \frac{T}{3}.$$

At 121 fps that is 2.8 ms; at 421 fps, 0.8 ms. Whether that matters is a question
about *motion*, not about sync: the robot moves 15–22 mm/s in hover, so 2.8 ms is
about 0.05 mm: comfortably inside the budget. During a 1.4 m/s² climb it is
worse, and the software remedy is `filter.PoseFilter.predict_ahead`, which
advances the earlier view to the later one's timestamp. A hardware trigger remains
the right answer.

`StereoCamera` returns the **mean** capture time of the pair, not the first
camera's. Taking the first would bias every pose by half the skew in a fixed
direction: exactly the kind of error that survives filtering.

## 1.5 What the sensor is

The ELP module (VID `0x32E4`, PID `0x9281`) is an OV9281-class **monochrome global
shutter**. Both properties matter downstream and neither is incidental:

- **Global shutter** means every row is exposed simultaneously. A rolling shutter
  would skew the rim ellipse as a function of spin rate, and the conic solve of
  [chapter 3](../pose/theory.md) fits a circle's projection: a sheared ellipse
  would be absorbed as a false tilt.
- **Monochrome** means chroma is identically zero at every pixel. Measured over
  both frames in `pose/assets/captures/elp/`: `max(BGR) − min(BGR) ≡ 0`. That is
  what disqualifies colour-based clutter rejection, and the whole of
  [§15](../pose/theory.md#152-why-chroma-cannot-rescue-it) follows from it.

## 1.6 A timestamp row with no frame: encoder drops in the flight recorder

`FlightWriter` encodes on its own thread behind a bounded queue (`QUEUE_DEPTH = 64`). A
frame that arrives while the queue is full is **dropped from the mp4 and counted**, so
that a slow encoder can never stall a flight. Until 2026-09-13 its `frames.csv` row was
written anyway. This is a different counter from §1.1's `n_dropped`: that one is a pose
consumer skipping frames it never saw; this one is frames that were captured, timed, and
then never written.

`disc_axis` reads the video and `record.read_index` and pairs **mp4 frame $i$ with row
$i$**. Let $D(i)$ be the number of rows dropped before the $i$-th written frame. Its true
time is $t_{i+D(i)}$ but it is given $t_i$, an error of

$$\varepsilon(i) = t_{i+D(i)} - t_i \approx \frac{D(i)}{f},$$

which grows through the take and never recovers. At $f \approx 210$ Hz a take that loses
1761 frames before its coil cut (~40 s in) has its cut drawn up to 8 s late.

**Proof it is this and nothing else.** On take `2026-09-13_101717`: `frames.csv` 12734 rows,
each mp4 10973 frames, `meta.json` `dropped` 1761, and $12734 - 10973 = 1761$ exactly. A
drop-free take reads 13177 / 13177 / 13177. The mp4 cannot help: `ffprobe` gives 10973
packets exactly $1/210$ s apart, no gap. `cv2.VideoWriter` writes a constant-rate container,
so the file keeps the order of the frames and none of their times.

**What it looked like.** On 91 of design B's 125 takes (`results/alignment_rate/`): the
axis swing drawn *before* $t = 0$, the scheduled spin-down drawn ~1 s *after* the cut,
takes whose `axis.csv` ends seconds before `TILT_OFF`. It was read as physics for a whole
session -- "non-responders", a bistable hold attitude, CW spin-up -- and 14 takes were
excluded on it, until the operator, who watched every run, said they were all fine. The
negative results that came out of it are retracted in the campaign README (2026-09-13).

**What causes the drops is only partly known.** Not the medium: 70 s of a still scene
dropped nothing to the FAT32 stick and nothing to the internal disk (both ~209 fps). Not
only concurrent analysis either: a re-record with nothing else running still dropped 43
to 5585 frames a take, heaviest at 120 Hz. The working explanation is encoder cost on the
spinning, blurred disc, made worse by anything else competing for the machine.

**The fix** (`record.py`): each row carries `written` (1 queued, 0 dropped) and
`read_index` returns written rows only, so row $i$ is frame $i$ for every consumer
(`disc_axis`, `live_viz`, `sync`; `pose/spin.py` reads the file itself and filters too).
Drops still cost frames, but no longer cost time.

**Repairing takes recorded before the flag.** Which rows were dropped is gone, but a take
is often drop-free at one end. Pairing from the *end* -- frame $i$ to row
$N_{\text{rows}} - N_{\text{mp4}} + i$ -- is exact from the last drop onward; the original
forward pairing is exact up to the first. Each is tested against the one event every run
has: the axis swings when coils A and C are cut. Drop-free takes place the swing onset
0.02-0.08 s after the `KILL` label (the response lag, per frequency); a pairing is accepted
when the onset lands within 0.04 s of its frequency's drop-free median
(`ai/alignment/align_pairing_test.py`):

| verdict | takes | timing after the cut |
|---|---|---|
| drop-free | 43 | exact |
| backward pairing lands the swing | 31 | exact: `retime_takes.py` rewrites `t` from the end, originals kept as `*.forward_t.csv` |
| forward pairing lands the swing | 20 | exact at the cut, approximate after (drops fell later) |
| neither | 39 | drops on both sides of the cut: re-recorded with the flag (38), or excluded (120 Hz) |

`retime.json` in each take records the verdict and `post_cut_exact`, so a settling time or a
cone decay can be taken only where the seconds after the cut are exact.

## 1.7 Correspondence with the implementation

| Model element | Code |
|---|---|
| Grabber thread, drop-oldest slot (§1.1) | `sources.py` `MonoCamera._grab_loop`, `read` |
| Consumer loss vs camera loss (§1.1) | `MonoCamera.n_dropped`, `n_grabbed` |
| Granted-vs-requested mode check (§1.2) | `elp.py` `open_elp(strict=True)` |
| Mode table as data (§1.3) | `elp_camera.json`, `elp.modes`, `elp.parse_mode` |
| Two measured rates (§1.1, §1.3) | `modes.py` `_stats` → `fps_grabbed`, `fps_consumed` |
| Frame-time fit (§1.3) | `modes.fit_frame_time` |
| Per-mode verdict (§1.3) | `modes.classify` |
| Native-mode probe (§1.3) | `elp.native_modes_ffmpeg` |
| Skew measurement (§1.4) | `sources.py` `StereoCamera.read`, `skew_stats` |
| Mean pair timestamp (§1.4) | `StereoCamera.read` |
| One camera or two, branch-free (§1.1) | `elp.open_group`, `elp.as_frames` |
| Encoder drop, row kept and flagged (§1.6) | `record.py` `FlightWriter.add` (`written`), `QUEUE_DEPTH` |
| Row $i$ = mp4 frame $i$ (§1.6) | `record.read_index`; `pose/spin.py` `watch` filters the same flag |
| Forward vs backward pairing test (§1.6) | `ai/alignment/align_pairing_test.py` |
| Retiming a pre-flag take (§1.6) | `ai/alignment/retime_takes.py` → `retime.json`, `*.forward_t.csv` |

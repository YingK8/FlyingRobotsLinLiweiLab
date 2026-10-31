# Design 1 vs design B -- alignment-rate comparison

| file | made by | what |
|---|---|---|
| `alignment_rate_1_vs_B.png` | `controller/control/compare_designs.py rates` | every repeat's alignment rate (top), every repeat's swing (middle), B/1 ratio of medians with bootstrap CI and Mann-Whitney p (bottom) |
| `alignment_rate_stats.csv` | same | per frequency: n, medians, ratio, CI, raw and Bonferroni p, `differs` |
| `rate_and_settling_1_vs_B.png` | `compare_designs.py summary TRIALS_1 TRIALS_B SETTLING_1 SETTLING_B` | alignment rate (top) and settling time to a 10% band (bottom) against drive frequency: dots = repeats, bar = median, whisker = middle 50%, measured / analysed under each cluster, * = designs differ (Mann-Whitney, Bonferroni per row) |
| `rate_and_settling_stats.csv` | same | per metric and frequency: n, medians, B/1 ratio with bootstrap CI, raw and Bonferroni p, `differs` |
| `cone_tau_1_vs_B.png` | `compare_designs.py cone SETTLING_1 SETTLING_B` | decay time constant of the cone ('precession') half-angle after the cut against drive frequency, same encoding as the settling row: per repeat `cone_tau_s`, rotor-stopped repeats excluded, drives above 104 Hz left out (coning aliased, `theory.md` 25.14) |
| `cone_tau_stats.csv` | same | per frequency: n, medians (the `*_deg_s` column names are inherited; the unit is s), B/1 ratio, CI, p, `differs` |
| `traces_1_vs_B.png` | `compare_designs.py traces` | average axis (angle to the final resting axis) and cone ('precession') half-angle vs time from the cut, every repeat thin with the median bold, one column per drive frequency, each repeat drawn to its own spin-down (15 s after the cut on design B, 5 s on design 1) |
| `traces_1_vs_B.csv` | same | the pooled series behind it: design, freq, n, t, median/q25/q75 of both |
| `spin_stop_survey.txt` | inline script over `alignment_rate.settle_rows` (`_spin_end`) | per take, when the rotor stopped after the cut, both designs; the detector is `alignment_rate.spin_stop_from` |
| `design1_campaign/` | `alignment_rate.campaign(..., out_dir=...)` | design 1 (`../20260909_205843`) re-analysed to get `campaign_trials.csv`; its `campaign.csv` matches the tracked report exactly at all 11 frequencies. Written here so the tracked design-1 report is untouched. |

Design B's per-repeat file is `../20260910_droneB/report/campaign_trials.csv`.

```bash
uv run python controller/control/compare_designs.py summary \
  results/alignment_rate/compare_1_vs_B/design1_campaign/campaign_trials.csv \
  results/alignment_rate/20260910_droneB/report/campaign_trials.csv \
  results/alignment_rate/20260909_205843/report/settling.csv \
  results/alignment_rate/20260910_droneB/report/settling.csv
uv run python controller/control/compare_designs.py traces \
  results/alignment_rate/20260909_205843 results/alignment_rate/20260910_droneB
uv run python controller/control/compare_designs.py rates \
  results/alignment_rate/compare_1_vs_B/design1_campaign/campaign_trials.csv \
  results/alignment_rate/20260910_droneB/report/campaign_trials.csv
```

**2026-09-14: regenerated after the timing repair** (`../20260910_droneB/README.md`, 2026-09-13
~14:00 onward; `controller/camera/theory.md` §1.6). Design B: 43 drop-free takes, 31 lined up
at the cut by pairing from the end, 20 exact at the cut by the original pairing (their
post-cut seconds approximate, `retime.json` `post_cut_exact: false`), 31 re-recorded with the
fixed recorder, 7 excluded and not re-recorded (90 x3, 100 x3, 110 x1), 120 Hz all refused
(Nyquist). **Design 1 was checked too:** 6 of its 117 good takes dropped frames (60 Hz 1
take, 138 frames; 70 Hz 2, max 27; 80 Hz 2, max 28; 90 Hz 1, 44), a cut drawn up to ~0.13-0.66 s
late on those six. Its video and `frames.csv` are archived (`results/flights/Archive.zip`), so
they are not lined up; they are still in these figures.

**Results after the repair (2026-09-14):**
- **Alignment rate: no frequency differs** after Bonferroni. Closest: 50 Hz (B/1 0.37, p 0.071)
  and 30 Hz (B/1 0.55, p 0.098). The 2026-09-13 "30 Hz differs" was the timing error.
- **Settling time differs at 80 Hz only:** design B 1.12 s vs design 1 1.82 s (B/1 0.61,
  p 0.003, n 14 vs 5).
- **Cone decay:** design B decays faster below ~50 Hz (0.12 vs 0.86 s at 10 Hz, 0.29 vs 0.72 s at
  20 Hz) and the two meet from 60 Hz up; no frequency passes Bonferroni (20 Hz p 0.05, 40 Hz
  p 0.058).
- Design-B repeat counts at 90-110 Hz are small (analysed 5, 4, 6); read those points as indicative.

**Superseded (2026-09-13 ~13:30):** every figure here was mistimed for design-B takes with
dropped frames (91 of 125). The bullet list further down headed "Regenerated 2026-09-13" is
from that mistimed state.

**Regenerated 2026-09-13** with design B complete (100 takes; see its README). What to know:
- design B has no 120 Hz column anywhere: all five takes are refused by the Nyquist gate;
- design B's 70 Hz lost its rate (3/7 -> 0/9) when takes 9-10 flipped the modal population;
- median lines in `rate_and_settling_1_vs_B.png` and `cone_tau_1_vs_B.png` are joined through
  every frequency with a median, at the operator's request. Design B's rate line therefore runs
  straight from 40 to 110 Hz across 50-100 Hz, where fewer than 3 repeats gave a rate -- read
  the m/n counts, not the line, there. `alignment_rate_1_vs_B.png` still breaks its lines.
- the only frequency where the designs differ after correction is the 30 Hz rate (B/1 0.42,
  Bonferroni p 0.049). Design B's cone decays faster at 10-50 Hz (B/1 0.14-0.68) but no
  frequency passes the Bonferroni test (n 3-8 per side).

**Solved takes only.** A take enters the moment its `axis.csv` exists; `disc_axis.solve`
now writes `.partial` and renames on success, so an existing `axis.csv` is always complete.
Re-run `alignment_rate.campaign` on design B and then the command above after more takes
are solved or recorded.

Read before quoting: design 1's settled window was clamped to 4-5 s, design B's is 4-6 s
(`theory.md` 24.12). Segmentation is NOT a difference between them -- view A/B agreement is
~16-18 deg median on both, and the same on design-B takes with and without a rate.

**Rotor stops.** Past the moment a rotor stops, `disc_axis` fits an ellipse to four still blades
and the "axis" means nothing. `alignment_rate.spin_stop` detects the stop per take from the
segmented mask's fill of its fitted ellipse, and `traces` cuts every repeat 1 s before it.
Design B stops within its 15 s hold on 13 of 53 repeats, most at 50 Hz; design 1 never does
within its 5 s hold. `theory.md` 24.13.

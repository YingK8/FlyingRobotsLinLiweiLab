# Design 1 vs design B -- alignment-rate comparison

| file | made by | what |
|---|---|---|
| `alignment_rate_1_vs_B.png` | `controller/control/compare_designs.py rates` | every repeat's alignment rate (top), every repeat's swing (middle), B/1 ratio of medians with bootstrap CI and Mann-Whitney p (bottom) |
| `alignment_rate_stats.csv` | same | per frequency: n, medians, ratio, CI, raw and Bonferroni p, `differs` |
| `rate_and_settling_1_vs_B.png` | `compare_designs.py summary TRIALS_1 TRIALS_B SETTLING_1 SETTLING_B` | alignment rate (top) and settling time to a 10% band (bottom) against drive frequency: dots = repeats, bar = median, whisker = middle 50%, measured / analysed under each cluster, * = designs differ (Mann-Whitney, Bonferroni per row) |
| `rate_and_settling_stats.csv` | same | per metric and frequency: n, medians, B/1 ratio with bootstrap CI, raw and Bonferroni p, `differs` |
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

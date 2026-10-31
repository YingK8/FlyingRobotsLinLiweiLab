# Design A vs B vs C

Design A is the campaign the two-design figures in `../compare_1_vs_B/` call "design 1"
(`../20260909_205843`). Design B is `../20260910_droneB`, design C is `../20260914_droneC_10hzs`.

| file | made by | what |
|---|---|---|
| `alignment_rate_A_vs_B_vs_C.png` | `compare_designs.py abc ...` (`build_rate_abc`) | alignment rate after the cut against drive frequency: dots = modal repeats, bar = median, whisker = middle 50%, m/n under each cluster |
| `alignment_rate_A_vs_B_vs_C.csv` | same | one row per design x drive: `n_measured`, `n_analysed`, `median`, `q25`, `q75`, and every repeat's value (`;`-joined) |
| `cone_time_constant_A_vs_B_vs_C.png` | `compare_designs.py abc ...` (`build_cone` with a third file) | decay time constant of the cone ('precession') half-angle after the cut, drives <= 104 Hz |
| `cone_time_constant_A_vs_B_vs_C.csv` | same | same columns as above |
| `traces_A_vs_B_vs_C.png` | `compare_designs.py traces ROOT_A ROOT_B ROOT_C` | average axis (angle to the final resting axis, top) and cone half-angle (bottom) vs time from the cut, every repeat thin with the median bold, one column per drive any design has (a design without that drive is absent from its panel) |
| `traces_A_vs_B_vs_C.csv` | same | the pooled series behind it: design, freq, n, t, median/q25/q75 of both rows |

```bash
uv run python controller/control/compare_designs.py abc \
  results/alignment_rate/compare_1_vs_B/design1_campaign/campaign_trials.csv \
  results/alignment_rate/20260910_droneB/report/campaign_trials.csv \
  results/alignment_rate/20260914_droneC_10hzs/report/campaign_trials.csv \
  results/alignment_rate/20260909_205843/report/settling.csv \
  results/alignment_rate/20260910_droneB/report/settling.csv \
  results/alignment_rate/20260914_droneC_10hzs/report/settling.csv
uv run python controller/control/compare_designs.py traces \
  results/alignment_rate/20260909_205843 results/alignment_rate/20260910_droneB \
  results/alignment_rate/20260914_droneC_10hzs
```

**Not like-for-like.** Design C's ramp climbed to the drive at 10 Hz/s (`tilt_run --seg2-rate 10`);
A and B at 3.5 Hz/s, 2.8 from 60 Hz. The asterisks test A against B only (two-sided
Mann-Whitney, Bonferroni); no pair involving C is tested.

**Design C is incomplete (2026-09-14).** 10-70 Hz only: 80 Hz has one take, which gives no usable
kill, and 90-120 Hz were not recorded by the operator's choice. 50 Hz repeats 1-2 were rejected by
the operator (`operator_rejected` in `050hz/index.csv`), so 50 Hz has 8. The missing 80 Hz x9 and
50 Hz x2 are still to be recorded; re-run the solve, `alignment_rate.py --campaign` and `--settle`
on design C, then the command above.

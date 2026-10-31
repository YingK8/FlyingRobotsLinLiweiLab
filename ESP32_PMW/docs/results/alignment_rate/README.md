# results/alignment_rate

The alignment-rate campaigns: design 1 (`20260909_205843`) and design B, the half outer ring
(`20260910_droneB`, complete 2026-09-13), plus design B's 20 and 50 Hz re-recorded after the
100 C stop (`20260912_droneB_rerun`). Each campaign folder and `compare_1_vs_B/` has its own README; this file
maps what sits at this level to the code that made it.

| file | made by | what |
|---|---|---|
| `tilt_flight_processed_csv_design_1.zip` | `controller/control/package_takes.py --designs 1` | design 1's 117 good takes, one labelled CSV each (`tilt_flight_design_1_<freq>hz_<repeat>.csv`: fused axis, minor-axis estimate, disc centre and radius, time from the cut, schedule phase, rotor-stopped flag), plus `manifest.csv`, the campaign tables and a README |
| `tilt_flight_processed_csv.zip` | `controller/control/package_takes.py` | the same for BOTH designs -- built once design B's takes (on the USB stick) are reachable |

`summary.csv` predates these and is stale -- `20260909_205843/report/README.md` says so.
Videos are never deleted: design 1's are in `results/flights/Archive.zip`, design B's on the
USB stick recorded per take in each package's `manifest.csv`.

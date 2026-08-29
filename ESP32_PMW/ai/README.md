# `ai/`: everything that is not the flight pipeline

The four-stage vision → control pipeline lives in
[`../controller/`](../controller/README.md). This directory holds everything
around it: the sweeps that drive the hardware, the system ID that fits models to
what came back, the validation that checks the pipeline against ground truth, and
the plotting. Code here may depend on all four controller stages; nothing in
`controller/` may depend on this.

| directory | what is in it |
|---|---|
| `experiments/` | drives the rig: schedule generators (`gen_coupling_experiment.py`), run drivers (`run_experiment.py`, `run_coupling_sweep.py`), serial capture (`record_serial.py`, `trigger_reset_log.py`) |
| `instrumentation/` | the PicoScope: `picoscope_capture.py` (see [`picoscope_capture.md`](instrumentation/picoscope_capture.md)), `picoscope_record.py`, `segment_rms.py` |
| `sysid/` | fits models to captured data: `coupling_matrix.py`, `fit_rlc_model.py`, `fit_mutual_inductance.py`, `cs_calibration.py`, `pid_autotune.py` |
| `validation/` | checks the pose pipeline against rendered and recorded ground truth: sweeps, error-model fits, overlays |
| `tests/` | the test suite. `run_tests.py` is the entry point |
| `plots/` | every figure in `results/` and `data/`; one `plot_*.py` per figure family |
| `notes/` | written-up findings that do not belong in a code comment, e.g. [`pose_appearance.md`](notes/pose_appearance.md) |

Run anything here from the `ESP32_PMW/` directory:

```bash
uv run python ai/tests/run_tests.py
uv run python ai/plots/plot_rms.py
```

`results/README.md` maps each output directory back to the script that generates
it; `data/README.md` indexes the captures those scripts consume.

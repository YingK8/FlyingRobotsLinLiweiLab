# Live-preview captures, 2026-09-04

Bench diagnostics from bringing up the C++ capture + interleaved tracker
(`controller/pose/tracker.py`, `pose/theory.md` 22). Scratch material, not measurements
anything depends on -- delete freely.

| file | what it shows | made by |
|---|---|---|
| `2026-09-04_tracker_live_224Hz.mp4` | the tracker running live on the drone, **3891 poses at 223.8 Hz, 5 lost**. Green = rim from the fused 3-D pose, blue arrow = rotor axis, thin red = each camera's own segmentation | live capture through `pose/tracker.py`, decimated to 30 fps |
| `2026-09-04_preview_underlit.mp4` | the same rig once the scene was under-lit: both views raw on top, contrast-stretched below. Yellow outline is a brightest-blob segmenter finding the **hub, not the rim**, which is why the discrepancy gate rejects every frame | same, with the diagnostic HUD |
| `2026-09-04_cameras_now.png` | single frame pair in the under-lit state, mean 20/22 against a healthy 59/82 | `pose/tracker.py` |
| `2026-09-04_blob_vs_rim.png` | why the silhouette shortcut of `pose/theory.md` 16.27 fails: the mast drags the bounding box off the rim | as above |
| `2026-09-04_fit_tolerance_ab.mp4` | **the fitting change**: same frames, same core, `REFINE_TOL_ANALYTIC` 1e-3 -> 5e-4. Red is before, green is now, both reprojected onto camera A -- they land together, and only one is steady. Right panel traces rotor tilt; 2nd difference 0.37 -> 0.24 deg | `scratchpad/demo_ab.py`, `pose/theory.md` 16.29 |
| `2026-09-04_bench_overlay.mp4` | the whole 08-29 flight with every overlay, 1820 frames, 100% posed, at the current tolerance | `pose/demo_video.py` |
| `2026-09-04_demo_bench_replay.mp4` | `2026-08-29_231418` replayed, 1820 frames, 100% posed. **The Python estimator**, not the tracker -- kept as the before picture | `pose/demo_video.py` |

Healthy lighting on this bench reads frame **mean 59 / 82 with max 235**; at mean 20 the
rim stops being the brightest thing in frame and nothing solves. See `pose/theory.md` 22.6.

"""Fresh-frame contract tests using deterministic decoder/detector doubles."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from LLM.auto_pose import _fresh_centre, generate_annotations
from LLM.pose import TIP_COLUMNS


def _seg(frame):
    if frame.mean() == 0:
        return None
    return SimpleNamespace(mask=np.ones_like(frame), ellipse=((10., 11.), (20., 12.), 0.),
                           fit_rms_px=.1, area_px=1000.)


class _Decoder:
    def __init__(self, values):
        self.frames = iter(np.full((24, 32, 3), value, np.uint8) for value in values)
        self.closed = False

    def isOpened(self):
        return True

    def read(self):
        frame = next(self.frames, None)
        return frame is not None, frame

    def release(self):
        self.closed = True


def run():
    captures = {"A": _Decoder([200, 0, 200]), "B": _Decoder([200, 200, 0])}
    flight = {"video_a": "A", "video_b": "B", "frame_indices": np.array([0, 2, 5]),
              "metadata": {"decoded_image_sizes": {"A": [32, 24], "B": [32, 24]}}}
    calls = []

    def detect(frame, **kwargs):
        calls.append(frame.mean())
        return _seg(frame)

    with patch("LLM.auto_pose.load_flight", return_value=flight), \
         patch("LLM.auto_pose._validated_rig"), \
         patch("LLM.auto_pose.cv2.VideoCapture", side_effect=lambda path: captures[path]), \
         patch("controller.pose.disc_pose.segment_disc", side_effect=detect):
        result = generate_annotations("unused", "unused.json")
    assert result["frame_index"].tolist() == [0, 2, 5]
    assert len(calls) == 6
    assert np.isnan(result.iloc[1]["ax_px"]) and result.iloc[1]["bx_px"] == 10
    assert result.iloc[2]["ax_px"] == 10 and np.isnan(result.iloc[2]["bx_px"])
    assert not result.iloc[1]["a_detection_valid"] and not result.iloc[2]["b_detection_valid"]
    assert result[list(TIP_COLUMNS)].isna().all().all()
    assert all(decoder.closed for decoder in captures.values())
    assert result.attrs["orientation_observed"] is False
    config = result.attrs["config"]
    frame = np.full((24, 32), 200, np.uint8)
    seg = _seg(frame)
    seg.mask = None
    assert _fresh_centre(frame, lambda *_a, **_k: seg, None, config)[2] == "missing_current_frame_mask"
    seg = _seg(frame)
    seg.fit_rms_px = 10
    assert _fresh_centre(frame, lambda *_a, **_k: seg, None, config)[2] == "ellipse_fit_residual"
    import cv2
    from controller.pose.disc_pose import segment_disc
    synthetic = np.zeros((120, 160), np.uint8)
    cv2.ellipse(synthetic, (80, 60), (40, 20), 0, 0, 360, 200, -1)
    centre, _, reason = _fresh_centre(synthetic, segment_disc, None, config)
    assert reason == "experimental_ellipse_center_proxy" and np.linalg.norm(centre - [80, 60]) < 1
    assert _fresh_centre(np.zeros_like(synthetic), segment_disc, None, config)[0] is None
    print("LLM auto pose: fresh-frame gating, dropped-index mapping, missing-axis, fit quality, and synthetic segmentation checks passed")


if __name__ == "__main__":
    run()

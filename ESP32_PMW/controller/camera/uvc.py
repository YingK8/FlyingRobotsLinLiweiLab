#!/usr/bin/env python3
"""Fixed, fully manual camera settings on the ELPs over UVC, which AVFoundation cannot set.

macOS gives no exposure control for a UVC camera through AVFoundation, and OpenCV's
AVFoundation backend ignores CAP_PROP_EXPOSURE. So this sends the Camera Terminal requests
itself (UVC 1.5, 4.2.2.1: CT_AE_MODE_CONTROL, CT_EXPOSURE_TIME_ABSOLUTE_CONTROL) as class
requests on the VideoControl interface. libusb can send those while macOS's own UVC driver
holds the stream; that is how uvc-util does it. `calib/theory.md` 16.4a has the reason.

Verified 2026-11-04 on both ELPs, while AVFoundation streams 1280x800: read-back 2.00 ms,
frame mean 84 (auto) -> 4 (2 ms) -> 12 (8 ms) -> 85 (auto again). Auto holds 119 fps, so
its extra ~7x is GAIN, not exposure; GET_CUR on exposure does not track auto.

The ELP's automatic functions (from its descriptors, 2026-11-04): auto exposure, auto white
balance, backlight compensation, power-line (anti-flicker) compensation. `apply` turns off
exposure, white balance and power-line auto (backlight only weights auto exposure's
metering, so manual exposure leaves it nothing to do) and sets exposure and gain; everything else it has (brightness, contrast, gamma,
sharpness...) is a fixed value already. Nothing in the pipeline turns auto back on (operator,
2026-11-04): the cameras stay as `apply` left them until replugged.

    apply(0.002, gain=100)   # every connected ELP: all manual; returns the read-back
    set_auto()               # by hand only: the camera's own auto exposure

    uv run python controller/camera/uvc.py               # self-check, no hardware
    uv run python controller/camera/uvc.py --set 2 100   # exposure ms, gain
    uv run python controller/camera/uvc.py --auto
    uv run python controller/camera/uvc.py --meter 2 100 # live view, brightness, fps
"""

from __future__ import annotations

import sys
from array import array

VID, PID = 0x32E4, 0x9281          # camera/elp_camera.json
SET_CUR, GET_CUR, GET_MIN, GET_MAX = 0x01, 0x81, 0x82, 0x83
CT_AE_MODE, CT_EXPOSURE_ABS = 0x02, 0x04
# Processing Unit selectors (UVC 1.5 table A-13) and their sizes in bytes
PU_BACKLIGHT, PU_GAIN, PU_POWER_LINE, PU_WB_TEMP_AUTO = 0x01, 0x04, 0x05, 0x0B
AE_MANUAL, AE_APERTURE_PRIORITY = 0x01, 0x08   # UVC's "auto" for a fixed-iris camera
UNIT_S = 100e-6                    # CT_EXPOSURE_TIME_ABSOLUTE counts 100 us
HOMEBREW_LIBUSB = "/opt/homebrew/lib/libusb-1.0.dylib"   # not on ctypes' search path


def units(config):
    """(VideoControl interface, Camera Terminal ID, Processing Unit ID) from a raw
    configuration descriptor. They go into every request's wIndex, and differ between
    UVC devices."""
    ifc, ct, pu, i = None, None, None, 0
    while i + 1 < len(config) and config[i]:
        d = config[i:i + config[i]]
        i += config[i]
        if d[1] == 0x04 and len(d) >= 7:                          # INTERFACE
            ifc = d[2] if (d[5], d[6]) == (0x0E, 0x01) else None   # video control only
        elif d[1] == 0x24 and ifc is not None and len(d) >= 6:     # CS_INTERFACE
            if d[2] == 0x02 and int.from_bytes(d[4:6], "little") == 0x0201:   # ITT_CAMERA
                ct = d[3]
            elif d[2] == 0x05:                                     # PROCESSING_UNIT
                pu = d[3]
        if ct is not None and pu is not None:
            return ifc, ct, pu
    raise OSError("no UVC camera terminal + processing unit in the configuration descriptor")


def _devices():
    import usb.backend.libusb1 as libusb1
    import usb.core
    be = libusb1.get_backend() or libusb1.get_backend(find_library=lambda _: HOMEBREW_LIBUSB)
    devs = list(usb.core.find(find_all=True, idVendor=VID, idProduct=PID, backend=be))
    if not devs:
        raise OSError(f"no ELP ({VID:04x}:{PID:04x}) on USB")
    out = []
    for dev in devs:
        config = bytes(dev.ctrl_transfer(0x80, 6, 0x0200, 0, 4096))   # GET_DESCRIPTOR(config)
        out.append((dev, *units(config)))
    return out


def _xfer(dev, rtype, req, wvalue, windex, buf):
    """A control transfer WITHOUT claiming the interface. pyusb's `ctrl_transfer` claims
    any interface a request is addressed to, and macOS's UVC driver already owns the
    VideoControl interface: Errno 13 (measured 2026-11-04). The request itself needs no
    claim, so it goes to the libusb backend directly."""
    ctx = dev._ctx
    ctx.managed_open()
    n = ctx.backend.ctrl_transfer(ctx.handle, rtype, req, wvalue, windex, buf, 1000)
    return bytes(buf[:n])


def _get(dev, ifc, tid, req, cs, n):
    return int.from_bytes(_xfer(dev, 0xA1, req, cs << 8, tid << 8 | ifc, array("B", bytes(n))),
                          "little")


def _set(dev, ifc, tid, cs, value, n):
    _xfer(dev, 0x21, SET_CUR, cs << 8, tid << 8 | ifc, array("B", value.to_bytes(n, "little")))


def _in_range(dev, ifc, unit, cs, n, value, what):
    lo, hi = (_get(dev, ifc, unit, r, cs, n) for r in (GET_MIN, GET_MAX))
    if not lo <= value <= hi:   # refused, not clamped: the caller reasons from the value asked
        raise ValueError(f"{what} {value} is outside this ELP's {lo}-{hi}")


def apply(exposure_s, gain):
    """Every connected ELP fully manual: exposure (s), gain (raw 16-255 on this camera),
    and auto white balance and power-line compensation OFF. Backlight compensation is left
    at its default: on this ELP it is a 32-100 weight on auto exposure's metering, not an
    on/off (0 stalls, measured 2026-11-04), and with exposure manual it acts on nothing.
    Returns each camera's read-back as a dict."""
    exp = round(exposure_s / UNIT_S)
    got = []
    for dev, ifc, ct, pu in _devices():
        _in_range(dev, ifc, ct, CT_EXPOSURE_ABS, 4, exp, "exposure [100 us]")
        _in_range(dev, ifc, pu, PU_GAIN, 2, gain, "gain")
        _set(dev, ifc, ct, CT_AE_MODE, AE_MANUAL, 1)
        _set(dev, ifc, ct, CT_EXPOSURE_ABS, exp, 4)
        _set(dev, ifc, pu, PU_GAIN, gain, 2)
        _set(dev, ifc, pu, PU_WB_TEMP_AUTO, 0, 1)
        _set(dev, ifc, pu, PU_POWER_LINE, 0, 1)   # 0 = disabled
        got.append(dict(
            ae_mode=_get(dev, ifc, ct, GET_CUR, CT_AE_MODE, 1),
            exposure_s=_get(dev, ifc, ct, GET_CUR, CT_EXPOSURE_ABS, 4) * UNIT_S,
            gain=_get(dev, ifc, pu, GET_CUR, PU_GAIN, 2),
            wb_auto=_get(dev, ifc, pu, GET_CUR, PU_WB_TEMP_AUTO, 1),
            backlight=_get(dev, ifc, pu, GET_CUR, PU_BACKLIGHT, 2),
            power_line=_get(dev, ifc, pu, GET_CUR, PU_POWER_LINE, 1)))
    return got


def set_auto():
    """The camera's own auto exposure. By hand only: no pipeline step calls this."""
    for dev, ifc, ct, pu in _devices():
        _set(dev, ifc, ct, CT_AE_MODE, AE_APERTURE_PRIORITY, 1)


def meter(exposure_s, gain, width=640, height=400, fps=210, rotate180=True, indices=None,
          backend=None, scale=1.0):
    """Live A | B at the given mode and fps, all camera settings manual (`apply`), until q,
    Ctrl-C or a kernel interrupt. For aiming, focusing and lighting: the same frames
    `calib.capture` and a recording get. Per camera, on the image and twice a second in
    text: the grab rate (average since start, and the best 0.5 s window as max), and frame
    mean/max brightness (healthy ~59/82, max 235: CLAUDE.md). ``scale``: display size only.
    The grab rate counts frames the camera thread received, whatever the display keeps up
    with."""
    import time
    import cv2
    import numpy as np
    from controller.camera import identify, sources
    idx = identify.elp_indices(backend=backend) if indices is None else list(indices)
    src = sources.open_stereo([f"camera:{i}" for i in idx], max_skew_s=None, width=width,
                              height=height, fps=fps, grayscale=True, rotate180=rotate180,
                              backend=backend)
    sink = sources.Sink("A | B   (q = quit)").open()
    try:
        for k, r in enumerate(apply(exposure_s, gain)):
            print(f"ELP {k}: {r}")
        # averaged from 1 s in: the first ~1 s after `apply` runs at 170-195 fps on both
        # cameras (measured 2026-11-04) and would drag the mean for minutes
        n0 = t0 = None
        t_open = time.monotonic()
        best, text = [0.0] * len(idx), ["" for _ in idx]
        while True:
            nw, tw, acc = [c.n_grabbed for c in src.sources], time.monotonic(), []
            while time.monotonic() - tw < 0.5:
                acc.append(src.read()[1])
                if sink.due():
                    frames = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in acc[-1]]
                    for f, s in zip(frames, text):
                        cv2.putText(f, s, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                    view = np.hstack(frames)
                    if scale != 1.0:
                        view = cv2.resize(view, None, fx=scale, fy=scale)
                    if sink.show(view) == ord("q"):
                        raise KeyboardInterrupt
            now = time.monotonic()
            if n0 is None:
                if now - t_open >= 1.0:
                    n0, t0 = [c.n_grabbed for c in src.sources], now
                continue
            for k, c in enumerate(src.sources):
                win = (c.n_grabbed - nw[k]) / (now - tw)
                best[k] = max(best[k], win)
                avg = (c.n_grabbed - n0[k]) / (now - t0)
                text[k] = (f"{'AB'[k]}: avg {avg:5.1f} fps  max {best[k]:5.1f}  "
                           f"mean {np.mean([f[k].mean() for f in acc]):5.1f}  "
                           f"max px {max(int(f[k].max()) for f in acc)}")
            print("   ".join(text), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        sink.close()
        src.close()


def _self_check():
    # config header, a VC interface (#3), its header, a camera terminal (ID 1), a PU (ID 2)
    head = (bytes([9, 2, 0, 0, 2, 1, 0, 0x80, 250])
            + bytes([9, 4, 3, 0, 1, 0x0E, 0x01, 0, 0])
            + bytes([13, 0x24, 0x01, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1]))
    camera = bytes([18, 0x24, 0x02, 1, 0x01, 0x02, 0] + [0] * 11)
    pu = bytes([11, 0x24, 0x05, 2, 1, 0, 0, 2, 0xFF, 0x07, 0])
    assert units(head + camera + pu) == (3, 1, 2)
    # a USB-streaming input terminal (0x0101) is not the camera
    streaming = bytes([18, 0x24, 0x02, 1, 0x01, 0x01, 0] + [0] * 11)
    try:
        units(head + streaming + pu)
        raise AssertionError("took a non-camera terminal")
    except OSError:
        pass
    # a CS descriptor after a video STREAMING interface is not a VC unit
    vs = bytes([9, 4, 1, 0, 1, 0x0E, 0x02, 0, 0])
    try:
        units(head[:9] + vs + camera + pu)
        raise AssertionError("parsed units outside the control interface")
    except OSError:
        pass
    assert round(0.002 / UNIT_S) == 20
    print("uvc self-check OK")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _self_check()
    elif sys.argv[1] == "--auto":
        set_auto()
        print("auto exposure on every ELP")
    elif sys.argv[1] == "--meter":
        meter(float(sys.argv[2]) / 1e3, int(sys.argv[3]))
    elif sys.argv[1] == "--set":
        for k, r in enumerate(apply(float(sys.argv[2]) / 1e3, int(sys.argv[3]))):
            print(f"ELP {k}: {r}")
    else:
        sys.exit(__doc__)

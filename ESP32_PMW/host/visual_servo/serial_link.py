"""Serial link to the ESP32 `servo` firmware (src/main_servo.cpp).

Outgoing (we send), '\\n'-terminated ASCII:
    F<hz>          set target frequency
    A<ch>,<duty>   set one carrier duty %  (reserved: attitude)
    S              safe descent (ramp to F_LAND)
    P              ping -> immediate telemetry line
Incoming (firmware emits ~50 Hz):
    T,<millis>,<freqApplied>,<dutyA>,<dutyB>,<dutyC>,<dutyD>

Use as a context manager so loss-of-control always sends 'S' (controlled descent)
on the way out -- the firmware watchdog is the backstop, this is belt-and-braces.
"""
import threading
import time

import serial

from config import SERIAL_PORT, BAUD


class Telemetry:
    __slots__ = ("t_ms", "freq", "duty", "host_t")

    def __init__(self, t_ms=0, freq=0.0, duty=(0, 0, 0, 0), host_t=0.0):
        self.t_ms = t_ms
        self.freq = freq
        self.duty = duty
        self.host_t = host_t

    def __repr__(self):
        return f"Telemetry(t={self.t_ms} f={self.freq:.1f} duty={self.duty})"


class SerialLink:
    def __init__(self, port=SERIAL_PORT, baud=BAUD):
        self.port = port
        self.baud = baud
        self.ser = None
        self._latest = Telemetry()
        self._lock = threading.Lock()
        self._reader = None
        self._stop = threading.Event()

    # -- lifecycle ----------------------------------------------------------
    def open(self):
        self.ser = serial.Serial(self.port, self.baud, timeout=0.05)
        time.sleep(2.0)  # ESP32 resets on port open; wait for boot
        self.ser.reset_input_buffer()
        self._stop.clear()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        return self

    def close(self):
        try:
            self.safe_descent()
            time.sleep(0.05)
        except Exception:
            pass
        self._stop.set()
        if self._reader:
            self._reader.join(timeout=1.0)
        if self.ser and self.ser.is_open:
            self.ser.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    # -- outgoing -----------------------------------------------------------
    def _send(self, line: str):
        if self.ser and self.ser.is_open:
            self.ser.write((line + "\n").encode("ascii"))

    def set_frequency(self, hz: float):
        self._send(f"F{hz:.2f}")

    def set_carrier(self, ch: int, duty: float):
        self._send(f"A{ch},{duty:.1f}")

    def safe_descent(self):
        self._send("S")

    def ping(self):
        self._send("P")

    # -- incoming -----------------------------------------------------------
    def _read_loop(self):
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self.ser.read(128)
            except Exception:
                break
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                self._parse(raw.decode("ascii", "ignore").strip())

    def _parse(self, line: str):
        if not line.startswith("T,"):
            return
        parts = line.split(",")
        if len(parts) != 7:
            return
        try:
            tm = int(parts[1])
            freq = float(parts[2])
            duty = tuple(float(p) for p in parts[3:7])
        except ValueError:
            return
        with self._lock:
            self._latest = Telemetry(tm, freq, duty, time.monotonic())

    def latest(self) -> Telemetry:
        with self._lock:
            return self._latest

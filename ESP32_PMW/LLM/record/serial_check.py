"""Flight startup diagnosis only: reset + receive, never send a drive command.

Run with coil power off. Uses the existing CommandLink/SerialComm owner; see
controller/control/theory.md, Recorded flight START / startup diagnosis.
"""
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import argparse


def wake(port, rom_baud=115200):
    """Read the port clear, reset the ESP32 while listening at the ROM's baud, and
    return to the flight baud: ``{drained, rom_boot, ready}``. Sends no command.

    UNEXPLAINED, 2026-11-05: in a "dead" state every `CommandLink` (app and shell
    alike) read zero bytes for its whole 5 s READY wait, though a raw read of the
    same port at that moment returned ~16 kB of garbage in 1.5 s (~10.9 kB/s, about
    a busy 115200-baud line) and the ESP32, once reset, booted normally and printed
    SYNC_READY. Ruled out on the bench: another process on the port, the second
    macOS node, both cameras streaming, a stale wrong-baud buffer. This exact
    sequence -- unconditional read, EN pulse at 115200, back to the flight baud --
    restored it every time it was tried (3/3); a bare baud switch did not. The
    `CommandLink` that follows resets again; the ESP32 just boots twice.
    """
    import serial
    import time
    from controller.control.link import SerialComm
    s = serial.Serial()
    s.port, s.baudrate, s.timeout, s.dtr, s.rts = port, SerialComm.BAUD, 0.05, False, False
    s.open()
    try:
        drained, t = 0, time.monotonic()
        while time.monotonic() - t < 0.5:
            drained += len(s.read(4096))
        s.baudrate = rom_baud
        s.rts = True                     # EN low, IO0 high (DTR false): a normal boot
        time.sleep(0.15)
        s.rts = False
        rom, t = b'', time.monotonic()
        while time.monotonic() - t < 1.2:
            rom += s.read(4096)
        s.baudrate = SerialComm.BAUD
        app, t = b'', time.monotonic()
        while time.monotonic() - t < 3.0 and b'SYNC_READY' not in app:
            app += s.read(4096)
    finally:
        s.close()
    return dict(drained=drained, rom_boot=b'SPI_FAST_FLASH_BOOT' in rom, ready=b'SYNC_READY' in app)


def check(port, output_root):
    from controller.control.hover_controller_runner import CommandLink
    folder = Path(output_root).resolve() / (datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid4().hex[:6])
    folder.mkdir(parents=True, exist_ok=False)
    print(f'Startup check: {port}; reset then wait up to 5 seconds. No drive commands.', flush=True)
    print(f'Logs: {folder}', flush=True)
    owner = None
    try:
        owner = CommandLink(port, False, str(folder/'host.log'), takeoff_cmd=[],
                            raw_log_path=folder/'telemetry.jsonl', sync_start=True)
        print('PASS: received SYNC_READY v=1. No ramp sent.', flush=True)
        return 0
    except (ValueError, OSError) as exc:
        print(f'FAILED: {exc}', flush=True)
        return 1
    finally:
        if owner is not None:
            owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--output-root', default='LLM/runs/serial_checks')
    args = parser.parse_args()
    return check(args.port, args.output_root)


if __name__ == '__main__':
    raise SystemExit(main())

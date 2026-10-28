"""Flight startup diagnosis only: reset + receive, never send a drive command.

Run with coil power off. Uses the existing CommandLink/SerialComm owner; see
controller/control/theory.md, Recorded flight START / startup diagnosis.
"""
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import argparse


def check(port, output_root):
    from controller.control.hover_controller_runner import CommandLink
    folder = Path(output_root).resolve() / (datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid4().hex[:6])
    folder.mkdir(parents=True, exist_ok=False)
    print(f'Startup check: {port}; reset then wait up to 5 seconds. No pattern commands.', flush=True)
    print(f'Logs: {folder}', flush=True)
    owner = None
    try:
        owner = CommandLink(port, False, str(folder/'host.log'), takeoff_cmd=[],
                            raw_log_path=folder/'telemetry.jsonl', sync_start=True)
        print('PASS: received SYNC_READY v=1. No pattern sent.', flush=True)
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

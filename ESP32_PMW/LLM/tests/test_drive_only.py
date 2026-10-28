"""No hardware: command identity and stop/close on each failure path."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from LLM import drive_only
from controller.control import ramp


class DriveOnlyTests(unittest.TestCase):
    def test_real_link_ready_without_fabricated_capture(self):
        from LLM.tests.test_start import Port
        from controller.control.hover_controller_runner import CommandLink
        for ready in (True, False):
            with self.subTest(ready=ready), tempfile.TemporaryDirectory() as tmp:
                now = [0.0]
                emitted = [False]
                port = Port()
                def sleep(seconds):
                    now[0] += seconds
                    if ready and now[0] >= 3 and not emitted[0]:
                        port.chunks.append(b'SYNC_READY v=1\n')
                        emitted[0] = True
                with patch('controller.control.link.serial.Serial', return_value=port), \
                        patch.object(drive_only.time, 'monotonic', side_effect=lambda: now[0]), \
                        patch.object(drive_only.time, 'sleep', side_effect=sleep):
                    def create():
                        return CommandLink('fake', False, str(Path(tmp)/'commands.log'),
                                           raw_log_path=Path(tmp)/'rx.jsonl', require_ready=True)
                    if not ready:
                        with self.assertRaises(ValueError): create()
                        self.assertFalse(port.sent)
                        self.assertTrue(port.closed)
                    else:
                        owner = create()
                        self.assertTrue(owner.sync_ready)
                        self.assertIsNone(owner.capture_first)
                        # Capture no actual drive: replace send after exercising real handshake.
                        with patch.object(owner, 'send') as send:
                            owner.send_takeoff()
                            self.assertEqual(send.call_args.args, ('seq=go',))
                        owner.close()

    def exercise(self, mode):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pattern = dict(schema_version=1, name='test', trigger='time',
                           frequency_segments=[[2, 10, 3, 1, 2]], channel_updates=[],
                           channel_settings_status='firmware_defaults_unrecorded')
            (root/'input.json').write_text(json.dumps(pattern))
            now = [0.0]
            commands = []
            closed = []
            class Link:
                def __init__(self, *args, **kw):
                    assert kw['require_ready'] and not kw.get('sync_start', False)
                    self.lines = kw['takeoff_cmd']
                    self.on_line = None
                    if mode == 'no_ready':
                        raise ValueError('No READY')
                def send_takeoff(self):
                    for c in self.lines:
                        commands.append(c)
                    if mode == 'reject':
                        self.on_line('!seq rejected')
                    if mode != 'no_ack':
                        self.on_line('SYNC_START v=1 led0=1 led1=2 drive0=3 drive1=4')
                def drain(self):
                    if mode == 'interrupt' and 'stop' not in commands:
                        raise KeyboardInterrupt()
                def send(self, c):
                    commands.append(c)
                    if mode == 'stop_failure':
                        raise OSError('Disconnected')
                    self.on_line('state=4 col=100')
                def close(self):
                    closed.append(True)
            def sleep(seconds):
                now[0] += seconds
            with patch.object(drive_only, 'CommandLink', Link), \
                    patch('controller.control.hover_controller_runner._thermal_gate',
                          side_effect=AssertionError('LLM must not wait on thermal model')) as gate, \
                    patch.object(drive_only.time, 'monotonic', side_effect=lambda: now[0]), \
                    patch.object(drive_only.time, 'sleep', side_effect=sleep):
                if mode in ('ok', 'stop_failure'):
                    drive_only.run_frequency_pattern(root/'input.json', 'fake', output_root=root/'out')
                else:
                    with self.assertRaises((RuntimeError, ValueError, KeyboardInterrupt)):
                        drive_only.run_frequency_pattern(root/'input.json', 'fake', output_root=root/'out')
                gate.assert_not_called()
            summary = json.loads(next((root/'out').glob('*/result.json')).read_text())
            if mode == 'no_ready':
                self.assertEqual(commands, [])
            else:
                self.assertEqual(commands, ramp.seq_lines(pattern['frequency_segments']) + ['stop'])
                self.assertEqual(closed, [True])
            if mode == 'ok':
                self.assertGreaterEqual(now[0], 3)
                self.assertTrue(summary['start_acknowledged'] and summary['stop_acknowledged'])
            if mode == 'no_ack':
                self.assertLess(now[0], 3)
            if mode == 'stop_failure':
                self.assertFalse(summary['stop_sent'])
                self.assertIn('stop_error', summary)
            self.assertFalse(summary['video_recorded'])

    def test_normal(self): self.exercise('ok')
    def test_no_ready_sends_nothing(self): self.exercise('no_ready')
    def test_no_start_ack_stops_early(self): self.exercise('no_ack')
    def test_firmware_rejection_stops(self): self.exercise('reject')
    def test_interrupt_stops(self): self.exercise('interrupt')
    def test_failed_stop_still_closes(self): self.exercise('stop_failure')

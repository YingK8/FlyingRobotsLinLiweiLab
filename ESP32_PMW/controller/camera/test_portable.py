"""Offline regressions for platform dispatch, paired ownership and recording.

Run: python -m controller.camera.test_portable. Never opens cameras or serial.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from controller.camera import sources, identify, elp, record, session


class PortableTests(unittest.TestCase):
    def test_backend_dispatch(self):
        for system, expected in [('win32', cv2.CAP_DSHOW), ('darwin', cv2.CAP_AVFOUNDATION),
                                 ('linux', cv2.CAP_V4L2), ('unknown', cv2.CAP_ANY)]:
            with patch.object(sources.sys, 'platform', system):
                self.assertEqual(sources.resolve_backend('auto'), expected)
        self.assertEqual(sources.resolve_backend('msmf'), cv2.CAP_MSMF)
        with self.assertRaises(ValueError):
            sources.resolve_backend('typo')

    def test_windows_identity_and_actual_read_probe(self):
        good, empty = Mock(), Mock()
        good.read.return_value = (1.0, np.zeros((800,1280),np.uint8))
        empty.read.return_value = None
        with patch.object(identify.sys, 'platform', 'win32'), \
             patch.object(identify.subprocess, 'run') as command, \
             patch.object(sources, 'MonoCamera', side_effect=[good, empty, OSError('absent')]):
            self.assertEqual(identify.connected(), [])
            self.assertEqual(identify.elp_indices(n=1, max_index=3, backend='msmf'), [0])
            command.assert_not_called()
        good.close.assert_called_once()
        empty.close.assert_called_once()

    def test_second_open_failure_releases_first(self):
        first = Mock()
        with patch.object(sources, 'open_source', side_effect=[first, OSError('B failed')]):
            with self.assertRaises(OSError):
                sources.open_stereo(['camera:0','camera:1'])
        first.close.assert_called_once()
        first.reset_mock()
        with patch.object(elp, 'open_elp', side_effect=[first, OSError('B failed')]):
            with self.assertRaises(OSError):
                elp.open_group([0,1])
        first.close.assert_called_once()
        with patch.object(sources, 'open_source') as opening:
            with self.assertRaises(ValueError):
                sources.open_stereo(['camera:0','camera:0'])
            opening.assert_not_called()

    def test_codec_fallback_is_same_for_both_streams(self):
        a, b, c, d = [Mock() for _ in range(4)]
        a.isOpened.return_value = True
        b.isOpened.return_value = False
        c.isOpened.return_value = d.isOpened.return_value = True
        with tempfile.TemporaryDirectory() as td:
            with patch.object(record.sys, 'platform', 'darwin'), \
                 patch.object(record.cv2, 'VideoWriter', side_effect=[a,b,c,d]):
                writer = record.FlightWriter(td)
                try:
                    frame = np.zeros((8,10), np.uint8)
                    writer.add(0, [frame,frame], (0,.001), .001)
                finally:
                    folder = writer.close()
                self.assertEqual(writer.codec, 'mp4v')
                self.assertEqual(json.loads((folder/'meta.json').read_text())['encoder_finalized'], True)
                writer.close()
        for w in [a,b,c,d]:
            w.release.assert_called_once()

    def test_late_encoder_error_is_saved_after_drain(self):
        faulty, other = Mock(), Mock()
        faulty.isOpened.return_value = other.isOpened.return_value = True
        faulty.write.side_effect = OSError('simulated encoder failure')
        with tempfile.TemporaryDirectory() as td, \
             patch.object(record.cv2, 'VideoWriter', side_effect=[faulty, other]):
            writer = record.FlightWriter(td, codec='mp4v')
            f = np.zeros((8,10),np.uint8)
            writer.add(0, [f,f], (0,.001), .001)
            folder = writer.close()
            meta = json.loads((folder/'meta.json').read_text())
            self.assertEqual(meta['encoder_errors'], 1)
            self.assertTrue(meta['encoder_finalized'])

    def test_recorder_forwards_fps_backend_and_closes_on_shape_error(self):
        camera = Mock(actual={'width':80, 'height':64, 'fps':30}, n_grabbed=1, n_dropped=0)
        src = Mock(sources=[camera,camera], last_stamps=(0,.001), last_skew=.001)
        src.read.return_value = (0, [np.zeros((32,40),np.uint8)]*2)
        src.skew_stats.return_value = {}
        with tempfile.TemporaryDirectory() as td, \
             patch.object(sources, 'open_stereo', return_value=src) as opening:
            with self.assertRaises(OSError):
                record.record(td, indices=[0,1], width=80, height=64, fps=30,
                              backend='msmf', preview=False, start=True, seconds=1)
            self.assertEqual(opening.call_args.kwargs['fps'],30)
            self.assertEqual(opening.call_args.kwargs['backend'],'msmf')
            src.close.assert_called_once()
            meta = json.loads(next(Path(td).glob('*/meta.json')).read_text())
            self.assertIn('capture_error',meta)

    def test_interrupt_finalises_and_returns_take(self):
        camera = Mock(actual={'width':80,'height':64,'fps':30}, n_grabbed=2,n_dropped=0)
        f=np.zeros((64,80),np.uint8)
        src=Mock(sources=[camera,camera],last_stamps=(1,1.001),last_skew=.001)
        src.read.side_effect=[(1,[f,f]), KeyboardInterrupt()]
        src.skew_stats.return_value={}
        with tempfile.TemporaryDirectory() as td, patch.object(sources,'open_stereo',return_value=src):
            done=record.record(td,indices=[0,1],width=80,height=64,fps=30,
                               preview=False,start=True,seconds=10,codec='mp4v')
            self.assertEqual(len(done),1)
            meta=json.loads((done[0]/'meta.json').read_text())
            self.assertEqual(meta['n_frames'],1)
            self.assertTrue(meta['encoder_finalized'])
            self.assertNotIn('capture_error',meta)
        src.close.assert_called_once()

    def test_two_real_synthetic_clips_round_trip(self):
        for codec in ['auto','mp4v']:
            result=session.check_environment(codec)
            self.assertIn('PASS',result['two_clip_encode_decode'])


if __name__ == '__main__':
    unittest.main(verbosity=2)

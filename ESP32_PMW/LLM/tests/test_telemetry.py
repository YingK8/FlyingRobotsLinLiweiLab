import json
from pathlib import Path
import tempfile
import unittest

from LLM.telemetry import parse_log


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "telemetry.jsonl"

    @staticmethod
    def line(stamp="1000", current="A=1.00 B=2.00 C=3.00 D=4.00", frequency="2.0"):
        return f"t={stamp} freq={frequency} | I[A]: {current} | duty[%]: A=20 B=20 C=20 D=20"

    def parse(self, lines, hosts=None):
        hosts = hosts if hosts is not None else list(range(len(lines)))
        text = "".join(json.dumps({"host_receive_s": h, "line": line}) + "\n"
                       for h, line in zip(hosts, lines))
        self.path.write_text(text)
        result = parse_log(self.path)
        self.assertEqual(self.path.read_text(), text)
        return result

    def test_valid_reordered_channels_complete_exponents_and_equal_receives(self):
        result = self.parse(["START device_ms=1000", self.line(current="D=4 C=-.3 B=2e-1 A=+1.5"),
                             self.line(stamp="1500")], [9, 9, 9])
        self.assertEqual(result.t_device_s.tolist(), [1, 1.5])
        self.assertEqual(result.current_A_A.tolist(), [1.5, 1])
        self.assertEqual(result.current_B_A.tolist(), [.2, 2])
        self.assertEqual(result.current_C_A.tolist(), [-.3, 3])
        self.assertEqual(result.raw_line_number.tolist(), [2, 3])
        self.assertEqual(result.attrs["parse_errors"], [])

    def test_corrupted_numeric_tokens_missing_sections_and_duplicate_labels(self):
        bad = [self.line(current="A=1e3 B=2x C=3.4oops D=4.5"),
               self.line(current=""), self.line(current="A=1 A=2 C=3 D=4"),
               self.line(current="A=1 B=2 C=3"), self.line(current="A=1 B=2 C=3 D=4 E=5"),
               self.line(current="A=1e999 B=2 C=3 D=4"), self.line(current="A=nan B=2 C=3 D=4"),
               self.line(frequency="2oops"), "t=1000 I[A]:| duty[%]: A=1 B=1 C=1 D=1"]
        result = self.parse(bad + [self.line(stamp="2000")])
        self.assertEqual(len(result), 1)
        self.assertEqual([e["line"] for e in result.attrs["parse_errors"]], list(range(1, len(bad) + 1)))
        self.assertTrue(all(e["reason"] for e in result.attrs["parse_errors"]))

    def test_device_timestamp_must_be_complete_integer_token(self):
        bad = [self.line(stamp=stamp) for stamp in ["1000.0", "1e3", "1000x", "-1000", "+1000", ""]]
        result = self.parse(bad + [self.line()])
        self.assertEqual(result.t_device_s.tolist(), [1.0])
        self.assertEqual(len(result.attrs["parse_errors"]), len(bad))

    def test_clock_discontinuity_fails_and_normal_lines_participate_in_host_order(self):
        for stamps in [("1000", "1000"), ("1000", "1")]:
            with self.subTest(stamps=stamps), self.assertRaisesRegex(ValueError, "device time"):
                self.parse([self.line(stamp=s) for s in stamps])
        with self.assertRaisesRegex(ValueError, "host receive time decreased"):
            self.parse(["boot text", self.line()], [5, 4])

    def test_malformed_json_and_receive_fields_report_errors(self):
        lines = ["{", json.dumps({"line": self.line(), "host_receive_s": None}),
                 json.dumps({"line": self.line(), "host_receive_s": float("nan")}),
                 json.dumps({"line": self.line(), "host_receive_s": True}),
                 json.dumps({"line": 3, "host_receive_s": 1})]
        self.path.write_text("\n".join(lines))
        result = parse_log(self.path)
        self.assertTrue(result.empty)
        self.assertEqual(len(result.attrs["parse_errors"]), len(lines))


def run():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TelemetryTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    run()

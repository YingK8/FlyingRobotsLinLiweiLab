"""Compact writer for JsonPhaseSequencer experiment files.

Matches the hand-authored spiffs_data/*.json style: config keys indented,
scalar arrays inline (e.g. "initial_duty": [50, 50, 50, 50]), and one schedule
step per line (e.g. { "method": "addWaitTask", "duration_ms": 500 }). Keeping
each step on its own line is compact yet still diff- and grep-friendly.
"""
import json


def _fmt_step(step):
    # json.dumps already inlines scalar arrays ([0, 3]); just add brace padding.
    return "{ " + json.dumps(step)[1:-1].strip() + " }"


def dumps_experiment(obj):
    """Return the object-form document as a compact, parsable string."""
    lines = ["{"]
    keys = list(obj)
    for i, k in enumerate(keys):
        tail = "," if i < len(keys) - 1 else ""
        if k == "schedule":
            steps = obj[k]
            lines.append('  "schedule": [')
            for j, s in enumerate(steps):
                sc = "," if j < len(steps) - 1 else ""
                lines.append("    " + _fmt_step(s) + sc)
            lines.append("  ]" + tail)
        else:
            lines.append(f"  {json.dumps(k)}: {json.dumps(obj[k])}" + tail)
    lines.append("}")
    return "\n".join(lines)


def write_experiment(obj, path):
    """Write the object-form document to `path` in the compact style."""
    with open(path, "w") as f:
        f.write(dumps_experiment(obj) + "\n")

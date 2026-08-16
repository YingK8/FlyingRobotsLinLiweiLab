"""Compact writer for JsonPhaseSequencer files: config keys indented, scalar
arrays inline, one schedule step per line."""
import json


def _fmt_step(step):
    # json.dumps already inlines scalar arrays; just pad the braces.
    return "{ " + json.dumps(step)[1:-1].strip() + " }"


def dumps_experiment(obj):
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
    with open(path, "w") as f:
        f.write(dumps_experiment(obj) + "\n")

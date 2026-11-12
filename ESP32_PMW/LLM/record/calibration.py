"""Continue a session's ChArUco capture in separate rounds and fit their union."""
import csv
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from LLM.analyse.rig import same_rotation
from LLM.record.io import read_json, write_json


def _check_bag(bench, bag):
    meta = read_json(bag / "meta.json")
    camera = bench.camera
    if (meta["board"] != bench.board.summary()
            or meta["camera_indices"] != list(camera["indices"])
            or meta["mode"] != [camera["width"], camera["height"]]
            or not same_rotation(meta["rotate180"], camera["rotate180"])):
        raise ValueError(f"Board or camera setup differs from {bag}")
    return meta


def _combine(bench, bags, output):
    """Build a new snapshot, renumbering both views and the corresponding CSV together."""
    for tag in "AB":
        (output / tag).mkdir(parents=True, exist_ok=False)
    pair_count = 0
    solo_counts = [0, 0]
    camera_ids = []
    with (output / "pairs.csv").open("x", newline="") as combined:
        writer = csv.writer(combined)
        header = None
        for bag in bags:
            meta = _check_bag(bench, bag)
            if meta.get("elp_ids"):
                if camera_ids and camera_ids != meta["elp_ids"]:
                    raise ValueError(f"Camera devices differ in {bag}")
                camera_ids = meta["elp_ids"]
            with (bag / "pairs.csv").open(newline="") as source:
                reader = csv.reader(line for line in source if not line.startswith("#"))
                fields = next(reader)
                if header is None:
                    header = fields
                    writer.writerow(header)
                if fields != header or fields[0] != "index":
                    raise ValueError(f"Incompatible pairs.csv in {bag}")
                rows = list(reader)

            names = [f"pair_{int(row[0]):03d}.png" for row in rows]
            if (len(set(names)) != len(rows) or len(rows) != meta["n_pairs"]
                    or any({p.name for p in (bag / tag).glob("pair_*.png")} != set(names)
                           for tag in "AB")):
                raise ValueError(f"Incomplete pair images or timing rows in {bag}")
            for name, row in zip(names, rows):
                for tag in "AB":
                    shutil.copyfile(bag / tag / name, output / tag / f"pair_{pair_count:03d}.png")
                writer.writerow([pair_count, *row[1:]])
                pair_count += 1

            for k, tag in enumerate("AB"):
                photos = sorted((bag / tag).glob(f"solo_{tag}_*.png"))
                if len(photos) != meta["n_solo"][k]:
                    raise ValueError(f"Incomplete camera {tag} solo frames in {bag}")
                for photo in photos:
                    shutil.copyfile(photo, output / tag / f"solo_{tag}_{solo_counts[k]:03d}.png")
                    solo_counts[k] += 1

    # Each source retains its own clock/USB statistics and original creation time.
    write_json(output / "meta.json", dict(meta, n_pairs=pair_count, n_solo=solo_counts,
               elp_ids=camera_ids, skew={}, source_bags=[str(b) for b in bags]))
    return pair_count


def collect_appended_frames(bench, *, seed=None):
    """Collect one new round, then return all completed rounds of this session plus seed."""
    group = bench.folder / "calibration_append"
    group.mkdir(exist_ok=True)
    sources = []
    if seed is not None:
        seed = Path(seed)
        if seed.parent.parent.resolve() != bench.folder.resolve():
            raise ValueError("Resume in the original bench session")
        sources.append(seed)

    def finished_rounds():
        return [p.parent for p in sorted(group.glob("round_*/images/meta.json"))]

    previous = sources + finished_rounds()
    previous_count = sum(_check_bag(bench, bag)["n_pairs"] for bag in previous)
    print(f"已保留 {previous_count} 对。预览中的 saved 只计本轮；停止后合并标定。", flush=True)
    bench.calibration = bench.verification = bench.world = None
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:8]
    new_bag = group / f"round_{run_id}" / "images"
    bench.collect_charuco_frames(new_bag)
    # A failed fit must not discard the new round. The next call finds it by meta.json.
    bags = sources + finished_rounds()
    output = group / f"fit_{run_id}" / "images"
    total = _combine(bench, bags, output)
    print(f"本轮新增 {read_json(new_bag / 'meta.json')['n_pairs']} 对，合计 {total} 对用于筛选和标定。")
    return output

"""Extract deployment-available instance confidence from sealed round-2 LAS files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import laspy
import numpy as np
import pandas as pd


ARMS = ["control", "acpe", "epco", "acpe_epco"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for arm in ARMS:
        arm_root = args.prediction_root / "predictions" / arm
        for result_path in sorted(arm_root.glob("*/result.json")):
            record = json.loads(result_path.read_text(encoding="utf-8"))
            scene = str(record["scene"])
            las_rel = next(x["relative_path"] for x in record["artifacts"] if x["relative_path"].startswith("round_2/") and x["relative_path"].endswith(".las"))
            las_path = result_path.parent / "artifacts" / las_rel
            stats: dict[int, list[float]] = {}
            with laspy.open(las_path) as reader:
                dims = set(reader.header.point_format.dimension_names)
                if not {"instance_pred", "score"}.issubset(dims):
                    raise RuntimeError(f"missing sealed confidence dimensions: {las_path}")
                for points in reader.chunk_iterator(2_000_000):
                    inst = np.asarray(points.instance_pred, dtype=np.int64)
                    score = np.asarray(points.score, dtype=np.float64)
                    for instance_id in np.unique(inst[inst >= 0]):
                        values = score[inst == instance_id]
                        item = stats.setdefault(int(instance_id), [0.0, 0.0, np.inf, -np.inf])
                        item[0] += float(values.sum())
                        item[1] += int(len(values))
                        item[2] = min(item[2], float(values.min()))
                        item[3] = max(item[3], float(values.max()))
            scene_rows = []
            for instance_id, (total, count, lo, hi) in stats.items():
                scene_rows.append({"arm": arm, "scene": scene, "las_instance_id": instance_id, "las_point_count": int(count), "instance_confidence": total / count, "confidence_within_instance_range": hi - lo})
            frame = pd.DataFrame(scene_rows).sort_values("las_instance_id")
            med = float(frame.instance_confidence.median())
            mad = float(np.median(np.abs(frame.instance_confidence - med)))
            frame["confidence_scene_robust_z"] = (frame.instance_confidence - med) / max(1e-6, 1.4826 * mad)
            frame["confidence_scene_percentile"] = frame.instance_confidence.rank(method="average", pct=True)
            ordered = np.sort(frame.instance_confidence.to_numpy(float))
            frame["confidence_gap_to_lower"] = [float(x - ordered[max(0, np.searchsorted(ordered, x, side="left") - 1)]) for x in frame.instance_confidence]
            frame["confidence_gap_to_higher"] = [float(ordered[min(len(ordered)-1, np.searchsorted(ordered, x, side="right"))] - x) for x in frame.instance_confidence]
            rows.extend(frame.to_dict(orient="records"))
    out = pd.DataFrame(rows)
    # Match the frozen official28 schema adapter exactly: LAS labels are
    # zero-based while the evaluation ledger uses one-based prediction IDs.
    out["pred_instance_id"] = out["las_instance_id"].astype(int) + 1
    if out.duplicated(["arm", "scene", "las_instance_id"]).any():
        raise RuntimeError("duplicate instance confidence keys")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(json.dumps({"rows": len(out), "arms": sorted(out.arm.unique()), "scenes_per_arm": out.groupby("arm").scene.nunique().to_dict(), "max_within_instance_score_range": float(out.confidence_within_instance_range.max())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

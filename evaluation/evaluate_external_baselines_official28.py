#!/usr/bin/env python3
"""Evaluate sealed external baselines on the frozen Official28 truth set.

This adapter is deliberately model-specific only at the file-loading boundary.
All methods then use the same pointwise semantic definitions and the same
Hungarian IoU>0.5 instance evaluator.
"""

import os
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from plyfile import PlyData


MAX_AXIS_RESIDUAL_M = 0.000501
MATCH_IOU = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ply(path: Path) -> dict[str, np.ndarray]:
    vertex = PlyData.read(str(path))["vertex"].data
    return {name: np.asarray(vertex[name]) for name in vertex.dtype.names}


def read_truth(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = read_ply(path)
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
    semantic = np.asarray(data["semantic_seg"]) != 1  # frozen: class 1 is ground
    instance = np.asarray(data["treeID"], dtype=np.int64)
    instance = np.where(semantic & (instance > 0), instance, 0)
    return xyz, semantic, instance


def stable_align(
    truth_xyz: np.ndarray,
    prediction_xyz: np.ndarray,
    semantic: np.ndarray,
    instance: np.ndarray,
    *,
    las_truncation_grid: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if truth_xyz.shape != prediction_xyz.shape:
        raise ValueError(f"point-count mismatch: {truth_xyz.shape} vs {prediction_xyz.shape}")
    if semantic.shape != (len(truth_xyz),) or instance.shape != (len(truth_xyz),):
        raise ValueError("prediction payload is not point-aligned")
    # LAS writers quantize at 1 mm.  Sorting the raw float coordinates can
    # interleave points differently after that quantization, even though the
    # coordinate multisets are identical.  Align on the frozen millimetre grid
    # and use original row order only as a deterministic duplicate tie-breaker.
    if las_truncation_grid:
        # The wrapper first writes normalized float32 PLY and then assigns LAS
        # integer coordinates directly.  That two-stage truncation can move a
        # point by up to 1 mm per axis.  Recover a strict one-to-one nearest
        # assignment, resolving only the small collision subproblem globally.
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial import cKDTree, distance_matrix

        nearest_distance, nearest = cKDTree(truth_xyz).query(
            prediction_xyz, k=1, workers=-1
        )
        counts = np.bincount(nearest, minlength=len(truth_xyz))
        conflict_pred = np.flatnonzero(counts[nearest] > 1)
        pool_truth = np.flatnonzero(counts != 1)
        if len(conflict_pred) != len(pool_truth):
            raise ValueError("nearest-neighbour collision accounting failed")
        mapping = nearest.copy()
        if len(conflict_pred):
            cost = distance_matrix(
                prediction_xyz[conflict_pred], truth_xyz[pool_truth]
            )
            rows, cols = linear_sum_assignment(cost)
            mapping[conflict_pred[rows]] = pool_truth[cols]
        if len(np.unique(mapping)) != len(mapping):
            raise ValueError("coordinate alignment is not one-to-one")
        residual = truth_xyz[mapping] - prediction_xyz
        max_axis = float(np.max(np.abs(residual), initial=0.0))
        max_l2 = float(np.max(np.linalg.norm(residual, axis=1), initial=0.0))
        # A collision component can require a second adjacent grid step to
        # preserve a one-to-one mapping.  Fail above 2.5 mm Euclidean distance;
        # the observed collision count and maximum residual are retained per
        # scene for audit rather than hidden by the adapter.
        if max_axis > 0.002501 or max_l2 > 0.003468:
            raise ValueError(
                "two-stage LAS coordinate residual exceeds contract: "
                f"axis={max_axis:g} m, l2={max_l2:g} m"
            )
        semantic_aligned = np.empty(len(semantic), dtype=bool)
        instance_aligned = np.empty(len(instance), dtype=np.int64)
        semantic_aligned[mapping] = np.asarray(semantic, dtype=bool)
        instance_aligned[mapping] = np.asarray(instance, dtype=np.int64)
        return semantic_aligned, instance_aligned, {
            "max_abs_axis_residual_m": max_axis,
            "max_l2_residual_m": max_l2,
            "nearest_collision_rows": int(len(conflict_pred)),
        }

    truth_mm = np.rint(truth_xyz * 1000.0).astype(np.int64)
    pred_mm = np.rint(prediction_xyz * 1000.0).astype(np.int64)
    truth_order = np.lexsort(
        (np.arange(len(truth_mm)), truth_mm[:, 2], truth_mm[:, 1], truth_mm[:, 0])
    )
    pred_order = np.lexsort(
        (np.arange(len(pred_mm)), pred_mm[:, 2], pred_mm[:, 1], pred_mm[:, 0])
    )
    if not np.array_equal(truth_mm[truth_order], pred_mm[pred_order]):
        mismatch = int(
            np.count_nonzero(np.any(truth_mm[truth_order] != pred_mm[pred_order], axis=1))
        )
        raise ValueError(f"millimetre coordinate multiset mismatch: {mismatch} rows")
    residual = truth_xyz[truth_order] - prediction_xyz[pred_order]
    max_axis = float(np.max(np.abs(residual), initial=0.0))
    max_l2 = float(np.max(np.linalg.norm(residual, axis=1), initial=0.0))
    if max_axis > MAX_AXIS_RESIDUAL_M:
        raise ValueError(f"coordinate mismatch: max axis residual {max_axis:g} m")
    inverse_truth = np.empty(len(truth_order), dtype=np.int64)
    inverse_truth[truth_order] = np.arange(len(truth_order))
    semantic_aligned = np.asarray(semantic, dtype=bool)[pred_order][inverse_truth]
    instance_aligned = np.asarray(instance, dtype=np.int64)[pred_order][inverse_truth]
    return semantic_aligned, instance_aligned, {
        "max_abs_axis_residual_m": max_axis,
        "max_l2_residual_m": max_l2,
    }


def read_las_prediction(path: Path, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cloud = laspy.read(str(path))
    xyz = np.column_stack([cloud.x, cloud.y, cloud.z]).astype(np.float64)
    if method == "chm_watershed":
        instance = np.asarray(cloud["treeID"], dtype=np.int64)
        semantic = instance > 0
    elif method == "segmentanytree":
        instance = np.asarray(cloud["PredInstance"], dtype=np.int64)
        semantic = np.asarray(cloud["PredSemantic"], dtype=np.int64) == 1
        instance = np.where(semantic & (instance > 0), instance, 0)
    else:
        raise ValueError(method)
    return xyz, semantic, instance


def read_treelearn(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        xyz = np.asarray(data["points"], dtype=np.float64)
        instance = np.rint(data["labels"]).astype(np.int64)
    instance = np.where(instance > 0, instance, 0)
    return xyz, instance > 0, instance


def read_forainet(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = read_ply(path)
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
    raw = np.asarray(data["preds"], dtype=np.int64)
    instance = np.where(raw >= 0, raw + 1, 0)
    return xyz, instance > 0, instance


def read_segmentanytree_aligned(
    path: Path, truthfree_input: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read sealed r2 labels recovered into the original truth-free row order."""
    xyz_data = read_ply(truthfree_input)
    xyz = np.column_stack([xyz_data["x"], xyz_data["y"], xyz_data["z"]]).astype(np.float64)
    with np.load(path) as payload:
        semantic = np.asarray(payload["semantic"], dtype=np.uint8) == 1
        instance = np.asarray(payload["instance"], dtype=np.int64)
    if len(xyz) != len(semantic) or len(xyz) != len(instance):
        raise ValueError(f"SegmentAnyTree aligned payload length mismatch: {path}")
    instance = np.where(semantic & (instance > 0), instance, 0)
    return xyz, semantic, instance


def instance_iou_matrix(truth: np.ndarray, prediction: np.ndarray):
    truth_ids = np.unique(truth[truth > 0])
    pred_ids = np.unique(prediction[prediction > 0])
    intersections = np.zeros((len(truth_ids), len(pred_ids)), dtype=np.int64)
    if len(truth_ids) and len(pred_ids):
        joint = (truth > 0) & (prediction > 0)
        ti = np.searchsorted(truth_ids, truth[joint])
        pi = np.searchsorted(pred_ids, prediction[joint])
        intersections = np.bincount(
            ti * len(pred_ids) + pi,
            minlength=len(truth_ids) * len(pred_ids),
        ).reshape(len(truth_ids), len(pred_ids))
    truth_sizes = np.bincount(np.searchsorted(truth_ids, truth[truth > 0]), minlength=len(truth_ids))
    pred_sizes = np.bincount(np.searchsorted(pred_ids, prediction[prediction > 0]), minlength=len(pred_ids))
    union = truth_sizes[:, None] + pred_sizes[None, :] - intersections
    iou = np.divide(intersections, union, out=np.zeros_like(union, dtype=float), where=union > 0)
    return truth_ids, pred_ids, iou


def evaluate_scene(
    semantic_gt: np.ndarray,
    instance_gt: np.ndarray,
    semantic_pred: np.ndarray,
    instance_pred: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from scipy.optimize import linear_sum_assignment

    sem_tp = int(np.count_nonzero(semantic_gt & semantic_pred))
    sem_fn = int(np.count_nonzero(semantic_gt & ~semantic_pred))
    sem_fp = int(np.count_nonzero(~semantic_gt & semantic_pred))
    truth_ids, pred_ids, iou = instance_iou_matrix(instance_gt, instance_pred)
    if len(truth_ids) and len(pred_ids):
        rows, cols = linear_sum_assignment(-iou)
        keep = iou[rows, cols] > MATCH_IOU
        rows, cols = rows[keep], cols[keep]
    else:
        rows = cols = np.empty(0, dtype=np.int64)
    tp = int(len(rows)); fn = int(len(truth_ids) - tp); fp = int(len(pred_ids) - tp)
    iou_sum = float(iou[rows, cols].sum()) if tp else 0.0
    matches = [
        {"truth_id": int(truth_ids[r]), "prediction_id": int(pred_ids[c]), "iou": float(iou[r, c])}
        for r, c in zip(rows, cols)
    ]
    return {
        "points": int(len(instance_gt)),
        "semantic_tp": sem_tp, "semantic_fp": sem_fp, "semantic_fn": sem_fn,
        "truth_instances": int(len(truth_ids)), "prediction_instances": int(len(pred_ids)),
        "tp": tp, "fp": fp, "fn": fn, "matched_iou_sum": iou_sum,
    }, matches


def ratios(counts: dict[str, Any]) -> dict[str, float]:
    def div(a, b):
        return float(a / b) if b else float("nan")
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    stp, sfp, sfn = counts["semantic_tp"], counts["semantic_fp"], counts["semantic_fn"]
    precision = div(tp, tp + fp); recall = div(tp, tp + fn)
    return {
        "instance_precision": precision,
        "instance_recall": recall,
        "instance_f1": div(2 * precision * recall, precision + recall),
        "matched_iou_mean": div(counts["matched_iou_sum"], tp),
        "panoptic_quality": div(counts["matched_iou_sum"], tp + 0.5 * fp + 0.5 * fn),
        "semantic_iou": div(stp, stp + sfp + sfn),
        "semantic_precision": div(stp, stp + sfp),
        "semantic_recall": div(stp, stp + sfn),
    }


COUNT_FIELDS = (
    "points", "semantic_tp", "semantic_fp", "semantic_fn", "truth_instances",
    "prediction_instances", "tp", "fp", "fn", "matched_iou_sum",
)


def sum_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {key: sum(row[key] for row in rows) for key in COUNT_FIELDS}


def locate_prediction(method: str, scene: str, base: Path, forainet_index: dict[str, int]) -> Path:
    if method == "chm_watershed":
        return base / "chm_predictions" / f"{scene}.laz"
    if method == "treelearn":
        return base / "treelearn_inputs" / scene / "results_official28_frozen" / "full_forest" / f"{scene}.npz"
    if method == "forainet":
        return base / "forainet_official28_truthfree_run_r2" / f"Instance_Results_forEval_{forainet_index[scene]}.ply"
    if method == "segmentanytree":
        return base / "segmentanytree_official28_sealed_r2_aligned" / f"{scene}.npz"
    raise ValueError(method)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--truth-inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inventory_doc = json.loads(args.truth_inventory.read_text())
    inventory = inventory_doc["truth_inventory"]
    fseal_path = args.base / "forainet_official28_truthfree_run_r2" / "FORAINET_OFFICIAL28_TRUTHFREE_SEAL.json"
    fseal = json.loads(fseal_path.read_text())
    forainet_index = {Path(row["source_name"]).stem: int(row["index"]) for row in fseal["scenes"]}
    methods = ("chm_watershed", "treelearn", "forainet", "segmentanytree")
    scene_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []

    for method in methods:
        for truth_row in inventory:
            scene = truth_row["scene"]
            source = truth_row["physical_source"]
            truth_path = Path(truth_row["declared_truth_path"])
            if sha256(truth_path) != truth_row["declared_sha256"]:
                raise ValueError(f"truth SHA mismatch: {scene}")
            truth_xyz, semantic_gt, instance_gt = read_truth(truth_path)
            prediction_path = locate_prediction(method, scene, args.base, forainet_index)
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            if method == "chm_watershed":
                pred_xyz, semantic_pred, instance_pred = read_las_prediction(prediction_path, method)
            elif method == "segmentanytree":
                truthfree_input = (
                    Path(os.environ.get("ALS_PROJECT_ROOT", ".")) / "evaluations"
                    / "forestformer3d_teacher_official28_truthfree_predictions_20260829_r2"
                    / "truthfree_inputs"
                    / f"{scene}.ply"
                )
                pred_xyz, semantic_pred, instance_pred = read_segmentanytree_aligned(
                    prediction_path, truthfree_input
                )
            elif method == "treelearn":
                pred_xyz, semantic_pred, instance_pred = read_treelearn(prediction_path)
            else:
                pred_xyz, semantic_pred, instance_pred = read_forainet(prediction_path)
            semantic_pred, instance_pred, coordinate = stable_align(
                truth_xyz,
                pred_xyz,
                semantic_pred,
                instance_pred,
                las_truncation_grid=False,
            )
            counts, matches = evaluate_scene(semantic_gt, instance_gt, semantic_pred, instance_pred)
            row = {
                "method": method, "source": source, "scene": scene,
                "truth_sha256": truth_row["declared_sha256"],
                "prediction_path": str(prediction_path.resolve()),
                "prediction_sha256": sha256(prediction_path),
                **coordinate, **counts, **ratios(counts),
            }
            scene_rows.append(row)
            for match in matches:
                match_rows.append({"method": method, "source": source, "scene": scene, **match})
            print(method, scene, counts["tp"], counts["fp"], counts["fn"], flush=True)

    source_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    for method in methods:
        method_rows = [row for row in scene_rows if row["method"] == method]
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in method_rows:
            by_source[row["source"]].append(row)
        for source, rows in sorted(by_source.items()):
            counts = sum_counts(rows)
            source_rows.append({"method": method, "source": source, "scene_count": len(rows), **counts, **ratios(counts)})
        counts = sum_counts(method_rows)
        source_metrics = [row for row in source_rows if row["method"] == method]
        macro_keys = tuple(ratios(counts))
        macro = {
            f"source_macro_{key}": float(np.nanmean([row[key] for row in source_metrics]))
            for key in macro_keys
        }
        overall_rows.append({
            "method": method, "scene_count": len(method_rows), "source_count": len(source_metrics),
            **counts, **ratios(counts), **macro,
            "average_precision": None,
            "average_precision_status": "UNSUPPORTED_NO_COMPARABLE_INSTANCE_CONFIDENCE",
        })

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        fields = list(rows[0]) if rows else []
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)

    write_csv(args.output_dir / "per_scene_metrics.csv", scene_rows)
    write_csv(args.output_dir / "per_source_metrics.csv", source_rows)
    write_csv(args.output_dir / "overall_metrics.csv", overall_rows)
    write_csv(args.output_dir / "instance_matches.csv", match_rows)
    audit = {
        "schema_version": "external_baseline_official28_unified_v1",
        "scene_count": len(inventory),
        "methods": list(methods),
        "truth_definition": "semantic_seg != 1 is tree; semantic_seg == 1 is ground; treeID 0 is background",
        "instance_matching": "per-scene Hungarian one-to-one; IoU strictly greater than 0.5",
        "panoptic_quality": "sum matched IoU / (TP + 0.5 FP + 0.5 FN)",
        "coordinate_guard_max_abs_axis_m": MAX_AXIS_RESIDUAL_M,
        "segmentanytree_alignment": "sealed r2 labels in verified truth-free input order",
        "segmentanytree_alignment_seal_sha256": sha256(
            args.base / "segmentanytree_official28_sealed_r2_aligned"
            / "SEGMENTANYTREE_SEALED_R2_ALIGNMENT_SEAL.json"
        ),
        "truth_inventory_sha256": sha256(args.truth_inventory),
        "forainet_seal_sha256": sha256(fseal_path),
        "outputs": {},
    }
    for name in ("per_scene_metrics.csv", "per_source_metrics.csv", "overall_metrics.csv", "instance_matches.csv"):
        audit["outputs"][name] = sha256(args.output_dir / name)
    audit_path = args.output_dir / "EVALUATION_SEAL.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print("EVALUATION_COMPLETE", sha256(audit_path), flush=True)


if __name__ == "__main__":
    main()

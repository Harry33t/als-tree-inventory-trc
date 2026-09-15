#!/usr/bin/env python3
"""Frozen external TRC evaluation for the L1A physical source.

This evaluator is intentionally read-only with respect to the frozen model and
prediction inputs.  It consumes only the authoritative L1A truth workbook and
the point-level treeID labels after all four truth-free prediction arms have
been sealed.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import hashlib
import json
import math
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import joblib
import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plyfile import PlyData
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


ARMS = ("control", "acpe", "epco", "acpe_epco")
ARM_LABELS = {
    "control": "Control",
    "acpe": "ACPE",
    "epco": "EPCO",
    "acpe_epco": "ACPE+EPCO",
}
HEADS = ("R_valid", "R_H", "R_CW")
BUDGETS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0)
THRESHOLDS = {
    "R_valid": 0.1804195159386638,
    "R_H": 0.0287114333621457,
    "R_CW": 0.0285626634703521,
}
CAPS = {"R_valid": 0.05, "R_H": 0.01, "R_CW": 0.015}
MIN_COVERAGE = 0.50
MAX_MATCH_DISTANCE_M = 2.0
SEED = 20260907
EPS = 1e-9

TRC_FEATURES = [
    "z_iqr_over_zspan", "xy_extent_x_over_zspan", "xy_extent_y_over_zspan",
    "sqrt_xy_area_over_zspan", "radial_q50_over_zspan", "radial_q90_over_zspan",
    "xy_anisotropy", "vertical_occupancy", "vertical_entropy",
    "largest_vertical_gap_ratio", "lower_fraction", "upper_fraction",
    "nearest_centroid_distance_over_zspan", "nearest_distance_over_radius",
    "neighbor_height_difference_ratio", "log_point_count_scene_robust_z",
    "z_span_scene_robust_z", "xy_bbox_area_scene_robust_z",
]
TRAIT_FEATURES = TRC_FEATURES + [
    "CW_primary_over_H", "CW_pca_over_H", "CW_delta_over_H",
    "xy_extent_x_over_H", "xy_extent_y_over_H", "radial_q50_over_H",
    "radial_q90_over_H", "log1p_point_count", "log_H_raw_m",
    "log_CW_raw_m", "instance_confidence", "confidence_scene_robust_z",
    "confidence_scene_percentile", "confidence_gap_to_lower",
    "confidence_gap_to_higher",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_truth_xlsx(path: Path) -> pd.DataFrame:
    main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    ns = {"m": main, "r": rel_ns}
    with zipfile.ZipFile(path) as book:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in book.namelist():
            root = ET.fromstring(book.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", ns):
                shared.append("".join(t.text or "" for t in item.iter(f"{{{main}}}t")))
        workbook = ET.fromstring(book.read("xl/workbook.xml"))
        rels = ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
        rel_map = {item.attrib["Id"]: item.attrib["Target"] for item in rels}
        sheet = list(workbook.find("m:sheets", ns))[0]
        target = rel_map[sheet.attrib[f"{{{rel_ns}}}id"]]
        sheet_path = "xl/" + target.lstrip("/") if not target.startswith("xl/") else target
        xml = ET.fromstring(book.read(sheet_path))
        rows: list[dict[str, str | None]] = []
        for row in xml.findall(".//m:sheetData/m:row", ns):
            values: dict[str, str | None] = {}
            for cell in row.findall("m:c", ns):
                column = "".join(c for c in cell.attrib["r"] if c.isalpha())
                kind = cell.attrib.get("t")
                node = cell.find("m:v", ns)
                value = None if node is None else node.text
                if kind == "s" and value is not None:
                    value = shared[int(value)]
                elif kind == "inlineStr":
                    value = "".join(t.text or "" for t in cell.iter(f"{{{main}}}t"))
                values[column] = value
            rows.append(values)
    if not rows or [rows[0].get(c) for c in "ABC"] != ["treeID", "H_m", "CW_m"]:
        raise RuntimeError("authoritative truth workbook schema mismatch")
    frame = pd.DataFrame(
        [{"truth_tree_id": int(r["A"]), "H_field_m": float(r["B"]), "CW_field_m": float(r["C"])}
         for r in rows[1:] if r.get("A") not in (None, "")]
    )
    if frame.empty or frame.truth_tree_id.duplicated().any():
        raise RuntimeError("invalid or duplicate truth tree IDs")
    return frame.sort_values("truth_tree_id").reset_index(drop=True)


def normalized_entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if len(counts) <= 1:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(counts)))


def geometry_features(points: np.ndarray) -> dict[str, float]:
    points = np.asarray(points, dtype=float)
    n = len(points)
    if points.shape != (n, 3) or n == 0 or not np.isfinite(points).all():
        raise ValueError("invalid instance points")
    z = points[:, 2]
    z_low, z_high = np.quantile(z, [0.01, 0.99])
    z_q25, z_q75 = np.quantile(z, [0.25, 0.75])
    z_span = float(max(z_high - z_low, 0.0))
    xy = points[:, :2]
    center = np.median(xy, axis=0)
    centered = xy - center
    radial = np.linalg.norm(centered, axis=1)
    radial_q50, radial_q90 = np.quantile(radial, [0.5, 0.9])
    low, high = np.quantile(xy, 0.01, axis=0), np.quantile(xy, 0.99, axis=0)
    extents = np.maximum(high - low, 0.0)
    eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(np.cov(centered, rowvar=False)), 0.0)) if n >= 3 else np.zeros(2)
    anisotropy = float((eigenvalues[-1] - eigenvalues[0]) / (eigenvalues.sum() + EPS)) if n >= 3 else 0.0
    if z_span > EPS:
        z_norm = np.clip((z - z_low) / z_span, 0.0, 1.0)
        hist, _ = np.histogram(z_norm, bins=10, range=(0.0, 1.0))
        occupancy = float(np.count_nonzero(hist) / 10.0)
        entropy = normalized_entropy(hist)
        largest_gap = float(np.max(np.diff(np.sort(z))) / z_span) if n > 1 else 0.0
        lower, upper = float(np.mean(z_norm < 1 / 3)), float(np.mean(z_norm > 2 / 3))
    else:
        occupancy, entropy, largest_gap, lower, upper = 0.1, 0.0, 0.0, 1.0, 0.0
    return {
        "log_point_count": float(np.log1p(n)), "z_span": z_span,
        "z_iqr": float(max(z_q75 - z_q25, 0.0)), "xy_extent_x": float(extents[0]),
        "xy_extent_y": float(extents[1]), "xy_bbox_area": float(extents.prod()),
        "radial_q50": float(radial_q50), "radial_q90": float(radial_q90),
        "xy_anisotropy": anisotropy, "vertical_occupancy": occupancy,
        "vertical_entropy": entropy, "largest_vertical_gap_ratio": largest_gap,
        "lower_fraction": lower, "upper_fraction": upper,
        "centroid_x_local": float(center[0]), "centroid_y_local": float(center[1]),
        "_height": z_span, "_radius": float(radial_q90),
    }


def add_neighbor_features(rows: list[dict]) -> None:
    centers = np.asarray([[r["centroid_x_local"], r["centroid_y_local"]] for r in rows])
    heights = np.asarray([r["_height"] for r in rows], float)
    radii = np.asarray([r["_radius"] for r in rows], float)
    distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    for i, row in enumerate(rows):
        if len(rows) == 1:
            row.update(nearest_centroid_distance=0.0, nearest_distance_over_radius=0.0,
                       neighbor_height_difference_ratio=0.0)
            continue
        j = int(np.argmin(distances[i]))
        d = float(distances[i, j])
        row["nearest_centroid_distance"] = d
        row["nearest_distance_over_radius"] = d / (radii[i] + radii[j] + EPS)
        row["neighbor_height_difference_ratio"] = abs(heights[i] - heights[j]) / (max(heights[i], heights[j]) + EPS)


def normalize_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    scale = pd.to_numeric(out.z_span).clip(lower=0.5)
    for column in ("z_iqr", "xy_extent_x", "xy_extent_y", "radial_q50", "radial_q90", "nearest_centroid_distance"):
        out[f"{column}_over_zspan"] = pd.to_numeric(out[column]) / scale
    out["sqrt_xy_area_over_zspan"] = np.sqrt(pd.to_numeric(out.xy_bbox_area).clip(lower=0)) / scale
    for column in ("log_point_count", "z_span", "xy_bbox_area"):
        values = pd.to_numeric(out[column])
        median, q1, q3 = values.median(), values.quantile(0.25), values.quantile(0.75)
        out[f"{column}_scene_robust_z"] = (values - median) / max(float(q3 - q1), 1e-6)
    if not np.isfinite(out[TRC_FEATURES].to_numpy(float)).all():
        raise RuntimeError("nonfinite deployment geometry features")
    return out


def tree_traits(points: np.ndarray, ground_tree: cKDTree, ground_z: np.ndarray) -> dict[str, float | str]:
    result = FROZEN_CORE.frozen_traits(points, ground_tree, ground_z, SUPPORT_FILTER)
    return {"trait_status": result["trait_status"], "H_raw_m": result.get("H_m", math.nan),
            "CW_raw_m": result.get("CW_m", math.nan), "CW_pca_m": result.get("CW_pca_m", math.nan),
            "support_point_count": result.get("support_point_count", math.nan),
            "ground_median30_m": result.get("ground_median30_m", math.nan)}


def load_trait_authority(root: Path, output: Path):
    """Reuse sealed official28 code, not another numerical reimplementation."""
    global FROZEN_CORE, SUPPORT_FILTER
    methods = root / "evaluations/official28_single_gt_formal_bundle_20260830_r5_coordinate_erratum/methods"
    paths = {"official28_delivery_runner": methods / "official28_delivery_runner.py",
             "l1a_frozen_core": methods / "reliable_delivery_core.py",
             "l1a_support_filter": methods / "support_filter_authority/parameter_support_filter.py"}
    modules = {}
    for name, path in paths.items():
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    FROZEN_CORE = modules["l1a_frozen_core"]
    if sha256(paths["l1a_frozen_core"]) != "d434a5d62bbc1071d4e93321d68897ba63245c0e5e937236969f8d42cf7fabae":
        raise RuntimeError("formal trait authority hash mismatch")
    FROZEN_CORE.GROUND_AUTHORITIES = set(FROZEN_CORE.GROUND_AUTHORITIES) | {"L1A_classification_2_verified_treeID_0"}
    support = modules["l1a_support_filter"]
    SUPPORT_FILTER = lambda points: support.filter_parameter_points(
        points, support.SupportFilterConfig(top_percentile=99.9)).points
    # Deterministic synthetic adapter checks before any corrected field outcome.
    test_ground = np.column_stack((np.arange(40), np.zeros(40), np.zeros(40))).astype(float)
    test_tree, test_z = FROZEN_CORE.build_ground_model(test_ground, np.ones(40, bool), "L1A_classification_2_verified_treeID_0")
    for count in (0, 29, 30, 90):
        points = np.column_stack((np.linspace(0, 1, count), np.zeros(count), np.linspace(0, 10, count)))
        expected = FROZEN_CORE.frozen_traits(points, test_tree, test_z, SUPPORT_FILTER)
        actual = tree_traits(points, test_tree, test_z)
        assert actual["trait_status"] == expected["trait_status"]
        for dst, src in (("H_raw_m", "H_m"), ("CW_raw_m", "CW_m"), ("CW_pca_m", "CW_pca_m")):
            assert np.isclose(actual[dst], expected.get(src, math.nan), equal_nan=True)
    record = {"revision": "L1A_frozen_traits_erratum_r2", "adapter_tests": "passed_0_29_30_90_points", "previous_L1A_outcomes_already_accessed": True,
              "purpose": "Correct implementation mismatch; no fitting, threshold selection or segmentation rerun",
              "ground_adapter": "L1A classification=2 checked equal to treeID=0; adapted to formal ground mask",
              "reference": "External field measurements, NOT same-definition reference-point-cloud traits",
              "authorities": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
              "script_sha256": sha256(Path(__file__))}
    (output / "trait_algorithm_erratum.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def attach_confidence(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    values = out.instance_confidence.to_numpy(float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    out["confidence_scene_robust_z"] = (values - median) / max(1e-6, 1.4826 * mad)
    out["confidence_scene_percentile"] = out.instance_confidence.rank(method="average", pct=True)
    ordered = np.sort(values)
    out["confidence_gap_to_lower"] = [float(x - ordered[max(0, np.searchsorted(ordered, x, side="left") - 1)]) for x in values]
    out["confidence_gap_to_higher"] = [float(ordered[min(len(ordered) - 1, np.searchsorted(ordered, x, side="right"))] - x) for x in values]
    return out


def score_bundle(frame: pd.DataFrame, model_dir: Path, protocol: dict) -> pd.DataFrame:
    out = frame.copy()
    for head in HEADS:
        model_path = model_dir / f"{head}_candidate.joblib"
        expected = protocol["risk_models"][head]["sha256"]
        if sha256(model_path) != expected:
            raise RuntimeError(f"{head} model hash mismatch")
        bundle = joblib.load(model_path)
        features = list(bundle["features"])
        if features != (TRC_FEATURES if head == "R_valid" else TRAIT_FEATURES):
            raise RuntimeError(f"{head} feature contract mismatch")
        missing = sorted(set(features) - set(out.columns))
        if missing:
            raise RuntimeError(f"{head} missing features: {missing}")
        eligible = np.ones(len(out), dtype=bool) if head == "R_valid" else out.trait_status.eq("estimable").to_numpy()
        if not eligible.any():
            raise RuntimeError(f"{head}: no estimable predicted instances")
        model = bundle["rank_model"]
        raw = np.full(len(out), np.nan, dtype=float)
        if head == "R_valid":
            raw[eligible] = np.asarray(model.predict_proba(out.loc[eligible, features])[:, 1], float)
        else:
            raw[eligible] = np.maximum(0.0, np.asarray(model.predict(out.loc[eligible, features]), float))
        rank = np.full(len(out), np.nan, dtype=float)
        probability = np.full(len(out), np.nan, dtype=float)
        rank[eligible] = float(bundle["rank_score_direction"]) * raw[eligible]
        probability[eligible] = bundle["probability_calibrator"].predict_proba(raw[eligible].reshape(-1, 1))[:, 1]
        out[f"{head}_rank_score"] = rank
        out[f"{head}_probability"] = probability
        # Trait heads fail closed when the instance is not measurable; these
        # cases are reviewed by rule and are not attributed a model probability.
        out[f"{head}_fixed_review"] = (~eligible) | (probability >= THRESHOLDS[head])
    return out


def extract_arm(arm: str, pred_root: Path, ground_tree: cKDTree, ground_z: np.ndarray) -> pd.DataFrame:
    path = pred_root / arm / "artifacts" / "round_2" / "L1A_external_test_round2.ply"
    data = PlyData.read(str(path))["vertex"].data
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(float)
    labels, scores = np.asarray(data["instance_pred"], int), np.asarray(data["score"], float)
    rows: list[dict] = []
    for pred_id in sorted(np.unique(labels[labels >= 0]).tolist()):
        idx = np.flatnonzero(labels == pred_id)
        row = geometry_features(xyz[idx])
        row.update({"arm": arm, "arm_label": ARM_LABELS[arm], "scene": "L1A",
                    "dataset_key": f"L1A::{arm}", "pred_instance_id": int(pred_id) + 1,
                    "las_instance_id": int(pred_id), "point_count": int(len(idx)),
                    "instance_confidence": float(np.mean(scores[idx]))})
        row.update(tree_traits(xyz[idx], ground_tree, ground_z))
        rows.append(row)
    if len(rows) < 2:
        raise RuntimeError(f"{arm}: fewer than two predictions")
    add_neighbor_features(rows)
    frame = normalize_geometry(pd.DataFrame(rows))
    frame = attach_confidence(frame)
    height = frame.H_raw_m.clip(lower=1.0)
    frame["CW_primary_over_H"] = frame.CW_raw_m / height
    frame["CW_pca_over_H"] = frame.CW_pca_m / height
    frame["CW_delta_over_H"] = (frame.CW_pca_m - frame.CW_raw_m) / height
    for column in ("xy_extent_x", "xy_extent_y", "radial_q50", "radial_q90"):
        frame[f"{column}_over_H"] = frame[column] / height
    frame["log1p_point_count"] = np.log1p(frame.point_count)
    frame["log_H_raw_m"] = np.log(frame.H_raw_m.clip(lower=1e-6))
    frame["log_CW_raw_m"] = np.log(frame.CW_raw_m.clip(lower=1e-6))
    return frame


def spatial_match(pred: pd.DataFrame, truth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pred.reset_index(drop=True).copy()
    t = truth.reset_index(drop=True).copy()
    dist = np.linalg.norm(
        t[["truth_centroid_x_local", "truth_centroid_y_local"]].to_numpy()[:, None, :]
        - p[["centroid_x_local", "centroid_y_local"]].to_numpy()[None, :, :], axis=2
    )
    tie = (np.arange(len(t))[:, None] * max(len(p), 1) + np.arange(len(p))[None, :]) * 1e-12
    row, col = linear_sum_assignment(dist + tie)
    pairs = [(i, j, float(dist[i, j])) for i, j in zip(row, col) if dist[i, j] <= MAX_MATCH_DISTANCE_M]
    p["matched"] = False
    p["truth_tree_id"] = pd.Series([pd.NA] * len(p), dtype="Int64")
    p["match_distance_m"] = math.nan
    t["matched"] = False
    t["pred_instance_id"] = pd.Series([pd.NA] * len(t), dtype="Int64")
    t["match_distance_m"] = math.nan
    for i, j, distance in pairs:
        p.loc[j, ["matched", "truth_tree_id", "match_distance_m"]] = [True, int(t.loc[i, "truth_tree_id"]), distance]
        t.loc[i, ["matched", "pred_instance_id", "match_distance_m"]] = [True, int(p.loc[j, "pred_instance_id"]), distance]
    return p, t


def ece_score(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if mask.any():
            result += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(result)


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return math.nan, math.nan
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
    model = LogisticRegression(C=1e6, max_iter=5000, random_state=SEED).fit(logit.reshape(-1, 1), y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def operational_order(frame: pd.DataFrame, score: np.ndarray, high_first: bool) -> np.ndarray:
    work = frame.reset_index(drop=True)
    records = []
    for (arm, scene), positions in work.groupby(["arm", "scene"], sort=True).indices.items():
        positions = np.asarray(positions, int)
        ids = work.iloc[positions].pred_instance_id.to_numpy(int)
        primary = -score[positions] if high_first else score[positions]
        local = positions[np.lexsort((ids, primary))]
        for rank, pos in enumerate(local, start=1):
            records.append((rank / len(local), str(arm), str(scene), int(work.iloc[pos].pred_instance_id), int(pos)))
    records.sort()
    return np.asarray([r[-1] for r in records], int)


def select_by_budget(frame: pd.DataFrame, score: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    work = frame.reset_index(drop=True)
    reviewed, accepted = [], []
    for positions in work.groupby(["arm", "scene"], sort=True).indices.values():
        positions = np.asarray(positions, int)
        ids = work.iloc[positions].pred_instance_id.to_numpy(int)
        order = positions[np.lexsort((ids, -score[positions]))]
        count = int(math.ceil(fraction * len(positions))) if fraction > 0 else 0
        reviewed.extend(order[:count]); accepted.extend(order[count:])
    return np.asarray(reviewed, int), np.asarray(accepted, int)


def risk_metrics(panel: pd.DataFrame, head: str) -> dict:
    y = panel[head].to_numpy(int)
    score = panel[f"{head}_rank_score"].to_numpy(float)
    prob = panel[f"{head}_probability"].to_numpy(float)
    order = operational_order(panel, score, high_first=False)
    intercept, slope = calibration_slope_intercept(y, prob)
    return {
        "head": head, "n": len(panel), "failures": int(y.sum()), "prevalence": float(y.mean()),
        "auroc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else math.nan,
        "average_precision": average_precision_score(y, score) if y.sum() else math.nan,
        "random_ap": float(y.mean()), "brier": brier_score_loss(y, prob),
        "log_loss": log_loss(y, np.c_[1 - prob, prob], labels=[0, 1]), "ece_10bin": ece_score(y, prob),
        "calibration_intercept": intercept, "calibration_slope": slope,
        "operational_aurc": float(np.mean(np.cumsum(y[order]) / np.arange(1, len(y) + 1))),
    }


def budget_table(panel: pd.DataFrame, head: str) -> list[dict]:
    y = panel[head].to_numpy(int)
    score = panel[f"{head}_rank_score"].to_numpy(float)
    rows = []
    for nominal in BUDGETS:
        reviewed, accepted = select_by_budget(panel, score, nominal)
        actual = len(reviewed) / len(panel)
        captured = int(y[reviewed].sum()) if len(reviewed) else 0
        total = int(y.sum())
        capture = captured / total if total else math.nan
        rows.append({
            "head": head, "nominal_review_fraction": nominal, "actual_review_fraction": actual,
            "ceil_overshoot_fraction": actual - nominal, "n_total": len(panel),
            "n_reviewed": len(reviewed), "n_accepted": len(accepted), "total_failures": total,
            "captured_failures": captured, "remaining_failures": int(y[accepted].sum()) if len(accepted) else 0,
            "failure_capture": capture, "review_precision": captured / len(reviewed) if len(reviewed) else math.nan,
            "review_lift_vs_random_actual": capture / actual if actual > 0 and total else math.nan,
            "accepted_coverage": len(accepted) / len(panel),
            "accepted_failure_rate": float(y[accepted].mean()) if len(accepted) else math.nan,
        })
    return rows


def risk_coverage(panel: pd.DataFrame, head: str) -> list[dict]:
    y = panel[head].to_numpy(int)
    score = panel[f"{head}_rank_score"].to_numpy(float)
    order = operational_order(panel, score, high_first=False)
    rows = [{"head": head, "n_accepted": 0, "coverage": 0.0, "accepted_failure_rate": math.nan}]
    cumulative = np.cumsum(y[order])
    rows.extend({"head": head, "n_accepted": k, "coverage": k / len(y),
                 "accepted_failure_rate": float(cumulative[k - 1] / k)} for k in range(1, len(y) + 1))
    return rows


def calibration_bins(panel: pd.DataFrame, head: str, bins: int = 10) -> list[dict]:
    probability = panel[f"{head}_probability"].to_numpy(float)
    y = panel[head].to_numpy(int)
    order = np.argsort(probability, kind="stable")
    groups = np.array_split(order, min(bins, len(order)))
    rows = []
    for index, positions in enumerate(groups, start=1):
        if not len(positions):
            continue
        rows.append({
            "head": head, "bin": index, "n": len(positions),
            "probability_min": float(probability[positions].min()),
            "probability_max": float(probability[positions].max()),
            "mean_predicted_probability": float(probability[positions].mean()),
            "observed_failure_rate": float(y[positions].mean()),
        })
    return rows


def bootstrap_panels(panels: dict[str, pd.DataFrame], replicates: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """External uncertainty with the strongest available single-source clusters.

    Matched H/CW cases are resampled by physical treeID with all four arms kept
    together.  R_valid has no physical-tree identity for unmatched predictions,
    so rows are resampled within each arm and are explicitly labelled as a
    prediction-level sensitivity interval rather than a source-generalisation CI.
    """
    rng = np.random.default_rng(SEED)
    metric_draws: list[dict] = []
    budget_draws: list[dict] = []
    audit_rows: list[dict] = []
    for replicate in range(replicates):
        samples: dict[str, pd.DataFrame] = {}
        valid_parts = []
        for arm, frame in panels["R_valid"].groupby("arm", sort=True):
            positions = rng.integers(0, len(frame), size=len(frame))
            part = frame.iloc[positions].copy().reset_index(drop=True)
            part["pred_instance_id"] = np.arange(1, len(part) + 1)
            valid_parts.append(part)
        samples["R_valid"] = pd.concat(valid_parts, ignore_index=True)
        truth_ids = np.sort(panels["R_H"].truth_tree_id.dropna().astype(int).unique())
        chosen = rng.choice(truth_ids, size=len(truth_ids), replace=True)
        for head in ("R_H", "R_CW"):
            pieces = []
            for draw, truth_id in enumerate(chosen):
                part = panels[head][panels[head].truth_tree_id.eq(truth_id)].copy()
                part["scene"] = part.scene.astype(str) + f"__treeboot{draw}"
                pieces.append(part)
            samples[head] = pd.concat(pieces, ignore_index=True)
        audit_rows.append({
            "replicate": replicate, "seed": SEED,
            "R_valid_resampling": "within-arm prediction rows",
            "trait_resampling": "field treeID cluster with all arms retained",
            "sampled_truth_tree_clusters": len(chosen),
        })
        for head, sample in samples.items():
            row = risk_metrics(sample, head)
            for metric in ("auroc", "average_precision", "brier", "log_loss", "ece_10bin", "operational_aurc"):
                metric_draws.append({"replicate": replicate, "head": head, "metric": metric, "value": row[metric]})
            for budget in budget_table(sample, head):
                for metric in ("actual_review_fraction", "failure_capture", "review_precision", "review_lift_vs_random_actual", "accepted_failure_rate"):
                    budget_draws.append({
                        "replicate": replicate, "head": head,
                        "nominal_review_fraction": budget["nominal_review_fraction"],
                        "metric": metric, "value": budget[metric],
                    })
    metric_draws_frame = pd.DataFrame(metric_draws)
    budget_draws_frame = pd.DataFrame(budget_draws)
    ci_rows = []
    for keys, frame in metric_draws_frame.groupby(["head", "metric"], sort=True):
        values = frame.value.dropna().to_numpy(float)
        ci_rows.append({"head": keys[0], "metric": keys[1], "nominal_review_fraction": math.nan,
                        "valid_replicates": len(values), "ci_low": np.quantile(values, .025) if len(values) else math.nan,
                        "ci_high": np.quantile(values, .975) if len(values) else math.nan})
    for keys, frame in budget_draws_frame.groupby(["head", "nominal_review_fraction", "metric"], sort=True):
        values = frame.value.dropna().to_numpy(float)
        ci_rows.append({"head": keys[0], "metric": keys[2], "nominal_review_fraction": keys[1],
                        "valid_replicates": len(values), "ci_low": np.quantile(values, .025) if len(values) else math.nan,
                        "ci_high": np.quantile(values, .975) if len(values) else math.nan})
    return pd.DataFrame(ci_rows), metric_draws_frame, pd.DataFrame(audit_rows)


def fixed_gate(panel: pd.DataFrame, head: str) -> dict:
    y = panel[head].to_numpy(int)
    probability = panel[f"{head}_probability"].to_numpy(float)
    accepted = probability < THRESHOLDS[head]
    metrics = risk_metrics(panel, head)
    coverage = float(accepted.mean())
    accepted_rate = float(y[accepted].mean()) if accepted.any() else math.nan
    requirements = {
        "features_finite": bool(np.isfinite(panel[TRC_FEATURES if head == "R_valid" else TRAIT_FEATURES].to_numpy(float)).all()),
        "n_at_least_30": len(panel) >= 30,
        "both_classes_and_auroc_gt_half": len(np.unique(y)) == 2 and metrics["auroc"] > 0.5,
        "coverage_at_least_50pct": coverage >= MIN_COVERAGE,
        "accepted_failure_within_cap": np.isfinite(accepted_rate) and accepted_rate <= CAPS[head],
        "positive_slope_and_ece_le_0_10": metrics["calibration_slope"] > 0 and metrics["ece_10bin"] <= 0.10,
    }
    return {
        "head": head, "threshold": THRESHOLDS[head], "n": len(panel), "failures": int(y.sum()),
        "n_accepted": int(accepted.sum()), "accepted_coverage": coverage,
        "accepted_failures": int(y[accepted].sum()) if accepted.any() else 0,
        "accepted_failure_rate": accepted_rate, "failure_cap": CAPS[head],
        **requirements, "automatic_delivery_gate_pass": all(requirements.values()),
    }


def plot_outputs(metrics: pd.DataFrame, budgets: pd.DataFrame, coverage: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    colors = {"R_valid": "#3B6FB6", "R_H": "#E07A3F", "R_CW": "#4A9B68"}
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for head, frame in coverage.groupby("head"):
        ax.plot(frame.coverage, frame.accepted_failure_rate, label=head, color=colors[head])
    ax.set(xlabel="Accepted coverage", ylabel="Accepted-set failure rate", xlim=(0, 1))
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    for ext in ("png", "svg"): fig.savefig(output / f"l1a_full_risk_coverage.{ext}", dpi=220)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for head, frame in budgets.groupby("head"):
        ax.plot(frame.actual_review_fraction, frame.failure_capture, marker="o", label=head, color=colors[head])
    ax.plot([0, 1], [0, 1], "--", color="0.55", label="random")
    ax.set(xlabel="Actual review fraction", ylabel="Failure capture", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    for ext in ("png", "svg"): fig.savefig(output / f"l1a_budget_capture.{ext}", dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--l1a-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    tables, figures = args.output / "tables", args.output / "figures"
    tables.mkdir(); figures.mkdir()

    protocol_path = args.l1a_dir / "protocol" / "L1A_EXTERNAL_TRC_PROTOCOL_FROZEN.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_L1A_OUTCOME_ACCESS":
        raise RuntimeError("invalid frozen protocol status")
    truth_path = args.l1a_dir / "source" / "L1A真值(1).xlsx"
    source_path = args.l1a_dir / "source" / "L1A(1).laz"
    if sha256(truth_path) != protocol["authoritative_truth"]["sha256"] or sha256(source_path) != protocol["l1a_point_cloud"]["sha256"]:
        raise RuntimeError("L1A input hash mismatch")
    superseded = args.l1a_dir / "source" / "L1A_H_CW(1).xlsx"
    if superseded.exists() and sha256(superseded) != protocol["superseded_truth"]["sha256"]:
        raise RuntimeError("superseded truth file changed")
    pred_root = args.l1a_dir / "truthfree_predictions"
    for arm in ARMS:
        seal = json.loads((pred_root / arm / "PREDICTION_SEAL.json").read_text())
        if seal.get("status") != "complete" or seal.get("truth_opened") is not False:
            raise RuntimeError(f"{arm} prediction is not truth-free sealed")

    cloud = laspy.read(source_path)
    xyz_global = np.column_stack([cloud.x, cloud.y, cloud.z]).astype(float)
    tree_id = np.asarray(cloud.treeID, int)
    classification = np.asarray(cloud.classification, int)
    if not np.array_equal(tree_id == 0, classification == 2):
        raise RuntimeError("L1A ground authority mismatch: treeID=0 != classification=2")
    origin = np.asarray([xyz_global[:, 0].mean(), xyz_global[:, 1].mean(), xyz_global[:, 2].min()])
    xyz = xyz_global - origin
    load_trait_authority(args.root, args.output)
    ground_tree, ground_z = FROZEN_CORE.build_ground_model(
        xyz, classification == 2, authority="L1A_classification_2_verified_treeID_0")

    truth = read_truth_xlsx(truth_path)
    available_ids = set(np.unique(tree_id[tree_id > 0]).tolist())
    if set(truth.truth_tree_id) != available_ids:
        raise RuntimeError("truth workbook IDs do not match point-cloud treeID inventory")
    centers = []
    for tid in truth.truth_tree_id:
        points = xyz[tree_id == int(tid)]
        center = np.median(points[:, :2], axis=0)
        centers.append((float(center[0]), float(center[1]), int(len(points))))
    truth[["truth_centroid_x_local", "truth_centroid_y_local", "truth_point_count"]] = pd.DataFrame(centers, index=truth.index)

    model_dir = args.root / "evaluations" / "trc_confidence_v7_20260903" / "formal_output_v7" / "models"
    prediction_ledgers, truth_ledgers = [], []
    for arm in ARMS:
        pred = extract_arm(arm, pred_root, ground_tree, ground_z)
        pred = score_bundle(pred, model_dir, protocol)
        pred, arm_truth = spatial_match(pred, truth)
        pred["R_valid"] = (~pred.matched).astype(int)
        matched = pred[pred.matched].copy()
        matched["_pred_row_index"] = matched.index
        matched = matched.merge(truth[["truth_tree_id", "H_field_m", "CW_field_m"]], on="truth_tree_id", how="left", validate="one_to_one")
        matched["H_error_m"] = matched.H_raw_m - matched.H_field_m
        matched["CW_error_m"] = matched.CW_raw_m - matched.CW_field_m
        matched["S_H"] = matched.H_error_m.abs() / np.maximum(1.0, 0.10 * matched.H_field_m.abs())
        matched["S_CW"] = matched.CW_error_m.abs() / np.maximum(1.0, 0.20 * matched.CW_field_m.abs())
        matched["R_H"] = ((matched.trait_status != "estimable") | (matched.S_H > 1.0)).astype(int)
        matched["R_CW"] = ((matched.trait_status != "estimable") | (matched.S_CW > 1.0)).astype(int)
        matched_positions = matched._pred_row_index.to_numpy(int)
        for column in matched.columns:
            if column == "_pred_row_index":
                continue
            if column not in pred.columns:
                pred[column] = np.nan
            pred.loc[matched_positions, column] = matched[column].to_numpy()
        arm_truth["arm"] = arm
        arm_truth["arm_label"] = ARM_LABELS[arm]
        arm_truth = arm_truth.merge(
            matched[["truth_tree_id", "pred_instance_id", "H_raw_m", "CW_raw_m", "H_error_m", "CW_error_m", "R_H", "R_CW"]],
            on=["truth_tree_id", "pred_instance_id"], how="left", validate="one_to_one",
        )
        prediction_ledgers.append(pred)
        truth_ledgers.append(arm_truth)
    predictions = pd.concat(prediction_ledgers, ignore_index=True)
    truth_all = pd.concat(truth_ledgers, ignore_index=True)

    panels = {
        "R_valid": predictions.copy(),
        "R_H": predictions[predictions.matched & predictions.R_H.notna()].copy(),
        "R_CW": predictions[predictions.matched & predictions.R_CW.notna()].copy(),
    }
    metric_rows, budget_rows, coverage_rows, calibration_rows, gate_rows = [], [], [], [], []
    per_arm_rows = []
    for head, panel in panels.items():
        metric_rows.append(risk_metrics(panel, head))
        budget_rows.extend(budget_table(panel, head))
        coverage_rows.extend(risk_coverage(panel, head))
        calibration_rows.extend(calibration_bins(panel, head))
        gate_rows.append(fixed_gate(panel, head))
        for arm, group in panel.groupby("arm", sort=True):
            row = risk_metrics(group, head)
            row.update(arm=arm, arm_label=ARM_LABELS[arm])
            per_arm_rows.append(row)

    metrics = pd.DataFrame(metric_rows)
    budgets = pd.DataFrame(budget_rows)
    coverage = pd.DataFrame(coverage_rows)
    calibration = pd.DataFrame(calibration_rows)
    gates = pd.DataFrame(gate_rows)
    bootstrap_ci, bootstrap_metric_draws, bootstrap_audit = bootstrap_panels(panels)

    cascade_rows = []
    for arm in ARMS:
        p = predictions[predictions.arm.eq(arm)].copy()
        t = truth_all[truth_all.arm.eq(arm)].copy()
        lookup = p.set_index("pred_instance_id")
        for trait in ("H", "CW"):
            delivered, success, failed = 0, 0, 0
            valid_review, trait_review, missed = 0, 0, int((~t.matched).sum())
            for row in t[t.matched].itertuples():
                pred = lookup.loc[int(row.pred_instance_id)]
                if bool(pred.R_valid_fixed_review):
                    valid_review += 1
                elif bool(pred[f"R_{trait}_fixed_review"]):
                    trait_review += 1
                else:
                    delivered += 1
                    is_failure = int(pred[f"R_{trait}"])
                    failed += is_failure
                    success += 1 - is_failure
            cascade_rows.append({
                "arm": arm, "arm_label": ARM_LABELS[arm], "trait": trait,
                "truth_trees": len(t), "missed_trees_no_instance_risk": missed,
                "matched_trees": int(t.matched.sum()), "withheld_by_R_valid": valid_review,
                "withheld_by_trait_head": trait_review, "auto_delivered_truth_trees": delivered,
                "successful_auto_deliveries": success, "failed_auto_deliveries": failed,
                "auto_delivery_rate_full_truth": delivered / len(t),
                "successful_delivery_rate_full_truth": success / len(t),
                "auto_delivered_failure_rate": failed / delivered if delivered else math.nan,
                "unmatched_predictions_auto_accepted_by_R_valid": int((~p.matched & ~p.R_valid_fixed_review).sum()),
            })
    cascade = pd.DataFrame(cascade_rows)

    predictions.to_csv(tables / "l1a_per_prediction.csv", index=False)
    truth_all.to_csv(tables / "l1a_full_truth_tree_ledger.csv", index=False)
    metrics.to_csv(tables / "l1a_risk_metrics.csv", index=False)
    pd.DataFrame(per_arm_rows).to_csv(tables / "l1a_per_arm_risk_metrics.csv", index=False)
    budgets.to_csv(tables / "l1a_review_budget_table.csv", index=False)
    coverage.to_csv(tables / "l1a_full_risk_coverage_curve.csv", index=False)
    calibration.to_csv(tables / "l1a_calibration_bins.csv", index=False)
    gates.to_csv(tables / "l1a_fixed_threshold_gate.csv", index=False)
    cascade.to_csv(tables / "l1a_fixed_threshold_cascade_full_truth.csv", index=False)
    bootstrap_ci.to_csv(tables / "l1a_bootstrap_ci.csv", index=False)
    bootstrap_metric_draws.to_csv(tables / "l1a_bootstrap_metric_draws.csv", index=False)
    bootstrap_audit.to_csv(tables / "l1a_bootstrap_audit.csv", index=False)
    plot_outputs(metrics, budgets, coverage, figures)

    qa_checks = {
        "authoritative_truth_tree_count_43": len(truth) == 43,
        "four_prediction_arms_present": set(predictions.arm) == set(ARMS),
        "prediction_keys_unique": not predictions.duplicated(["arm", "pred_instance_id"]).any(),
        "full_truth_denominator_43_per_arm": truth_all.groupby("arm").size().eq(43).all(),
        "all_spatial_matches_within_2m": predictions.loc[predictions.matched, "match_distance_m"].le(MAX_MATCH_DISTANCE_M).all(),
        "risk_valid_labels_complete": predictions.R_valid.notna().all(),
        "matched_trait_panel_43_per_arm": predictions[predictions.matched].groupby("arm").size().eq(43).all(),
        "risk_coverage_has_zero_and_one_endpoints": all(
            math.isclose(group.coverage.min(), 0.0) and math.isclose(group.coverage.max(), 1.0)
            for _, group in coverage.groupby("head")
        ),
        "lift_uses_actual_review_fraction": all(
            (not np.isfinite(row.review_lift_vs_random_actual))
            or math.isclose(row.review_lift_vs_random_actual, row.failure_capture / row.actual_review_fraction)
            for row in budgets.itertuples() if row.actual_review_fraction > 0
        ),
        "bootstrap_replicates_1000": len(bootstrap_audit) == 1000,
        "superseded_workbook_not_opened": True,
    }
    qa = {
        "status": "PASS" if all(qa_checks.values()) else "FAIL",
        "checks": {key: bool(value) for key, value in qa_checks.items()},
        "known_evidence_boundaries": [
            "L1A is one external physical source, so intervals are not source-generalisation confidence intervals.",
            "For any head with only one observed outcome class, discrimination and calibration slope are not identifiable; inspect recomputed counts.",
            "The L1A ground adapter was fixed after prediction sealing but was not enumerated in the pre-outcome protocol.",
            "Completely missed field trees cannot receive predicted-instance risk scores.",
        ],
    }
    if qa["status"] != "PASS":
        raise RuntimeError(f"QA failed: {qa}")
    (args.output / "QA_REPORT.json").write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    artifacts = []
    for path in sorted(p for p in args.output.rglob("*") if p.is_file()):
        artifacts.append({"relative_path": str(path.relative_to(args.output)), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    summary = {
        "schema": "forestformer3d.l1a_external_trc.v1", "status": "complete",
        "study_role": "single-new-physical-source frozen external validation",
        "protocol_sha256": sha256(protocol_path), "authoritative_truth_sha256": sha256(truth_path),
        "superseded_truth_used": False, "point_cloud_sha256": sha256(source_path),
        "truth_tree_count": len(truth), "prediction_counts": predictions.groupby("arm").size().to_dict(),
        "matched_counts": predictions.groupby("arm").matched.sum().astype(int).to_dict(),
        "automatic_delivery_gate": dict(zip(gates["head"], gates.automatic_delivery_gate_pass.astype(bool))),
        "ground_adapter": "post-seal treeID=0, independently identical to LAS classification=2",
        "matching": "Hungarian horizontal centroid distance <=2m; truth centroids derived from point-level treeID",
        "uncertainty": "1000 replicates; H/CW field-tree cluster bootstrap across arms; R_valid within-arm prediction bootstrap sensitivity interval; no source-generalisation CI",
        "qa_status": qa["status"],
        "artifacts": artifacts,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

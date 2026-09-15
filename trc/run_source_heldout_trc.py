"""Retrospective source-held-out TRC and CW-recovery development study.

This script consumes only the sealed r5 metrics artifact.  It never edits the
official28 benchmark package and labels every output as retrospective internal
validation rather than a new independent official28 test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260902
EXPECTED_METRICS_SHA256 = (
    "3772b5a348eacb0c5033c7a62c90d30e68880735143d60368f14140ff4a1ce74"
)
ARMS = ["control", "acpe", "epco", "acpe_epco"]
TRC_FEATURES = [
    "z_iqr_over_zspan",
    "xy_extent_x_over_zspan",
    "xy_extent_y_over_zspan",
    "sqrt_xy_area_over_zspan",
    "radial_q50_over_zspan",
    "radial_q90_over_zspan",
    "xy_anisotropy",
    "vertical_occupancy",
    "vertical_entropy",
    "largest_vertical_gap_ratio",
    "lower_fraction",
    "upper_fraction",
    "nearest_centroid_distance_over_zspan",
    "nearest_distance_over_radius",
    "neighbor_height_difference_ratio",
    "log_point_count_scene_robust_z",
    "z_span_scene_robust_z",
    "xy_bbox_area_scene_robust_z",
]
CW_FEATURES = [
    "CW_primary_over_H",
    "CW_pca_over_H",
    "CW_delta_over_H",
    "xy_extent_x_over_H",
    "xy_extent_y_over_H",
    "radial_q50_over_H",
    "radial_q90_over_H",
    "xy_anisotropy",
    "vertical_entropy",
    "largest_vertical_gap_ratio",
]
LEGACY_RISK = {
    "R_valid": "unmatched_prediction_risk",
    "R_H": "H_delivery_failure_risk",
    "R_CW": "CW_recovered_delivery_failure_risk",
}
TARGETS = ["R_valid", "R_H", "R_CW"]
REVIEW_BUDGETS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
TRAIT_STABILITY_FEATURES = CW_FEATURES + [
    "log1p_point_count",
    "log_H_raw_m",
    "log_CW_raw_m",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def cw_success(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    tolerance = np.maximum(1.0, 0.20 * np.abs(truth))
    return np.abs(pred - truth) <= tolerance


def load_panels(metrics_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require(sha256(metrics_path) == EXPECTED_METRICS_SHA256, "sealed metrics SHA mismatch")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    require(payload.get("status") == "complete", "sealed metrics are not complete")
    reliable = payload["reliable_delivery"]
    pred_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        scenes = reliable["arms"][arm]
        require(len(scenes) == 28, f"{arm}: expected 28 scenes")
        for scene_record in scenes:
            scene = str(scene_record["scene"])
            source = str(scene_record["source"])
            preds = scene_record["prediction_ledger"]
            truths = scene_record["truth_ledger"]
            matched_truth_by_pred = {
                int(row["pred_instance_id"]): row
                for row in truths
                if row.get("matched") and row.get("pred_instance_id") is not None
            }
            for row in preds:
                item = dict(row)
                item.update(arm=arm, source=source, scene=scene)
                item["R_valid"] = int(not bool(row.get("matched", False)))
                matched_truth = matched_truth_by_pred.get(int(row["pred_instance_id"]))
                if matched_truth is not None:
                    item["H_truth_m"] = matched_truth.get("H_truth_m")
                    item["H_success"] = matched_truth.get("H_raw_delivery_success")
                    item["CW_truth_m"] = matched_truth.get("CW_truth_m")
                    item["CW_raw_success"] = matched_truth.get("CW_raw_delivery_success")
                    item["match_iou"] = matched_truth.get("match_iou")
                pred_rows.append(item)
            for row in truths:
                item = dict(row)
                item.update(arm=arm, source=source, scene=scene)
                truth_rows.append(item)
            scene_rows.append(
                {
                    "arm": arm,
                    "source": source,
                    "scene": scene,
                    "truth_count": len(truths),
                    "prediction_count": len(preds),
                    "matched_count": sum(bool(x.get("matched")) for x in truths),
                    "missed_count": sum(bool(x.get("missed_truth")) for x in truths),
                    "fail_closed_instance_count_guard": len(preds) <= 1,
                }
            )
    pred = pd.DataFrame(pred_rows)
    truth = pd.DataFrame(truth_rows)
    scenes = pd.DataFrame(scene_rows)
    require(len(scenes) == 112, "expected 4 arms x 28 scenes")
    require(set(pred["source"]) == set(truth["source"]), "source mismatch")
    require(len(set(pred["source"])) == 8, "expected eight sources")
    for column in TRC_FEATURES + CW_FEATURES:
        require(column in pred, f"missing feature {column}")
    pred["log1p_point_count"] = np.log1p(pd.to_numeric(pred["point_count"], errors="coerce"))
    pred["log_H_raw_m"] = np.log(pd.to_numeric(pred["H_raw_m"], errors="coerce").clip(lower=1e-6))
    pred["log_CW_raw_m"] = np.log(pd.to_numeric(pred["CW_raw_m"], errors="coerce").clip(lower=1e-6))
    scored = pred["trc_status"].eq("scored_frozen_models")
    require(scored.any(), "no scored TRC rows")
    require(
        np.isfinite(pred.loc[scored, TRC_FEATURES].to_numpy(float)).all(),
        "non-finite features in scored TRC rows",
    )
    cw_estimable = pred["trait_status"].eq("estimable")
    require(
        np.isfinite(pred.loc[cw_estimable, CW_FEATURES].to_numpy(float)).all(),
        "non-finite CW recovery features in estimable rows",
    )
    return pred, truth, scenes


def matched_trait_panel(pred: pd.DataFrame) -> pd.DataFrame:
    result = pred[
        pred["matched"].astype(bool)
        & pred["trait_status"].eq("estimable")
        & pred["H_truth_m"].notna()
        & pred["CW_truth_m"].notna()
    ].copy()
    result["R_H"] = 1 - result["H_success"].astype(int)
    result["CW_raw_m"] = pd.to_numeric(result["CW_raw_m"])
    result["CW_truth_m"] = pd.to_numeric(result["CW_truth_m"])
    result["H_raw_m"] = pd.to_numeric(result["H_raw_m"]).clip(lower=1e-6)
    result["cw_residual_over_h"] = (
        result["CW_truth_m"] - result["CW_raw_m"]
    ) / result["H_raw_m"]
    return result


def build_recovery(name: str):
    if name == "ridge_residual":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "extra_trees_residual":
        return ExtraTreesRegressor(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=8,
            max_features=0.8,
            random_state=SEED,
            n_jobs=-1,
        )
    if name == "identity_raw":
        return None
    raise KeyError(name)


def recovery_predict(model: Any, frame: pd.DataFrame) -> np.ndarray:
    if model is None:
        return frame["CW_raw_m"].to_numpy(float)
    residual = np.asarray(model.predict(frame[CW_FEATURES]), dtype=float)
    return frame["CW_raw_m"].to_numpy(float) + residual * frame["H_raw_m"].to_numpy(float)


def recovery_score(frame: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    truth = frame["CW_truth_m"].to_numpy(float)
    error = pred - truth
    source_mae = []
    for source in sorted(frame["source"].unique()):
        mask = frame["source"].eq(source).to_numpy()
        source_mae.append(float(np.mean(np.abs(error[mask]))))
    return {
        "macro_source_mae_m": float(np.mean(source_mae)),
        "pooled_mae_m": float(np.mean(np.abs(error))),
        "pooled_rmse_m": float(np.sqrt(np.mean(error**2))),
        "pooled_bias_m": float(np.mean(error)),
        "delivery_success_rate": float(np.mean(cw_success(pred, truth))),
    }


def fit_recovery_nested(train: pd.DataFrame) -> tuple[str, Any, np.ndarray, pd.DataFrame]:
    candidates = ["identity_raw", "ridge_residual", "extra_trees_residual"]
    sources = sorted(train["source"].unique())
    oof_predictions = {name: np.full(len(train), np.nan) for name in candidates}
    for held in sources:
        fit_mask = ~train["source"].eq(held)
        val_mask = train["source"].eq(held)
        for name in candidates:
            model = build_recovery(name)
            if model is not None:
                model.fit(train.loc[fit_mask, CW_FEATURES], train.loc[fit_mask, "cw_residual_over_h"])
            oof_predictions[name][val_mask.to_numpy()] = recovery_predict(model, train.loc[val_mask])
    rows = []
    for name in candidates:
        require(np.isfinite(oof_predictions[name]).all(), f"non-finite recovery OOF: {name}")
        rows.append({"candidate": name, **recovery_score(train, oof_predictions[name])})
    scores = pd.DataFrame(rows)
    raw = scores.set_index("candidate").loc["identity_raw"]
    eligible = scores[
        scores["candidate"].ne("identity_raw")
        & (scores["macro_source_mae_m"] <= 0.99 * raw["macro_source_mae_m"])
        & (scores["delivery_success_rate"] >= raw["delivery_success_rate"] - 0.005)
    ]
    chosen = (
        "identity_raw"
        if eligible.empty
        else str(eligible.sort_values(["macro_source_mae_m", "candidate"]).iloc[0]["candidate"])
    )
    final_model = build_recovery(chosen)
    if final_model is not None:
        final_model.fit(train[CW_FEATURES], train["cw_residual_over_h"])
    return chosen, final_model, oof_predictions[chosen], scores


def build_classifier(kind: str):
    if kind == "new_calibrated_extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
    if kind == "logistic_sensitivity":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=SEED),
        )
    raise KeyError(kind)


def features_for_target(target: str, include_legacy_prior: bool = False) -> list[str]:
    if target in {"R_valid", "R_H"}:
        return TRC_FEATURES
    if target == "R_CW":
        features = list(dict.fromkeys(TRC_FEATURES + TRAIT_STABILITY_FEATURES))
        if include_legacy_prior:
            features.append(LEGACY_RISK[target])
        return features
    raise KeyError(target)


def source_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("source")["source"].transform("size").to_numpy(float)
    return 1.0 / counts


def fit_base(model: Any, frame: pd.DataFrame, target: str, include_legacy_prior: bool = False) -> Any:
    features = features_for_target(target, include_legacy_prior)
    if isinstance(model, ExtraTreesClassifier):
        model.fit(frame[features], frame[target].astype(int), sample_weight=source_balanced_weights(frame))
    else:
        model.fit(frame[features], frame[target].astype(int))
    return model


def positive_score(model: Any, frame: pd.DataFrame, target: str, include_legacy_prior: bool = False) -> np.ndarray:
    return np.asarray(
        model.predict_proba(frame[features_for_target(target, include_legacy_prior)])[:, 1],
        dtype=float,
    )


def fit_sigmoid(raw_score: np.ndarray, y: np.ndarray) -> Any:
    require(len(np.unique(y)) == 2, "calibration requires both classes")
    calibrator = LogisticRegression(C=1e6, max_iter=5000, random_state=SEED)
    calibrator.fit(raw_score.reshape(-1, 1), y.astype(int))
    return calibrator


def apply_sigmoid(calibrator: Any, raw_score: np.ndarray) -> np.ndarray:
    return np.asarray(calibrator.predict_proba(raw_score.reshape(-1, 1))[:, 1], dtype=float)


def point_count_score(frame: pd.DataFrame) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(frame["log_point_count_scene_robust_z"].to_numpy(float)))


def fit_risk_nested(train: pd.DataFrame, test: pd.DataFrame, target: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sources = sorted(train["source"].unique())
    families = ["extra_trees", "logistic", "point_count"]
    if target == "R_CW":
        families += ["stacked_logistic", "legacy_prior"]
    inner_raw = {family: np.full(len(train), np.nan) for family in families}
    inner_raw["point_count"] = point_count_score(train)
    if target == "R_CW":
        inner_raw["legacy_prior"] = pd.to_numeric(train[LEGACY_RISK[target]]).to_numpy(float)
    for held in sources:
        fit = train[~train["source"].eq(held)]
        val_mask = train["source"].eq(held)
        model_specs = [
            ("extra_trees", "new_calibrated_extra_trees", False),
            ("logistic", "logistic_sensitivity", False),
        ]
        if target == "R_CW":
            model_specs.append(("stacked_logistic", "logistic_sensitivity", True))
        for family, kind, include_prior in model_specs:
            model = fit_base(build_classifier(kind), fit, target, include_prior)
            inner_raw[family][val_mask.to_numpy()] = positive_score(
                model, train.loc[val_mask], target, include_prior
            )
    y_train = train[target].to_numpy(int)
    calibrators = {}
    inner_calibrated = {}
    selection_rows = []
    for family in families:
        require(np.isfinite(inner_raw[family]).all(), f"non-finite inner risk scores: {family}")
        calibrators[family] = fit_sigmoid(inner_raw[family], y_train)
        inner_calibrated[family] = apply_sigmoid(calibrators[family], inner_raw[family])
        selection_rows.append(
            {
                "family": family,
                "inner_aurc": aurc(y_train, inner_calibrated[family]),
                "inner_brier": brier_score_loss(y_train, inner_calibrated[family]),
                "inner_average_precision": average_precision_score(y_train, inner_calibrated[family]),
            }
        )
    selection = pd.DataFrame(selection_rows).sort_values(
        ["inner_aurc", "inner_brier", "family"], kind="stable"
    )
    chosen_family = str(selection.iloc[0]["family"])
    fitted_models = {
        "extra_trees": fit_base(build_classifier("new_calibrated_extra_trees"), train, target),
        "logistic": fit_base(build_classifier("logistic_sensitivity"), train, target),
        "point_count": None,
    }
    if target == "R_CW":
        fitted_models["stacked_logistic"] = fit_base(
            build_classifier("logistic_sensitivity"), train, target, True
        )
        fitted_models["legacy_prior"] = None
    test_raw = {
        "extra_trees": positive_score(fitted_models["extra_trees"], test, target),
        "logistic": positive_score(fitted_models["logistic"], test, target),
        "point_count": point_count_score(test),
    }
    if target == "R_CW":
        test_raw["stacked_logistic"] = positive_score(
            fitted_models["stacked_logistic"], test, target, True
        )
        test_raw["legacy_prior"] = pd.to_numeric(test[LEGACY_RISK[target]]).to_numpy(float)
    calibrated_test = {
        family: apply_sigmoid(calibrators[family], test_raw[family])
        for family in families
    }
    legacy = pd.to_numeric(test[LEGACY_RISK[target]], errors="coerce").to_numpy(float)
    for family, score in calibrated_test.items():
        require(np.isfinite(score).all(), f"non-finite calibrated risk: {family}")
    require(np.isfinite(legacy).all(), "non-finite legacy risk")
    scores = {
        "nested_selected": calibrated_test[chosen_family],
        "calibrated_extra_trees": calibrated_test["extra_trees"],
        "calibrated_logistic": calibrated_test["logistic"],
        "calibrated_point_count": calibrated_test["point_count"],
        "legacy_frozen_202607": legacy,
    }
    if target == "R_CW":
        scores["calibrated_stacked_logistic"] = calibrated_test["stacked_logistic"]
        scores["recalibrated_legacy_prior"] = calibrated_test["legacy_prior"]
    return (
        scores,
        {
            "selected_family": chosen_family,
            "selected_base_model": fitted_models[chosen_family],
            "selected_sigmoid_calibrator": calibrators[chosen_family],
            "candidate_selection": selection.to_dict(orient="records"),
        },
    )


def ece_equal_frequency(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[float, list[dict[str, float]]]:
    order = np.argsort(p, kind="stable")
    groups = np.array_split(order, min(bins, len(order)))
    rows = []
    ece = 0.0
    for i, idx in enumerate(groups):
        if not len(idx):
            continue
        mean_p = float(np.mean(p[idx]))
        rate = float(np.mean(y[idx]))
        weight = len(idx) / len(y)
        ece += weight * abs(mean_p - rate)
        rows.append({"bin": i + 1, "n": len(idx), "mean_predicted_risk": mean_p, "observed_failure_rate": rate})
    return float(ece), rows


def calibration_fit(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=5000)
    model.fit(logit, y.astype(int))
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def aurc(y: np.ndarray, risk: np.ndarray) -> float:
    order = np.argsort(risk, kind="stable")
    cumulative = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
    coverage = np.arange(1, len(y) + 1) / len(y)
    return float(np.trapz(cumulative, coverage))


def metric_row(target: str, model: str, frame: pd.DataFrame, risk: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y = frame[target].to_numpy(int)
    unique = np.unique(y)
    ece, bins = ece_equal_frequency(y, risk)
    intercept, slope = calibration_fit(y, risk) if len(unique) == 2 else (math.nan, math.nan)
    row = {
        "target": target,
        "model": model,
        "n": len(y),
        "failures": int(y.sum()),
        "prevalence": float(np.mean(y)),
        "auroc": float(roc_auc_score(y, risk)) if len(unique) == 2 else math.nan,
        "average_precision": float(average_precision_score(y, risk)) if len(unique) == 2 else math.nan,
        "brier": float(brier_score_loss(y, risk)),
        "log_loss": float(log_loss(y, np.column_stack([1 - risk, risk]), labels=[0, 1])),
        "ece_10_equal_frequency": ece,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "aurc": aurc(y, risk),
    }
    for item in bins:
        item.update(target=target, model=model)
    return row, bins


def budget_rows(target: str, model: str, frame: pd.DataFrame, risk: np.ndarray) -> list[dict[str, Any]]:
    y = frame[target].to_numpy(int)
    order = np.argsort(-risk, kind="stable")
    total_failures = int(y.sum())
    rows = []
    for budget in REVIEW_BUDGETS:
        n_review = int(math.ceil(budget * len(y))) if budget > 0 else 0
        reviewed = order[:n_review]
        accepted = order[n_review:]
        captured = int(y[reviewed].sum()) if n_review else 0
        remaining = int(y[accepted].sum()) if len(accepted) else 0
        capture = captured / total_failures if total_failures else math.nan
        accepted_rate = remaining / len(accepted) if len(accepted) else math.nan
        prevalence = total_failures / len(y)
        relative_reduction = (prevalence - accepted_rate) / prevalence if prevalence > 0 and len(accepted) else math.nan
        rows.append(
            {
                "target": target,
                "model": model,
                "review_budget": budget,
                "n_total": len(y),
                "n_reviewed": n_review,
                "n_accepted": len(accepted),
                "total_failures": total_failures,
                "captured_failures": captured,
                "remaining_failures": remaining,
                "failure_capture": capture,
                "review_precision": captured / n_review if n_review else math.nan,
                "review_lift_vs_random": capture / budget if budget > 0 and total_failures else math.nan,
                "accepted_coverage": len(accepted) / len(y),
                "accepted_failure_rate": accepted_rate,
                "relative_failure_reduction": relative_reduction,
            }
        )
    return rows


def risk_coverage_rows(target: str, model: str, frame: pd.DataFrame, risk: np.ndarray) -> list[dict[str, Any]]:
    y = frame[target].to_numpy(int)
    order = np.argsort(risk, kind="stable")
    rows = []
    for coverage in np.linspace(0.50, 1.00, 51):
        n_accept = max(1, int(math.floor(coverage * len(y))))
        accepted = order[:n_accept]
        rows.append(
            {
                "target": target,
                "model": model,
                "coverage": n_accept / len(y),
                "accepted_failure_rate": float(np.mean(y[accepted])),
            }
        )
    return rows


def bootstrap_metrics(oof: pd.DataFrame, risk_columns: dict[tuple[str, str], str], replicates: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    sources = sorted(oof["source"].unique())
    groups = {
        (source, scene): idx.to_numpy()
        for (source, scene), idx in oof.groupby(["source", "scene"]).groups.items()
    }
    scenes_by_source = {
        source: sorted(oof.loc[oof["source"].eq(source), "scene"].unique())
        for source in sources
    }
    records = []
    for replicate in range(replicates):
        sampled_idx = []
        for source in rng.choice(sources, size=len(sources), replace=True):
            scenes = scenes_by_source[source]
            for scene in rng.choice(scenes, size=len(scenes), replace=True):
                sampled_idx.extend(groups[(source, scene)])
        sample = oof.loc[sampled_idx]
        for (target, model), column in risk_columns.items():
            sub = sample[sample["target_scope"].eq(target)]
            y = sub[target].to_numpy(int)
            p = sub[column].to_numpy(float)
            if len(np.unique(y)) < 2:
                continue
            budget = budget_rows(target, model, sub, p)[3]
            records.append(
                {
                    "replicate": replicate,
                    "target": target,
                    "model": model,
                    "auroc": roc_auc_score(y, p),
                    "average_precision": average_precision_score(y, p),
                    "brier": brier_score_loss(y, p),
                    "aurc": aurc(y, p),
                    "capture_at_20pct_review": budget["failure_capture"],
                    "accepted_failure_rate_at_80pct_coverage": budget["accepted_failure_rate"],
                }
            )
    raw = pd.DataFrame(records)
    summary = []
    for (target, model), group in raw.groupby(["target", "model"]):
        for metric in ["auroc", "average_precision", "brier", "aurc", "capture_at_20pct_review", "accepted_failure_rate_at_80pct_coverage"]:
            values = group[metric].dropna().to_numpy(float)
            summary.append(
                {
                    "target": target,
                    "model": model,
                    "metric": metric,
                    "replicates": len(values),
                    "estimate_median": float(np.median(values)),
                    "ci95_low": float(np.quantile(values, 0.025)),
                    "ci95_high": float(np.quantile(values, 0.975)),
                }
            )
    return pd.DataFrame(summary)


def fit_final_risk(frame: pd.DataFrame, target: str) -> dict[str, Any]:
    _, fitted = fit_risk_nested(frame, frame, target)
    return {
        "schema": "forestformer3d.retrospective_risk_model.v1",
        "role": "future_external_validation_candidate_not_independently_validated_on_official28",
        "target": target,
        "features": features_for_target(
            target, fitted["selected_family"] == "stacked_logistic"
        ),
        "selected_family": fitted["selected_family"],
        "base_model": fitted["selected_base_model"],
        "sigmoid_calibrator": fitted["selected_sigmoid_calibrator"],
        "candidate_selection": fitted["candidate_selection"],
        "seed": SEED,
    }


def save_plots(metrics: pd.DataFrame, calibration: pd.DataFrame, coverage: pd.DataFrame, budgets: pd.DataFrame, recovery: pd.DataFrame, output: Path) -> None:
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    colors = {
        "nested_selected": "#0072B2",
        "calibrated_extra_trees": "#56B4E9",
        "calibrated_logistic": "#009E73",
        "calibrated_point_count": "#999999",
        "calibrated_stacked_logistic": "#CC79A7",
        "recalibrated_legacy_prior": "#E69F00",
        "legacy_frozen_202607": "#D55E00",
    }
    for target in TARGETS:
        fig, ax = plt.subplots(figsize=(6.6, 4.5))
        for model, group in coverage[coverage["target"].eq(target)].groupby("model"):
            ax.plot(group["coverage"], group["accepted_failure_rate"], label=model, color=colors[model])
        ax.set(xlabel="Automatic-delivery coverage", ylabel="Accepted failure rate", title=f"{target}: risk–coverage")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / f"{target}_risk_coverage.png", dpi=220)
        fig.savefig(figures / f"{target}_risk_coverage.svg")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.4, 4.8))
        for model, group in calibration[calibration["target"].eq(target)].groupby("model"):
            ax.plot(group["mean_predicted_risk"], group["observed_failure_rate"], marker="o", label=model, color=colors[model])
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set(xlabel="Predicted risk", ylabel="Observed failure rate", title=f"{target}: calibration")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures / f"{target}_calibration.png", dpi=220)
        fig.savefig(figures / f"{target}_calibration.svg")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 4.5))
        group = budgets[budgets["target"].eq(target)]
        for model, sub in group.groupby("model"):
            ax.plot(100 * sub["review_budget"], 100 * sub["failure_capture"], marker="o", label=model, color=colors[model])
        ax.plot([0, 50], [0, 50], "k--", linewidth=1, label="random expectation")
        ax.set(xlabel="Manual-review budget (%)", ylabel="Failures captured (%)", title=f"{target}: review-budget gain")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures / f"{target}_review_budget.png", dpi=220)
        fig.savefig(figures / f"{target}_review_budget.svg")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    pivot = recovery.groupby("method")[["pooled_mae_m", "delivery_success_rate"]].mean()
    x = np.arange(len(pivot))
    ax.bar(x - 0.18, pivot["pooled_mae_m"], width=0.36, label="MAE (m)", color="#0072B2")
    ax2 = ax.twinx()
    ax2.bar(x + 0.18, pivot["delivery_success_rate"], width=0.36, label="delivery success", color="#E69F00")
    ax.set_xticks(x, pivot.index, rotation=15)
    ax.set_ylabel("CW MAE (m)")
    ax2.set_ylabel("CW delivery success")
    ax.set_title("Nested source-held-out CW recovery")
    fig.tight_layout()
    fig.savefig(figures / "CW_recovery_comparison.png", dpi=220)
    fig.savefig(figures / "CW_recovery_comparison.svg")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir()
    (output / "models").mkdir()

    pred, truth, scenes = load_panels(args.metrics.resolve())
    trait = matched_trait_panel(pred)
    pred_scored = pred[pred["trc_status"].eq("scored_frozen_models")].copy()
    sources = sorted(pred["source"].unique())
    oof_parts = []
    recovery_rows = []
    recovery_selection_rows = []
    risk_selection_rows = []

    for held_source in sources:
        valid_train = pred_scored[~pred_scored["source"].eq(held_source)].copy()
        valid_test = pred_scored[pred_scored["source"].eq(held_source)].copy()
        valid_scores, valid_fit = fit_risk_nested(valid_train, valid_test, "R_valid")
        for item in valid_fit["candidate_selection"]:
            risk_selection_rows.append(
                {"outer_held_source": held_source, "target": "R_valid", "selected_family": valid_fit["selected_family"], **item}
            )
        valid_test["target_scope"] = "R_valid"
        for model, score in valid_scores.items():
            valid_test[f"risk__{model}"] = score
        oof_parts.append(valid_test)

        trait_train = trait[~trait["source"].eq(held_source)].copy()
        trait_test = trait[trait["source"].eq(held_source)].copy()
        chosen, recovery_model, train_cw_oof, selection = fit_recovery_nested(trait_train)
        test_cw = recovery_predict(recovery_model, trait_test)
        trait_train["CW_selected_m"] = train_cw_oof
        trait_test["CW_selected_m"] = test_cw
        trait_train["R_CW"] = 1 - cw_success(train_cw_oof, trait_train["CW_truth_m"].to_numpy(float)).astype(int)
        trait_test["R_CW"] = 1 - cw_success(test_cw, trait_test["CW_truth_m"].to_numpy(float)).astype(int)
        for _, row in selection.iterrows():
            recovery_selection_rows.append({"outer_held_source": held_source, "chosen": chosen, **row.to_dict()})
        for method, prediction_values in [
            ("raw_identity", trait_test["CW_raw_m"].to_numpy(float)),
            ("nested_selected", test_cw),
        ]:
            score = recovery_score(trait_test, prediction_values)
            recovery_rows.append({"held_source": held_source, "method": method, "chosen_candidate": chosen, "n": len(trait_test), **score})

        trait_train_risk = trait_train[trait_train["trc_status"].eq("scored_frozen_models")].copy()
        trait_test_risk = trait_test[trait_test["trc_status"].eq("scored_frozen_models")].copy()
        for target in ["R_H", "R_CW"]:
            scores, fitted = fit_risk_nested(trait_train_risk, trait_test_risk, target)
            for item in fitted["candidate_selection"]:
                risk_selection_rows.append(
                    {"outer_held_source": held_source, "target": target, "selected_family": fitted["selected_family"], **item}
                )
            part = trait_test_risk.copy()
            part["target_scope"] = target
            for model, score in scores.items():
                part[f"risk__{model}"] = score
            oof_parts.append(part)

    oof = pd.concat(oof_parts, ignore_index=True, sort=False)
    require(set(oof["target_scope"]) == set(TARGETS), "missing target scope")
    metric_rows = []
    calibration_rows = []
    budget_table = []
    coverage_table = []
    risk_columns: dict[tuple[str, str], str] = {}
    for target in TARGETS:
        frame = oof[oof["target_scope"].eq(target)].copy()
        risk_models = [
            "nested_selected",
            "calibrated_extra_trees",
            "calibrated_logistic",
            "calibrated_point_count",
            "legacy_frozen_202607",
        ]
        if target == "R_CW":
            risk_models += ["calibrated_stacked_logistic", "recalibrated_legacy_prior"]
        for model in risk_models:
            column = f"risk__{model}"
            risk = frame[column].to_numpy(float)
            row, bins = metric_row(target, model, frame, risk)
            metric_rows.append(row)
            calibration_rows.extend(bins)
            budget_table.extend(budget_rows(target, model, frame, risk))
            coverage_table.extend(risk_coverage_rows(target, model, frame, risk))
            risk_columns[(target, model)] = column

    metrics = pd.DataFrame(metric_rows)
    calibration = pd.DataFrame(calibration_rows)
    budgets = pd.DataFrame(budget_table)
    coverage = pd.DataFrame(coverage_table)
    recovery = pd.DataFrame(recovery_rows)
    selection = pd.DataFrame(recovery_selection_rows)
    risk_selection = pd.DataFrame(risk_selection_rows)
    bootstrap = bootstrap_metrics(oof, risk_columns, args.bootstrap_replicates)

    scenes.to_csv(output / "tables" / "scene_omission_guard.csv", index=False)
    metrics.to_csv(output / "tables" / "risk_metrics.csv", index=False)
    calibration.to_csv(output / "tables" / "calibration_bins.csv", index=False)
    budgets.to_csv(output / "tables" / "review_budget_table.csv", index=False)
    coverage.to_csv(output / "tables" / "risk_coverage_curve.csv", index=False)
    recovery.to_csv(output / "tables" / "cw_recovery_outer_folds.csv", index=False)
    selection.to_csv(output / "tables" / "cw_recovery_inner_selection.csv", index=False)
    risk_selection.to_csv(output / "tables" / "risk_model_inner_selection.csv", index=False)
    bootstrap.to_csv(output / "tables" / "bootstrap_ci.csv", index=False)
    oof.to_csv(output / "tables" / "source_heldout_predictions.csv", index=False)

    final_chosen, final_recovery, final_cw_oof, final_selection = fit_recovery_nested(trait.copy())
    final_trait = trait.copy()
    final_trait["R_CW"] = 1 - cw_success(final_cw_oof, final_trait["CW_truth_m"].to_numpy(float)).astype(int)
    joblib.dump(
        {
            "schema": "forestformer3d.retrospective_cw_recovery.v1",
            "role": "future_external_validation_candidate_not_independently_validated_on_official28",
            "chosen_candidate": final_chosen,
            "features": CW_FEATURES,
            "model": final_recovery,
            "identity_if_model_is_none": True,
            "seed": SEED,
        },
        output / "models" / "cw_recovery_candidate.joblib",
    )
    joblib.dump(fit_final_risk(pred_scored.copy(), "R_valid"), output / "models" / "R_valid_candidate.joblib")
    joblib.dump(
        fit_final_risk(trait[trait["trc_status"].eq("scored_frozen_models")].copy(), "R_H"),
        output / "models" / "R_H_candidate.joblib",
    )
    joblib.dump(
        fit_final_risk(final_trait[final_trait["trc_status"].eq("scored_frozen_models")].copy(), "R_CW"),
        output / "models" / "R_CW_candidate.joblib",
    )
    final_selection.to_csv(output / "tables" / "cw_recovery_final_selection.csv", index=False)

    save_plots(metrics, calibration, coverage, budgets, recovery, output)
    primary = metrics[metrics["model"].eq("nested_selected")].set_index("target")
    budget20 = budgets[
        budgets["model"].eq("nested_selected")
        & budgets["review_budget"].eq(0.20)
    ].set_index("target")
    recovery_summary = recovery.groupby("method").agg(
        outer_source_folds=("held_source", "count"),
        mean_fold_mae_m=("pooled_mae_m", "mean"),
        mean_fold_delivery_success=("delivery_success_rate", "mean"),
    )
    summary = {
        "schema": "forestformer3d.trc_cw_retraining.result.v1",
        "status": "complete",
        "study_role": "retrospective_source_heldout_development_not_new_official28_test",
        "metrics_sha256": sha256(args.metrics.resolve()),
        "sources": sources,
        "arms": ARMS,
        "primary_risk_results": primary[["n", "failures", "prevalence", "auroc", "average_precision", "brier", "ece_10_equal_frequency", "aurc"]].to_dict(orient="index"),
        "primary_20pct_review": budget20[["failure_capture", "review_precision", "review_lift_vs_random", "accepted_failure_rate", "relative_failure_reduction"]].to_dict(orient="index"),
        "cw_recovery": recovery_summary.to_dict(orient="index"),
        "final_cw_candidate": final_chosen,
        "missed_tree_boundary": "instance-level R_valid cannot detect reference trees with no prediction; see scene_omission_guard.csv",
        "claim_boundary": "internal source-held-out transfer evidence only; independent external validation required before automatic deployment",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {}
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        manifest[str(path.relative_to(output)).replace("\\", "/")] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    (output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

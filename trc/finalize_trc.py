"""Freeze the confidence-enhanced TRC v7 result bundle.

R_valid is retained from the calibrated v6 source-held-out model. R_H and
R_CW are re-trained on the current FOR-instance evidence with pre-GT sealed
instance confidence, strict outer leave-one-physical-source-out evaluation,
and inner source-held-out probability calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import run_source_heldout_trc_cw_v6 as v6


CONF = [
    "instance_confidence",
    "confidence_scene_robust_z",
    "confidence_scene_percentile",
    "confidence_gap_to_lower",
    "confidence_gap_to_higher",
]
KEYS = ["arm", "scene", "pred_instance_id"]
FIXED = {"R_H": "shallow_extra_trees", "R_CW": "robust_linear"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def attach_confidence(frame: pd.DataFrame, conf: pd.DataFrame) -> pd.DataFrame:
    out = frame.drop(columns=[c for c in CONF if c in frame], errors="ignore").merge(
        conf[KEYS + CONF], on=KEYS, how="left", validate="one_to_one"
    )
    v6.require(not out[CONF].isna().any().any(), "sealed confidence mapping incomplete")
    return out


def fixed_fold(train: pd.DataFrame, test: pd.DataFrame, target: str, family: str):
    inner = np.full(len(train), np.nan)
    for held in sorted(train.source.unique()):
        fit = train[~train.source.eq(held)]
        mask = train.source.eq(held)
        model = v6.fit_model(v6.build_model(target, family), fit, target)
        inner[mask.to_numpy()] = v6.raw_predict(model, train.loc[mask], target)
    v6.require(np.isfinite(inner).all(), f"nonfinite inner score: {target}")
    calibrator = v6.fit_calibrator(inner, train[target].to_numpy(int))
    direction = 1.0 if float(calibrator.coef_[0, 0]) >= 0 else -1.0
    model = v6.fit_model(v6.build_model(target, family), train, target)
    raw = v6.raw_predict(model, test, target)
    return direction * raw, v6.probability(calibrator, raw), model, calibrator, direction


def per_source_tables(oof: pd.DataFrame):
    rows = []
    for (target, source), frame in oof.groupby(["target_scope", "source"], sort=True):
        y = frame[target].to_numpy(int)
        score = frame.rank_score.to_numpy(float)
        prob = frame.calibrated_probability.to_numpy(float)
        b20 = v6.budget_row(frame, target, "primary", score, 0.20)
        rows.append({
            "target": target, "source": source, "n": len(frame), "failures": int(y.sum()),
            "prevalence": float(y.mean()),
            "auroc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
            "average_precision": average_precision_score(y, score) if y.sum() else np.nan,
            "brier": brier_score_loss(y, prob),
            "operational_aurc": v6.operational_aurc(frame, y, score),
            "actual_review_fraction_20": b20["actual_review_fraction"],
            "failure_capture_20": b20["failure_capture"],
            "review_precision_20": b20["review_precision"],
            "review_lift_actual_20": b20["review_lift_vs_random_actual"],
        })
    detail = pd.DataFrame(rows)
    macro = detail.groupby("target", sort=True).agg(
        sources=("source", "count"),
        sources_with_failures=("failures", lambda x: int((x > 0).sum())),
        macro_auroc=("auroc", "mean"), macro_average_precision=("average_precision", "mean"),
        macro_brier=("brier", "mean"), macro_operational_aurc=("operational_aurc", "mean"),
        macro_capture_20=("failure_capture_20", "mean"),
        macro_review_precision_20=("review_precision_20", "mean"),
    ).reset_index()
    return detail, macro


def bootstrap_all(oof: pd.DataFrame, replicates: int):
    """Hierarchical source->scene bootstrap with arms clustered in each scene."""
    rng = np.random.default_rng(v6.SEED)
    source_scenes = {s: sorted(oof.loc[oof.source.eq(s), "scene"].unique())
                     for s in sorted(oof.source.unique())}
    metric_draws, budget_draws = [], []
    for replicate in range(replicates):
        pieces = []
        for source_draw, source in enumerate(rng.choice(list(source_scenes), size=len(source_scenes), replace=True)):
            scenes = source_scenes[source]
            for scene_draw, scene in enumerate(rng.choice(scenes, size=len(scenes), replace=True)):
                z = oof[(oof.source.eq(source)) & (oof.scene.eq(scene))].copy()
                z["scene"] = z.scene.astype(str) + f"__boot{source_draw}_{scene_draw}"
                pieces.append(z)
        sample = pd.concat(pieces, ignore_index=True)
        for target in v6.TARGETS:
            frame = sample[sample.target_scope.eq(target)].copy()
            y = frame[target].to_numpy(int)
            if len(np.unique(y)) < 2:
                continue
            score = frame.rank_score.to_numpy(float)
            prob = frame.calibrated_probability.to_numpy(float)
            vals = {
                "auroc": roc_auc_score(y, score),
                "average_precision": average_precision_score(y, score),
                "brier": brier_score_loss(y, prob),
                "operational_aurc": v6.operational_aurc(frame, y, score),
            }
            for metric, value in vals.items():
                metric_draws.append({"replicate": replicate, "target": target, "metric": metric, "value": value})
            for nominal in v6.v5.REVIEW_BUDGETS:
                row = v6.budget_row(frame, target, "primary_v7", score, nominal)
                for metric in ["actual_review_fraction", "failure_capture", "review_precision",
                               "review_lift_vs_random_actual", "accepted_coverage", "accepted_failure_rate"]:
                    budget_draws.append({"replicate": replicate, "target": target,
                                         "nominal_review_fraction": nominal, "metric": metric,
                                         "value": row[metric]})
    raw_metric = pd.DataFrame(metric_draws)
    raw_budget = pd.DataFrame(budget_draws)

    def summarize(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        rows = []
        for key, group in frame.groupby(keys, sort=True, dropna=False):
            values = group.value.dropna().to_numpy(float)
            item = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
            item.update(valid_replicates=len(values), estimate_median=np.median(values) if len(values) else np.nan,
                        ci95_low=np.quantile(values, .025) if len(values) else np.nan,
                        ci95_high=np.quantile(values, .975) if len(values) else np.nan)
            rows.append(item)
        return pd.DataFrame(rows)

    return (summarize(raw_metric, ["target", "metric"]),
            summarize(raw_budget, ["target", "nominal_review_fraction", "metric"]),
            raw_metric, raw_budget)


def calibration_plots(bins: pd.DataFrame, out: Path) -> None:
    directory = out / "figures"
    directory.mkdir(exist_ok=True)
    for target in v6.TARGETS:
        frame = bins[bins.target.eq(target)].sort_values("mean_predicted_risk")
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
        ax.plot(frame.mean_predicted_risk, frame.observed_failure_rate, "o-", label=target)
        ax.set(xlabel="Mean predicted risk", ylabel="Observed failure frequency",
               xlim=(0, 1), ylim=(0, 1), title=f"{target}: probability calibration")
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
        fig.savefig(directory / f"{target}_calibration.png", dpi=220)
        fig.savefig(directory / f"{target}_calibration.svg")
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--confidence", type=Path, required=True)
    ap.add_argument("--v6-output", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing overwrite: {out}")
    for directory in [out, out / "tables", out / "models", out / "src"]:
        directory.mkdir(parents=True, exist_ok=True)

    pred, truth, scenes, trait = v6.prepare(args.metrics.resolve())
    scored = pred[pred.trc_status.eq("scored_frozen_models")].copy()
    conf = pd.read_csv(args.confidence.resolve())
    if "pred_instance_id" not in conf:
        conf["pred_instance_id"] = conf.las_instance_id.astype(int) + 1
    scored_conf = attach_confidence(scored, conf)
    trait_conf = attach_confidence(trait, conf)
    v6.require(len(scored_conf) == 4652, "unexpected scored-prediction denominator")
    for target in FIXED:
        v6.FEATURES[target] = list(dict.fromkeys(v6.FEATURES[target] + CONF))

    # Preserve v6 R_valid OOF scores exactly, joined by stable prediction identity.
    prior = pd.read_csv(args.v6_output / "tables" / "source_heldout_predictions.csv")
    prior = prior[prior.target_scope.eq("R_valid")][KEYS + ["rank_score", "calibrated_probability", "selected_family"]]
    valid = scored_conf.merge(prior, on=KEYS, how="left", validate="one_to_one")
    v6.require(valid[["rank_score", "calibrated_probability"]].notna().all().all(), "R_valid OOF join failed")
    valid["target_scope"] = "R_valid"

    parts = [valid]
    score_map = {"R_valid": dict(zip(valid.row_id.astype(int), valid.rank_score.astype(float))), "R_H": {}, "R_CW": {}}
    fold_rows = []
    for held in sorted(scored_conf.source.unique()):
        train = trait_conf[(~trait_conf.source.eq(held)) & trait_conf.trc_status.eq("scored_frozen_models")].copy()
        test_all = scored_conf[(scored_conf.source.eq(held)) & scored_conf.trait_status.eq("estimable")].copy()
        eval_trait = trait_conf[(trait_conf.source.eq(held)) & trait_conf.trc_status.eq("scored_frozen_models")].copy()
        for target, family in FIXED.items():
            rank, prob, _, _, direction = fixed_fold(train, test_all, target, family)
            score_map[target].update(dict(zip(test_all.row_id.astype(int), rank)))
            scored_test = test_all[KEYS + ["row_id"]].copy()
            scored_test["rank_score"] = rank
            scored_test["calibrated_probability"] = prob
            take = eval_trait.merge(scored_test, on=KEYS, how="left", validate="one_to_one", suffixes=("", "_score"))
            v6.require(take[["rank_score", "calibrated_probability"]].notna().all().all(), f"{target} OOF join failed")
            take["target_scope"] = target
            take["selected_family"] = family
            parts.append(take)
            fold_rows.append({"outer_held_source": held, "target": target, "fixed_family": family,
                              "train_n": len(train), "test_n": len(eval_trait), "score_direction": direction})

    oof = pd.concat(parts, ignore_index=True, sort=False)
    metrics, bins, budgets, coverage = [], [], [], []
    for target in v6.TARGETS:
        frame = oof[oof.target_scope.eq(target)].copy()
        score = frame.rank_score.to_numpy(float)
        prob = frame.calibrated_probability.to_numpy(float)
        row, target_bins = v6.metric_row(frame, target, "primary_v7", score, prob)
        metrics.append(row); bins.extend(target_bins)
        for budget in v6.v5.REVIEW_BUDGETS:
            budgets.append(v6.budget_row(frame, target, "primary_v7", score, budget))
        coverage.extend(v6.coverage_rows(frame, target, "primary_v7", score))
    metrics = pd.DataFrame(metrics)
    bins = pd.DataFrame(bins)
    budgets = pd.DataFrame(budgets)
    coverage = pd.DataFrame(coverage)
    cascade, cascade_ledger = v6.cascade_table(pred, truth, score_map)
    boot_input = oof.copy()
    bootstrap, budget_bootstrap, raw_bootstrap, raw_budget_bootstrap = bootstrap_all(boot_input, args.bootstrap_replicates)
    bootstrap["model"] = "primary_v7"
    per_source, source_macro = per_source_tables(oof)
    audit = bootstrap[["target", "metric", "valid_replicates"]].copy()
    audit["requested_replicates"] = args.bootstrap_replicates
    audit["undefined_replicates"] = audit.requested_replicates - audit.valid_replicates

    scenes.to_csv(out / "tables" / "scene_omission_guard.csv", index=False)
    metrics.to_csv(out / "tables" / "risk_metrics.csv", index=False)
    bins.to_csv(out / "tables" / "calibration_bins.csv", index=False)
    budgets.to_csv(out / "tables" / "review_budget_table.csv", index=False)
    coverage.to_csv(out / "tables" / "risk_coverage_curve_full.csv", index=False)
    cascade.to_csv(out / "tables" / "cascade_budget_table_full_truth.csv", index=False)
    cascade_ledger.to_csv(out / "tables" / "cascade_truth_ledger.csv", index=False)
    oof.to_csv(out / "tables" / "source_heldout_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "tables" / "outer_fold_audit.csv", index=False)
    bootstrap.to_csv(out / "tables" / "bootstrap_ci.csv", index=False)
    budget_bootstrap.to_csv(out / "tables" / "review_budget_bootstrap_ci.csv", index=False)
    raw_bootstrap.to_csv(out / "tables" / "bootstrap_metric_draws.csv", index=False)
    raw_budget_bootstrap.to_csv(out / "tables" / "bootstrap_budget_draws.csv", index=False)
    audit.to_csv(out / "tables" / "bootstrap_draw_audit.csv", index=False)
    per_source.to_csv(out / "tables" / "per_source_risk_metrics.csv", index=False)
    source_macro.to_csv(out / "tables" / "source_macro_risk_summary.csv", index=False)
    conf.to_csv(out / "tables" / "sealed_instance_confidence.csv", index=False)

    # R_valid remains the already-frozen v6 candidate; fit enhanced trait heads on all current data.
    shutil.copy2(args.v6_output / "models" / "R_valid_candidate.joblib", out / "models" / "R_valid_candidate.joblib")
    for target, family in FIXED.items():
        _, _, model, calibrator, direction = fixed_fold(trait_conf, trait_conf, target, family)
        joblib.dump({
            "schema": "forestformer3d.trc_risk_model.v3",
            "role": "manual_review_prioritization_internal_validation_only",
            "target": target, "features": v6.FEATURES[target], "selected_family": family,
            "rank_model": model, "rank_score_direction": direction,
            "probability_calibrator": calibrator,
            "score_target": "S_H" if target == "R_H" else "S_CW",
            "confidence_provenance": "sealed truth-free round2 LAS score generated before GT opening",
            "operational_unit": "arm_x_scene", "tie_break": "pred_instance_id", "seed": v6.SEED,
            "legacy_prior_in_primary_selection": False,
        }, out / "models" / f"{target}_candidate.joblib")

    # Plot helper expects its historical model label.
    plot_metrics = metrics.assign(model="nested_selected")
    plot_coverage = coverage.assign(model="nested_selected")
    plot_budgets = budgets.assign(model="nested_selected")
    v6.save_plots(plot_metrics, plot_coverage, plot_budgets, cascade, pd.DataFrame(), out)
    calibration_plots(bins, out)
    shutil.copy2(args.protocol, out / "protocol_frozen.json")
    for src in [Path(__file__), Path(__file__).with_name("extract_sealed_instance_confidence.py"),
                Path(__file__).with_name("score_trc_risk_bundle.py")]:
        shutil.copy2(src, out / "src" / src.name)

    b20 = budgets[np.isclose(budgets.nominal_review_fraction, 0.20)].set_index("target")
    summary = {
        "schema": "forestformer3d.trc_confidence_v7.result.v1", "status": "complete",
        "study_role": "posthoc retrospective official28 development; not independent external validation",
        "models": {"R_valid": "v6 retained", **FIXED},
        "risk_results": metrics.set_index("target").to_dict(orient="index"),
        "review_at_nominal_20pct": b20.to_dict(orient="index"),
        "confidence_mapping": "pred_instance_id = LAS instance_pred + 1",
        "mapped_scored_predictions": len(scored_conf), "truth_denominator": len(truth),
        "prediction_denominator": len(scored_conf), "cw_recovery_in_primary_method": False,
        "legacy_virtual_prior_in_primary_method": False,
        "claim_boundary": "instance-level manual-review prioritization; automatic thresholds require separate real ALS calibration and validation",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {}
    for path in sorted(x for x in out.rglob("*") if x.is_file()):
        manifest[str(path.relative_to(out)).replace("\\", "/")] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    (out / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(b20[["actual_review_fraction", "failure_capture", "review_lift_vs_random_actual"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

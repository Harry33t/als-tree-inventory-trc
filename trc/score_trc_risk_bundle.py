"""Truth-free scorer for ForestFormer3D TRC v6 model bundles."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    if bundle.get("schema") != "forestformer3d.trc_risk_model.v2":
        raise ValueError("unsupported model-bundle schema")
    frame = pd.read_csv(args.input)
    missing = sorted(set(bundle["features"]) - set(frame.columns))
    if missing:
        raise ValueError(f"missing required prediction descriptors: {missing}")
    if len(frame) <= 1:
        frame["fail_closed_manual_review"] = True
    else:
        frame["fail_closed_manual_review"] = False

    model = bundle["rank_model"]
    if model is None:
        z = pd.to_numeric(frame["log_point_count_scene_robust_z"], errors="raise").to_numpy(float)
        raw = 1.0 / (1.0 + np.exp(np.clip(z, -30, 30)))
    elif bundle["target"] == "R_valid":
        raw = np.asarray(model.predict_proba(frame[bundle["features"]])[:, 1], float)
    else:
        raw = np.maximum(0.0, np.asarray(model.predict(frame[bundle["features"]]), float))
    frame["rank_score"] = float(bundle["rank_score_direction"]) * raw
    frame["calibrated_probability"] = bundle["probability_calibrator"].predict_proba(
        raw.reshape(-1, 1)
    )[:, 1]
    frame["risk_target"] = bundle["target"]
    frame["probability_claim"] = (
        "manual_review_ranking_only"
        if bundle["target"] == "R_CW"
        else "internally_calibrated_not_external"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

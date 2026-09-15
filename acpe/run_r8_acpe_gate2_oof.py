#!/usr/bin/env python3
"""Paired, sequential physical-OOF short-training smoke for ACPE Gate2."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import random
import subprocess
import types
from pathlib import Path

os.environ.setdefault("SPCONV_DISABLE_JIT", "1")
os.environ.setdefault("CUMM_DISABLE_JIT", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.runner import Runner
from mmdet3d.registry import DATASETS

from run_r8_acpe_gate1_smoke import (
    acpe_loss, coarse_rows, majority_labels, sha256, set_point_cap)

FOLDS = (0, 1, 2, 3)
ARMS = (("control", 0.0), ("acpe", 0.10))


def atomic_json(path, payload):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def trusted_load_runner(runner):
    original = torch.load

    def trusted(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = trusted
    try:
        runner.load_or_resume()
    finally:
        torch.load = original


def build_acpe_from_capture(bare, capture, points, samples):
    if len(samples) != 1:
        raise RuntimeError("ACPE Gate2 is frozen to batch size one")
    coordinates, _, inverse, _ = bare.collate(points)
    labels = samples[0].gt_pts_seg.pts_instance_mask
    foreground = samples[0].gt_pts_seg.instance_mask
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    if torch.is_tensor(foreground):
        foreground = foreground.detach().cpu().numpy()
    final_labels = majority_labels(
        inverse.detach().cpu().numpy(), labels, foreground, coordinates.shape[0])
    fine_keys = coordinates[:, 1:].detach().cpu().numpy()
    p2_indices = capture["indices"]
    p2_batch = p2_indices[:, 0] == 0
    p2_features = capture["features"][p2_batch]
    p2_keys = p2_indices[p2_batch, 1:].detach().cpu().numpy()
    rows, missing, distance = coarse_rows(fine_keys, p2_keys, 4)
    mapped = p2_features[torch.as_tensor(
        rows, device=p2_features.device, dtype=torch.long)]
    raw, margin, eligibility = acpe_loss(mapped, final_labels)
    return raw, margin, {
        **eligibility,
        "p2_mapping_missing_exact_count": missing,
        "p2_mapping_fallback_max_key_distance": distance,
    }


def install_acpe_training_loss(model, weight):
    bare = model.module if hasattr(model, "module") else model
    base_loss = bare.loss
    capture = {}

    def hook(module, inputs, result):
        sparse = result[0] if isinstance(result, tuple) else result
        capture["features"] = sparse.features
        capture["indices"] = sparse.indices

    handle = bare.unet.u.u.blocks.register_forward_hook(hook)

    def wrapped(this, batch_inputs_dict, batch_data_samples, **kwargs):
        capture.clear()
        losses = base_loss(batch_inputs_dict, batch_data_samples, **kwargs)
        if "features" not in capture:
            raise RuntimeError("P2 hook did not fire during ACPE training")
        raw, margin, audit = build_acpe_from_capture(
            this, capture, batch_inputs_dict["points"], batch_data_samples)
        losses["acpe_loss"] = float(weight) * raw
        # Detached diagnostics are log-only and deliberately excluded from loss
        # parsing because their keys contain no `loss` token.
        losses["acpe_margin"] = margin.detach()
        losses["acpe_eligible_trees"] = raw.new_tensor(
            float(audit["eligible_tree_count"])).detach()
        return losses

    bare.loss = types.MethodType(wrapped, bare)
    bare._acpe_base_loss = base_loss
    bare._acpe_capture = capture
    bare._acpe_hook_handle = handle
    return bare


def configure(base_path, data_root, work_dir, fold, seed, point_cap):
    cfg = Config.fromfile(str(base_path))
    cfg.work_dir = str(work_dir)
    cfg.randomness = dict(seed=seed, deterministic=True)
    cfg.train_cfg.max_epochs = 1
    cfg.train_cfg.val_interval = 1000000
    cfg.train_dataloader.batch_size = 1
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    cfg.train_dataloader.pop("prefetch_factor", None)
    dataset = cfg.train_dataloader.dataset
    dataset.data_root = str(data_root)
    dataset.ann_file = f"fold{fold}_train_infos.pkl"
    dataset.filter_empty_gt = False
    set_point_cap(cfg, point_cap)
    for key in ("pts", "pts_instance_mask", "pts_semantic_mask"):
        dataset.data_prefix[key] = str(data_root / dataset.data_prefix[key])
    cfg.default_hooks.checkpoint.interval = 1
    cfg.default_hooks.checkpoint.max_keep_ckpts = 1
    cfg.default_hooks.logger.interval = 1
    cfg.resume = False
    return cfg


def build_holdout_dataset(train_cfg, data_root, fold, point_cap):
    dataset_cfg = copy.deepcopy(train_cfg.train_dataloader.dataset)
    dataset_cfg.data_root = str(data_root)
    dataset_cfg.ann_file = f"fold{fold}_holdout_infos.pkl"
    dataset_cfg.filter_empty_gt = False
    for step in dataset_cfg.pipeline:
        if step.get("type") == "PointSample_":
            step.num_points = int(point_cap)
    for key in ("pts", "pts_instance_mask", "pts_semantic_mask"):
        value = str(dataset_cfg.data_prefix[key])
        if not os.path.isabs(value):
            dataset_cfg.data_prefix[key] = str(data_root / value)
    dataset = DATASETS.build(dataset_cfg)
    dataset.full_init()
    if len(dataset) != 2:
        raise RuntimeError(f"fold {fold} expected two holdout sources, got {len(dataset)}")
    return dataset


@torch.no_grad()
def evaluate_holdout(model, bare, dataset, fold, seed):
    model.eval()
    rows = []
    for source_index in range(len(dataset)):
        crop_seed = int(seed + 1000 * fold + source_index)
        seed_everything(crop_seed)
        sample = dataset[source_index]
        batch = pseudo_collate([sample])
        data = model.data_preprocessor(batch, training=True)
        temporary_capture = {}

        def hook(module, inputs, result):
            sparse = result[0] if isinstance(result, tuple) else result
            temporary_capture["features"] = sparse.features
            temporary_capture["indices"] = sparse.indices

        handle = bare.unet.u.u.blocks.register_forward_hook(hook)
        try:
            base_loss = getattr(bare, "_acpe_base_loss", bare.loss)
            losses = base_loss(data["inputs"], data["data_samples"], epoch=0)
            original_total, original_log = model.parse_losses(losses)
        finally:
            handle.remove()
        raw, margin, audit = build_acpe_from_capture(
            bare, temporary_capture, data["inputs"]["points"], data["data_samples"])
        info = dataset.get_data_info(source_index)
        source = Path(info["lidar_points"]["lidar_path"]).name
        rows.append({
            "fold": fold,
            "source_index": source_index,
            "source": source,
            "crop_seed": crop_seed,
            "raw_acpe_loss": float(raw.cpu()),
            "prototype_margin": float(margin.cpu()),
            "original_total_loss": float(original_total.cpu()),
            "original_loss_terms": {
                key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
                for key, value in original_log.items()},
            **audit,
        })
    return rows


def train_and_evaluate(base_config, data_root, output, fold, arm, weight,
                       seed, point_cap):
    arm_root = output / f"fold{fold}_{arm}"
    complete_path = arm_root / "complete.json"
    if complete_path.exists():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        checkpoint = Path(complete["checkpoint"])
        if sha256(checkpoint) != complete["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint signature mismatch: {checkpoint}")
        print(f"ACPE_GATE2_SKIP fold={fold} arm={arm}", flush=True)
        return complete
    arm_root.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    cfg = configure(base_config, data_root, arm_root, fold, seed, point_cap)
    checkpoint = arm_root / "epoch_1.pth"
    if checkpoint.exists():
        cfg.load_from = str(checkpoint)
        print(f"ACPE_GATE2_EVAL_RECOVERY fold={fold} arm={arm}", flush=True)
    else:
        print(f"ACPE_GATE2_TRAIN_START fold={fold} arm={arm}", flush=True)
    runner = Runner.from_cfg(cfg)
    trusted_load_runner(runner)
    model = runner.model
    bare = model.module if hasattr(model, "module") else model
    if weight > 0:
        bare = install_acpe_training_loss(model, weight)
    if not checkpoint.exists():
        runner.train()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        print(f"ACPE_GATE2_TRAIN_DONE fold={fold} arm={arm}", flush=True)
    holdout = build_holdout_dataset(cfg, data_root, fold, point_cap)
    eval_rows = evaluate_holdout(model, bare, holdout, fold, seed)
    complete = {
        "fold": fold,
        "arm": arm,
        "acpe_weight": weight,
        "seed": seed,
        "epochs": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "holdout": eval_rows,
        "validation_accessed": False,
        "locked_test_accessed": False,
    }
    atomic_json(complete_path, complete)
    print("ACPE_GATE2_ARM_COMPLETE " + json.dumps(complete, sort_keys=True), flush=True)
    if hasattr(bare, "_acpe_hook_handle"):
        bare._acpe_hook_handle.remove()
    del runner, model, bare, holdout
    gc.collect()
    torch.cuda.empty_cache()
    return complete


def summarize(results):
    folds = []
    for fold in FOLDS:
        paired = {row["arm"]: row for row in results if row["fold"] == fold}
        summary = {"fold": fold}
        for arm in ("control", "acpe"):
            holdout = paired[arm]["holdout"]
            summary[f"{arm}_prototype_margin"] = float(np.mean(
                [row["prototype_margin"] for row in holdout]))
            summary[f"{arm}_raw_acpe_loss"] = float(np.mean(
                [row["raw_acpe_loss"] for row in holdout]))
            summary[f"{arm}_original_loss"] = float(np.mean(
                [row["original_total_loss"] for row in holdout]))
        summary["prototype_margin_gain"] = (
            summary["acpe_prototype_margin"] - summary["control_prototype_margin"])
        summary["raw_acpe_loss_change"] = (
            summary["acpe_raw_acpe_loss"] - summary["control_raw_acpe_loss"])
        folds.append(summary)
    positive_margin = [row for row in folds if row["prototype_margin_gain"] > 0]
    lower_loss = [row for row in folds if row["raw_acpe_loss_change"] < 0]
    macro_margin_gain = float(np.mean([row["prototype_margin_gain"] for row in folds]))
    macro_loss_change = float(np.mean([row["raw_acpe_loss_change"] for row in folds]))
    original_ratio = float(np.mean([row["acpe_original_loss"] for row in folds]) /
                           np.mean([row["control_original_loss"] for row in folds]))
    conditions = {
        "complete_training_runs": len(results) == 8,
        "complete_paired_holdout_evaluations":
            sum(len(row["holdout"]) for row in results) == 16,
        "minimum_positive_prototype_margin_folds": len(positive_margin) >= 3,
        "macro_prototype_margin_gain_strictly_positive": macro_margin_gain > 0,
        "minimum_raw_acpe_loss_improved_folds": len(lower_loss) >= 3,
        "macro_raw_acpe_loss_change_strictly_negative": macro_loss_change < 0,
        "maximum_original_holdout_loss_ratio": original_ratio <= 1.05,
        "validation_not_accessed": True,
        "locked_test_not_accessed": True,
    }
    return {
        "schema_version": 1,
        "experiment": "forestformer3d_r8_acpe_gate2_physical_oof_smoke",
        "decision": "PASS" if all(conditions.values()) else "FAIL",
        "formal_multiseed_authorized": all(conditions.values()),
        "conditions": conditions,
        "folds": folds,
        "positive_prototype_margin_folds": len(positive_margin),
        "raw_acpe_loss_improved_folds": len(lower_loss),
        "macro_prototype_margin_gain": macro_margin_gain,
        "macro_raw_acpe_loss_change": macro_loss_change,
        "macro_original_holdout_loss_ratio": original_ratio,
        "validation_accessed": False,
        "locked_test_accessed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--point-cap", type=int, default=320000)
    args = parser.parse_args()
    base_config = Path(args.config).resolve()
    preregistration = Path(args.preregistration).resolve()
    data_root = Path(args.data_root).resolve() / "data"
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "experiment": "forestformer3d_r8_acpe_gate2_physical_oof_smoke",
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "config": str(base_config),
        "config_sha256": sha256(base_config),
        "preregistration": str(preregistration),
        "preregistration_sha256": sha256(preregistration),
        "data_root": str(data_root),
        "data_manifest_sha256": sha256(data_root.parent / "manifest.json"),
        "folds": list(FOLDS),
        "arms": [name for name, _ in ARMS],
        "seed": args.seed,
        "epochs": 1,
        "point_cap": args.point_cap,
        "validation_accessed": False,
        "locked_test_accessed": False,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("config_sha256", "preregistration_sha256",
                    "data_manifest_sha256", "folds", "arms", "seed",
                    "epochs", "point_cap"):
            if existing[key] != manifest[key]:
                raise RuntimeError(f"manifest mismatch: {key}")
    else:
        atomic_json(manifest_path, manifest)
    results = []
    for fold in FOLDS:
        for arm, weight in ARMS:
            results.append(train_and_evaluate(
                base_config, data_root, output, fold, arm, weight,
                args.seed, args.point_cap))
    report = summarize(results)
    atomic_json(output / "gate_report.json", report)
    print("ACPE_GATE2_COMPLETE " + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

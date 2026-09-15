#!/usr/bin/env python3
"""Preregistered one-batch, zero-update ACPE gradient smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

os.environ.setdefault("SPCONV_DISABLE_JIT", "1")
os.environ.setdefault("CUMM_DISABLE_JIT", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import Runner
from scipy.spatial import cKDTree


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def majority_labels(inverse, labels, foreground, voxel_count):
    inverse = np.asarray(inverse, np.int64)
    labels = np.asarray(labels, np.int64)
    foreground = np.asarray(foreground, bool)
    pairs = np.column_stack((inverse[foreground], labels[foreground]))
    output = np.full(int(voxel_count), -1, np.int64)
    if not len(pairs):
        return output
    unique, counts = np.unique(pairs, axis=0, return_counts=True)
    order = np.lexsort((unique[:, 1], -counts, unique[:, 0]))
    ranked = unique[order]
    first = np.concatenate(([True], ranked[1:, 0] != ranked[:-1, 0]))
    output[ranked[first, 0]] = ranked[first, 1]
    return output


def coarse_rows(fine_keys, coarse_keys, scale=4):
    expected = np.floor_divide(np.asarray(fine_keys, np.int64), int(scale))
    lookup = {tuple(row): index
              for index, row in enumerate(np.asarray(coarse_keys).tolist())}
    rows = np.asarray([lookup.get(tuple(row), -1)
                       for row in expected.tolist()], np.int64)
    missing = rows < 0
    maximum_distance = 0.0
    if np.any(missing):
        tree = cKDTree(np.asarray(coarse_keys, np.float64))
        distances, nearest = tree.query(expected[missing].astype(float), k=1)
        rows[missing] = nearest
        maximum_distance = float(np.max(distances))
    return rows, int(np.count_nonzero(missing)), maximum_distance


def acpe_loss(features, labels, minimum=16, temperature=0.10):
    labels_np = np.asarray(labels, np.int64)
    parity_np = np.arange(len(labels_np), dtype=np.int64) & 1
    trees = sorted(int(tree) for tree in np.unique(labels_np) if tree >= 0)
    eligible = [tree for tree in trees
                if all(np.sum((labels_np == tree) & (parity_np == parity)) >= minimum
                       for parity in (0, 1))]
    if len(eligible) < 2:
        raise RuntimeError(f"only {len(eligible)} eligible trees")
    normalized = F.normalize(features.float(), dim=1)
    losses, margins = [], []
    eligible_voxels = 0
    for evaluated_parity in (0, 1):
        support_parity = 1 - evaluated_parity
        prototypes = []
        for tree in eligible:
            mask = torch.as_tensor(
                (labels_np == tree) & (parity_np == support_parity),
                device=features.device)
            prototypes.append(F.normalize(normalized[mask].mean(0), dim=0))
        prototypes = torch.stack(prototypes)
        eval_rows = np.flatnonzero(
            np.isin(labels_np, eligible) & (parity_np == evaluated_parity))
        eligible_voxels += len(eval_rows)
        eval_tensor = torch.as_tensor(eval_rows, device=features.device)
        scores = normalized[eval_tensor] @ prototypes.T
        target = torch.as_tensor(
            [eligible.index(int(labels_np[row])) for row in eval_rows],
            device=features.device, dtype=torch.long)
        correct = scores.gather(1, target[:, None]).squeeze(1)
        rival_scores = scores.masked_fill(
            F.one_hot(target, len(eligible)).bool(), float("-inf"))
        rival = rival_scores.max(1).values
        losses.append(F.softplus((rival - correct) / float(temperature)))
        margins.append(correct - rival)
    loss = torch.cat(losses).mean()
    margin = torch.cat(margins).mean()
    return loss, margin, {
        "eligible_tree_count": len(eligible),
        "eligible_voxel_count": int(eligible_voxels),
        "eligible_trees": eligible,
    }


def set_point_cap(cfg, cap):
    pipeline = cfg.train_dataloader.dataset.pipeline
    for transform in pipeline:
        if transform.get("type") == "PointSample_":
            transform["num_points"] = int(cap)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-data-root", required=True)
    parser.add_argument("--ann-file", default="train8_all_infos.pkl")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--point-cap", type=int, default=320000)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    cfg = Config.fromfile(args.config)
    cfg.work_dir = str(output / "runner_work")
    train_root = Path(args.train_data_root).resolve()
    dataset = cfg.train_dataloader.dataset
    annotation = train_root / args.ann_file
    if not annotation.is_file():
        raise FileNotFoundError(annotation)
    dataset.data_root = str(train_root)
    dataset.ann_file = args.ann_file
    prefix = dataset.data_prefix
    for key in ("pts", "pts_instance_mask", "pts_semantic_mask"):
        resolved = train_root / prefix[key]
        if not resolved.is_dir():
            raise FileNotFoundError(resolved)
        prefix[key] = str(resolved)
    dataset.filter_empty_gt = False
    cfg.randomness = dict(seed=args.seed, deterministic=True)
    cfg.train_dataloader.batch_size = 1
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    cfg.train_dataloader.pop("prefetch_factor", None)
    cfg.train_cfg.max_epochs = 1
    cfg.default_hooks.checkpoint.interval = 1000000
    cfg.default_hooks.logger.interval = 1
    set_point_cap(cfg, args.point_cap)

    checkpoint = Path(cfg.load_from).resolve()
    runner = Runner.from_cfg(cfg)
    original_load = torch.load

    def trusted_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_load(*load_args, **load_kwargs)

    torch.load = trusted_load
    try:
        runner.load_or_resume()
    finally:
        torch.load = original_load
    model = runner.model
    bare = model.module if hasattr(model, "module") else model
    model.train()

    capture = {}

    def hook(module, inputs, result):
        sparse = result[0] if isinstance(result, tuple) else result
        capture["features"] = sparse.features
        capture["indices"] = sparse.indices

    handle = bare.unet.u.u.blocks.register_forward_hook(hook)
    try:
        batch = next(iter(runner.train_dataloader))
        data = model.data_preprocessor(batch, training=True)
        original_losses = model._run_forward(data, epoch=0, mode="loss")
        original_total, original_log = model.parse_losses(original_losses)
    finally:
        handle.remove()
    if "features" not in capture:
        raise RuntimeError("P2 hook did not fire")
    if not torch.isfinite(original_total):
        raise RuntimeError("original loss is non-finite")

    points = data["inputs"]["points"]
    samples = data["data_samples"]
    coordinates, _, inverse, _ = bare.collate(points)
    if len(samples) != 1:
        raise RuntimeError("Gate1 requires batch size one")
    labels = samples[0].gt_pts_seg.pts_instance_mask
    foreground = samples[0].gt_pts_seg.instance_mask
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    if torch.is_tensor(foreground):
        foreground = foreground.detach().cpu().numpy()
    final_labels = majority_labels(
        inverse.detach().cpu().numpy(), labels, foreground, coordinates.shape[0])
    fine_keys = coordinates[:, 1:].detach().cpu().numpy()
    p2_keys = capture["indices"][:, 1:].detach().cpu().numpy()
    rows, missing_exact, fallback_distance = coarse_rows(fine_keys, p2_keys, 4)
    mapped = capture["features"][torch.as_tensor(
        rows, device=capture["features"].device, dtype=torch.long)]

    raw_loss, margin_before, eligibility = acpe_loss(mapped, final_labels)
    weighted_loss = 0.10 * raw_loss
    p2_feature_gradient = torch.autograd.grad(
        raw_loss, capture["features"], retain_graph=True)[0]
    block_parameters = [(name, parameter) for name, parameter
                        in bare.unet.u.u.blocks.named_parameters()
                        if parameter.requires_grad]
    parameter_gradients = torch.autograd.grad(
        weighted_loss, [parameter for _, parameter in block_parameters],
        retain_graph=True, allow_unused=True)
    finite_parameter_count = 0
    nonzero_parameter_count = 0
    for gradient in parameter_gradients:
        if gradient is None:
            continue
        if not torch.isfinite(gradient).all():
            raise RuntimeError("non-finite P2-block parameter gradient")
        finite_parameter_count += 1
        nonzero_parameter_count += int(torch.count_nonzero(gradient) > 0)

    replay_features = mapped.detach().clone().requires_grad_(True)
    replay_loss, replay_margin_before, replay_eligibility = acpe_loss(
        replay_features, final_labels)
    replay_gradient = torch.autograd.grad(replay_loss, replay_features)[0]
    rms = torch.sqrt(torch.mean(replay_gradient.square())).clamp_min(1e-12)
    replay_updated = replay_features.detach() - (0.001 / rms) * replay_gradient
    _, margin_after, after_eligibility = acpe_loss(replay_updated, final_labels)

    conditions = {
        "minimum_eligible_trees": eligibility["eligible_tree_count"] >= 2,
        "minimum_eligible_voxels": eligibility["eligible_voxel_count"] >= 32,
        "finite_positive_raw_loss": bool(torch.isfinite(raw_loss) and raw_loss > 0),
        "finite_positive_weighted_loss": bool(
            torch.isfinite(weighted_loss) and weighted_loss > 0),
        "finite_nonzero_p2_feature_gradient": bool(
            torch.isfinite(p2_feature_gradient).all()
            and torch.count_nonzero(p2_feature_gradient) > 0),
        "finite_nonzero_p2_block_parameter_gradient":
            finite_parameter_count > 0 and nonzero_parameter_count > 0,
        "replay_margin_gain_strictly_positive": bool(margin_after > margin_before),
        "replay_eligible_set_unchanged":
            eligibility == replay_eligibility == after_eligibility,
        "zero_optimizer_steps": True,
        "zero_checkpoint_writes": True,
        "validation_not_accessed": True,
        "locked_test_not_accessed": True,
    }
    decision = "PASS" if all(conditions.values()) else "FAIL"
    report = {
        "schema_version": 1,
        "experiment": "forestformer3d_r8_acpe_gate1_gradient_smoke",
        "decision": decision,
        "gate2_authorized": decision == "PASS",
        "conditions": conditions,
        **eligibility,
        "raw_acpe_loss": float(raw_loss.detach().cpu()),
        "weighted_acpe_loss": float(weighted_loss.detach().cpu()),
        "prototype_margin_before": float(margin_before.detach().cpu()),
        "prototype_margin_after": float(margin_after.detach().cpu()),
        "prototype_margin_gain": float((margin_after - margin_before).detach().cpu()),
        "p2_feature_gradient_l2": float(p2_feature_gradient.norm().detach().cpu()),
        "p2_block_finite_gradient_parameter_count": finite_parameter_count,
        "p2_block_nonzero_gradient_parameter_count": nonzero_parameter_count,
        "p2_mapping_missing_exact_count": missing_exact,
        "p2_mapping_fallback_max_key_distance": fallback_distance,
        "original_total_loss": float(original_total.detach().cpu()),
        "original_loss_terms": {key: float(value) for key, value in original_log.items()},
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "validation_accessed": False,
        "locked_test_accessed": False,
    }
    manifest = {
        "schema_version": 1,
        "experiment": report["experiment"],
        "decision": decision,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "config": str(Path(args.config).resolve()),
        "config_sha256": sha256(args.config),
        "preregistration": str(Path(args.preregistration).resolve()),
        "preregistration_sha256": sha256(args.preregistration),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "train_data_root": str(train_root),
        "ann_file": args.ann_file,
        "ann_file_sha256": sha256(annotation),
        "seed": args.seed,
        "point_cap": args.point_cap,
        "batches": 1,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "validation_accessed": False,
        "locked_test_accessed": False,
    }
    for name, payload in (("gate_report.json", report), ("manifest.json", manifest)):
        temporary = output / (name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        temporary.replace(output / name)
    print("ACPE_GATE1_COMPLETE " + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

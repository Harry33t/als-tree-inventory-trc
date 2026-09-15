#!/usr/bin/env python3
"""Run one frozen official47 matched fine-tuning arm without GT evaluation."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import subprocess
import time
import os
from pathlib import Path

os.environ.setdefault('SPCONV_DISABLE_JIT', '1')
os.environ.setdefault('CUMM_DISABLE_JIT', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner

from run_r8_acpe_fulltrain3 import trusted_load_runner
from run_r8_acpe_gate1_smoke import set_point_cap, sha256
from run_r8_acpe_gate2_oof import install_acpe_training_loss

SEED = 20260819
EPOCHS = 40
POINT_CAP = 320000
LR = 1e-4
ACPE_WEIGHT = 0.10
ANN_FILE = 'official47_all_infos.pkl'
EXPECTED_SCENE_COUNT = 47
EXCLUDED_EXTENSION = 'BlueCat_RN_merged_trees_train.bin'
EXPECTED_ANN_SHA256 = 'e04e35ee1a00b7cc538d1156ba419cf844b9ba820c223865ec12dc5cbccfd6fb'
EXPECTED_MANIFEST_SHA256 = 'eb509af41b592960b21981567e20c55c654398823ee81882f055e617cc50c830'
EXPECTED_PRETRAINED_SHA256 = '01037a648596832238ac72ea2f5eef87ceaf5aeb399e56ff4b760ba1ed1c777e'
ARM_CONFIG = {
    'control': 'configs/oneformer3d_official47_control_40e.py',
    'acpe': 'configs/oneformer3d_official47_acpe_40e.py',
    'epco': 'configs/oneformer3d_official47_epco_40e.py',
    'acpe_epco': 'configs/oneformer3d_official47_acpe_epco_40e.py',
}


def atomic_json(path: Path, payload: dict) -> None:
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def inspect_data(data_root: Path) -> list[str]:
    annotation = data_root / ANN_FILE
    manifest = data_root.parent / 'manifest.json'
    if sha256(annotation) != EXPECTED_ANN_SHA256:
        raise RuntimeError('official47 annotation signature mismatch')
    if sha256(manifest) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError('official47 manifest signature mismatch')
    with annotation.open('rb') as stream:
        payload = pickle.load(stream)
    sources = [row['lidar_points']['lidar_path'] for row in payload['data_list']]
    if len(sources) != EXPECTED_SCENE_COUNT:
        raise RuntimeError(f'expected {EXPECTED_SCENE_COUNT} sources, got {len(sources)}')
    if sources != sorted(sources):
        raise RuntimeError('official47 sources are not in frozen lexical order')
    if EXCLUDED_EXTENSION in sources or any('BlueCat' in source for source in sources):
        raise RuntimeError('BlueCat extension entered official47 training data')
    if any('_val' in source for source in sources):
        raise RuntimeError('validation source entered official47 training data')
    required = {
        'TUWIEN_TUWIEN_train_train.bin',
        'Yuchen_2023_dls_merged_230209_panoptic_train.bin',
    }
    if not required.issubset(sources):
        raise RuntimeError(f'official47 required sources missing: {required - set(sources)}')
    return sources


def configure(path: Path, data_root: Path, output: Path) -> Config:
    cfg = Config.fromfile(str(path))
    meta = dict(cfg.teacher_benchmark)
    if (meta['seed'] != SEED or meta['arm'] not in ARM_CONFIG or
            meta.get('protocol') != 'official47_matched_40epoch_finetune'):
        raise RuntimeError(f'invalid frozen teacher metadata: {meta}')
    cfg.work_dir = str(output)
    cfg.randomness = dict(seed=SEED, deterministic=True)
    cfg.train_cfg.max_epochs = EPOCHS
    cfg.train_cfg.val_interval = 1_000_000
    cfg.train_dataloader.batch_size = 1
    cfg.train_dataloader.num_workers = 6
    cfg.train_dataloader.persistent_workers = True
    cfg.train_dataloader.prefetch_factor = 2
    cfg.optim_wrapper.optimizer.lr = LR
    dataset = cfg.train_dataloader.dataset
    dataset.data_root = str(data_root)
    dataset.ann_file = ANN_FILE
    dataset.filter_empty_gt = False
    set_point_cap(cfg, POINT_CAP)
    for key in ('pts', 'pts_instance_mask', 'pts_semantic_mask'):
        dataset.data_prefix[key] = str(data_root / dataset.data_prefix[key])
    cfg.default_hooks.checkpoint.interval = 5
    cfg.default_hooks.checkpoint.max_keep_ckpts = 2
    cfg.default_hooks.logger.interval = 1
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True, choices=tuple(ARM_CONFIG))
    parser.add_argument('--repo', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    data_root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f'refusing implicit resume/nonempty output: {output}')
    output.mkdir(parents=True, exist_ok=True)
    config_path = repo / ARM_CONFIG[args.arm]
    sources = inspect_data(data_root)
    cfg = configure(config_path, data_root, output)
    pretrained = Path(cfg.load_from).resolve()
    if sha256(pretrained) != EXPECTED_PRETRAINED_SHA256:
        raise RuntimeError('published baseline checkpoint signature mismatch')
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    manifest = {
        'schema_version': 1, 'status': 'preregistered', 'arm': args.arm,
        'protocol': 'official47_matched_40epoch_finetune',
        'code_commit': head, 'config': str(config_path),
        'config_sha256': sha256(config_path), 'seed': SEED, 'epochs': EPOCHS,
        'point_cap': POINT_CAP, 'batch_size': 1, 'num_workers': 6,
        'prefetch_factor': 2, 'learning_rate': LR,
        'pretrained': str(pretrained), 'pretrained_sha256': EXPECTED_PRETRAINED_SHA256,
        'annotation_sha256': EXPECTED_ANN_SHA256,
        'data_manifest_sha256': EXPECTED_MANIFEST_SHA256, 'sources': sources,
        'acpe_enabled': args.arm in ('acpe', 'acpe_epco'),
        'epco_enabled': args.arm in ('epco', 'acpe_epco'),
        'validation_accessed': False, 'locked_test_accessed': False,
        'checkpoint_selection': False,
        'started_at_unix': time.time(),
    }
    atomic_json(output / 'preregistered_manifest.json', manifest)
    seed_everything()
    runner = Runner.from_cfg(cfg)
    trusted_load_runner(runner)
    hook = None
    if manifest['acpe_enabled']:
        bare = install_acpe_training_loss(runner.model, ACPE_WEIGHT)
        hook = bare._acpe_hook_handle
    params = sum(p.numel() for p in runner.model.parameters())
    trainable = sum(p.numel() for p in runner.model.parameters() if p.requires_grad)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    runner.train()
    duration = time.monotonic() - started
    checkpoint = output / f'epoch_{EPOCHS}.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    complete = dict(manifest)
    complete.update(
        status='complete', checkpoint=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint), parameter_count=params,
        trainable_parameter_count=trainable, training_seconds=duration,
        peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        completed_at_unix=time.time())
    atomic_json(output / 'training_complete.json', complete)
    if hook is not None:
        hook.remove()
    print('TEACHER_BENCHMARK_TRAIN_COMPLETE ' + json.dumps(complete, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

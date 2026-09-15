"""Formal EPCO training objective.

EPCO (Equivalence-Preserving Crown-contact Ownership) treats the one-to-many
queries matched to the same training tree as an equivalence class.  It first
forms a query-count-invariant group logit with log-mean-exp, then applies a
local, conservative competition between the correct tree group, geometrically
supported rival tree groups, and background.

Ground-truth instance identities are used only to form the training loss.  The
module has no inference-time input or post-processing rule: trained checkpoints
continue to emit the original query masks and use the official BM2 path.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _logmeanexp(values: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Numerically stable log-mean-exp, invariant to exact query duplication."""
    return torch.logsumexp(values, dim=dim) - math.log(values.shape[dim])


def epco_ownership_loss(
        mask_logits: torch.Tensor,
        gt_ids: torch.Tensor,
        query_xyz: torch.Tensor,
        voxel_xyz: torch.Tensor,
        gt_masks: torch.Tensor,
        contact_sigma_m: float = 0.60,
        min_contact_weight: float = 0.05,
        tau_contact: float = 0.75,
        tau_uncertain: float = 1.50,
        max_points_per_sample: int = 4096):
    """Compute the formal EPCO ownership loss for one scene crop.

    Args:
        mask_logits: Matched query logits with shape ``[Q, P]``.
        gt_ids: Training-only one-to-many tree assignment for each query,
            shape ``[Q]``; ``-1`` denotes background/unmatched.
        query_xyz: Query seed coordinates, shape ``[Q, 3]``.
        voxel_xyz: Voxel coordinates, shape ``[P, 3]``.
        gt_masks: Ground-truth tree masks, shape ``[T, P]``.
        contact_sigma_m: Spatial scale of the continuous crown-contact kernel.
        min_contact_weight: Minimum detached support for a competitive point.
        tau_contact: Softmax temperature at strong crown contact.
        tau_uncertain: Softmax temperature at weak/uncertain contact.
        max_points_per_sample: Deterministic cap using strongest contact points.

    Returns:
        Tuple of scalar loss and a detached audit dictionary.
    """
    if mask_logits.ndim != 2:
        raise ValueError('mask_logits must have shape [Q, P]')
    if gt_ids.ndim != 1 or gt_ids.shape[0] != mask_logits.shape[0]:
        raise ValueError('gt_ids must have shape [Q]')
    if query_xyz.shape != (mask_logits.shape[0], 3):
        raise ValueError('query_xyz must have shape [Q, 3]')
    if voxel_xyz.shape != (mask_logits.shape[1], 3):
        raise ValueError('voxel_xyz must have shape [P, 3]')
    if gt_masks.ndim != 2 or gt_masks.shape[1] != mask_logits.shape[1]:
        raise ValueError('gt_masks must have shape [T, P]')
    if contact_sigma_m <= 0:
        raise ValueError('contact_sigma_m must be positive')
    if not 0 <= min_contact_weight < 1:
        raise ValueError('min_contact_weight must be in [0, 1)')
    if tau_contact <= 0 or tau_uncertain < tau_contact:
        raise ValueError('temperatures require 0 < tau_contact <= tau_uncertain')

    zero = mask_logits.sum() * 0.0
    valid = gt_ids >= 0
    groups = torch.unique(gt_ids[valid])
    if groups.numel() < 2:
        return zero, {
            'selected_points': 0,
            'target_groups': int(groups.numel()),
            'competitive_pairs': 0,
        }

    group_rows = {
        int(group): torch.where(gt_ids == group)[0]
        for group in groups.tolist()
    }
    group_logits = {
        group: _logmeanexp(mask_logits[rows], dim=0)
        for group, rows in group_rows.items()
    }

    # Vertical extents are training geometry, detached from prediction logits.
    z_extents = {}
    for group in group_rows:
        points = torch.where(gt_masks[group].bool())[0]
        if points.numel() == 0:
            continue
        z = voxel_xyz[points, 2]
        z_extents[group] = (z.min(), z.max())

    weighted_loss_sum = zero
    weight_sum = mask_logits.new_zeros(())
    selected_total = 0
    competitive_pairs = 0

    for target in sorted(group_rows):
        target_points = torch.where(gt_masks[target].bool())[0]
        if target_points.numel() == 0 or target not in z_extents:
            continue

        point_xy = voxel_xyz[target_points, :2].float()
        rival_logits = []
        rival_support = []
        target_z_min, target_z_max = z_extents[target]

        for rival in sorted(group_rows):
            if rival == target or rival not in z_extents:
                continue
            rival_z_min, rival_z_max = z_extents[rival]
            vertical_overlap = (
                target_z_max >= rival_z_min and rival_z_max >= target_z_min)
            if not bool(vertical_overlap):
                continue

            seeds_xy = query_xyz[group_rows[rival], :2].float()
            if seeds_xy.numel() == 0:
                continue
            distance = torch.cdist(point_xy, seeds_xy).amin(dim=1)
            geometry = torch.exp(
                -0.5 * (distance / float(contact_sigma_m)).square())
            evidence = torch.sigmoid(
                group_logits[rival][target_points].detach())
            support = (geometry * evidence).detach()
            rival_logits.append(group_logits[rival][target_points])
            rival_support.append(support)
            competitive_pairs += 1

        if not rival_logits:
            continue

        support_bank = torch.stack(rival_support, dim=0)
        contact_weight = support_bank.amax(dim=0).clamp(0.0, 1.0)
        selected = torch.where(contact_weight >= min_contact_weight)[0]
        if selected.numel() == 0:
            continue
        if selected.numel() > max_points_per_sample:
            selected = torch.topk(
                contact_weight, k=max_points_per_sample,
                sorted=False).indices

        correct = group_logits[target][target_points][selected]
        rivals = torch.stack(rival_logits, dim=0)[:, selected]
        supports = support_bank[:, selected]
        rivals = torch.where(
            supports >= min_contact_weight,
            rivals,
            rivals.new_full(rivals.shape, -20.0))
        background = torch.zeros_like(correct).unsqueeze(0)
        competition = torch.cat(
            [correct.unsqueeze(0), rivals, background], dim=0).transpose(0, 1)

        weights = contact_weight[selected]
        temperature = (
            float(tau_uncertain) -
            (float(tau_uncertain) - float(tau_contact)) * weights)
        log_responsibility = F.log_softmax(
            competition / temperature.unsqueeze(1), dim=1)
        point_loss = -log_responsibility[:, 0]
        weighted_loss_sum = weighted_loss_sum + (weights * point_loss).sum()
        weight_sum = weight_sum + weights.sum()
        selected_total += int(selected.numel())

    if selected_total == 0:
        return zero, {
            'selected_points': 0,
            'target_groups': int(groups.numel()),
            'competitive_pairs': competitive_pairs,
        }
    return weighted_loss_sum / weight_sum.clamp_min(1e-6), {
        'selected_points': selected_total,
        'target_groups': int(groups.numel()),
        'competitive_pairs': competitive_pairs,
    }

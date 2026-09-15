import importlib.util
from pathlib import Path

import torch


spec = importlib.util.spec_from_file_location(
    'epco_standalone', Path(__file__).parents[1] / 'oneformer3d' / 'epco.py')
epco = importlib.util.module_from_spec(spec)
spec.loader.exec_module(epco)
_logmeanexp = epco._logmeanexp
epco_ownership_loss = epco.epco_ownership_loss


def _fixture():
    logits = torch.tensor([
        [3.0, 2.0, -1.0, -2.0],
        [2.0, 3.0, -1.0, -2.0],
        [-2.0, -1.0, 2.0, 3.0],
        [-2.0, -1.0, 3.0, 2.0],
    ], requires_grad=True)
    gt_ids = torch.tensor([0, 0, 1, 1])
    query_xyz = torch.tensor([
        [0.0, 0.0, 1.0], [0.1, 0.0, 1.0],
        [0.4, 0.0, 1.0], [0.5, 0.0, 1.0],
    ])
    voxel_xyz = torch.tensor([
        [0.0, 0.0, 1.0], [0.2, 0.0, 1.2],
        [0.4, 0.0, 1.1], [0.6, 0.0, 1.3],
    ])
    gt_masks = torch.tensor([
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ])
    return logits, gt_ids, query_xyz, voxel_xyz, gt_masks


def test_logmeanexp_is_duplicate_invariant():
    values = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    duplicated = values.repeat(4, 1)
    assert torch.allclose(_logmeanexp(values), _logmeanexp(duplicated))


def test_epco_is_finite_and_backpropagates():
    values = _fixture()
    loss, audit = epco_ownership_loss(*values)
    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert audit['selected_points'] > 0
    loss.backward()
    assert values[0].grad is not None
    assert torch.isfinite(values[0].grad).all()
    assert values[0].grad.abs().sum() > 0


def test_epco_requires_at_least_two_groups():
    logits, _, query_xyz, voxel_xyz, gt_masks = _fixture()
    one_group = torch.zeros(4, dtype=torch.long)
    loss, audit = epco_ownership_loss(
        logits, one_group, query_xyz, voxel_xyz, gt_masks[:1])
    assert loss.item() == 0
    assert audit['selected_points'] == 0

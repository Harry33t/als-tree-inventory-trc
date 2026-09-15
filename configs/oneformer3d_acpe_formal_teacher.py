_base_ = './oneformer3d_r8_acpe_formal.py'

# ACPE arm: the runner installs the frozen P2 loss with weight 0.10.
teacher_benchmark = dict(
    arm='acpe',
    seed=20260819,
    initialization='original_published_baseline_checkpoint',
    acpe_enabled=True,
    acpe_loss_weight=0.10,
    epco_enabled=False)

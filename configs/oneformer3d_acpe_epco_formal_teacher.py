_base_ = './oneformer3d_epco_formal_teacher.py'

# ACPE is installed by the existing full-train runner with its frozen weight
# 0.10.  This config contributes EPCO so both objectives train jointly from
# the original published Baseline checkpoint.
teacher_joint_objectives = dict(
    acpe_loss_weight=0.10,
    epco_loss_weight=0.10,
    initialization='original_published_baseline_checkpoint',
    seed=20260819)

teacher_benchmark = dict(
    arm='acpe_epco',
    seed=20260819,
    initialization='original_published_baseline_checkpoint',
    acpe_enabled=True,
    acpe_loss_weight=0.10,
    epco_enabled=True,
    epco_loss_weight=0.10)

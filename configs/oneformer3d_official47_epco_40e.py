_base_ = './oneformer3d_epco_formal_teacher.py'

teacher_benchmark = dict(
    protocol='official47_matched_40epoch_finetune',
    arm='epco',
    seed=20260819,
    initialization='original_published_baseline_checkpoint',
    acpe_enabled=False,
    epco_enabled=True,
    epco_loss_weight=0.10)

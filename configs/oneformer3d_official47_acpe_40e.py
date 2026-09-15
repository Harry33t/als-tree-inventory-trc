_base_ = './oneformer3d_acpe_formal_teacher.py'

teacher_benchmark = dict(
    protocol='official47_matched_40epoch_finetune',
    arm='acpe',
    seed=20260819,
    initialization='original_published_baseline_checkpoint',
    acpe_enabled=True,
    acpe_loss_weight=0.10,
    epco_enabled=False)

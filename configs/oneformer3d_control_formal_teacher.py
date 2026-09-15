_base_ = './oneformer3d_r8_acpe_formal.py'

# Matched fine-tune Control: no ACPE hook and no EPCO ownership loss.
teacher_benchmark = dict(
    arm='control',
    seed=20260819,
    initialization='original_published_baseline_checkpoint',
    acpe_enabled=False,
    epco_enabled=False)

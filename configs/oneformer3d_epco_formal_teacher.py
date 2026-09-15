_base_ = './oneformer3d_r8_acpe_formal.py'

# Frozen teacher-benchmark EPCO arm.  EPCO is applied to the final decoder
# layer only; the official per-query BCE/Dice, query count, NMS, and BM2 remain
# unchanged.  GT is consumed only by this training loss.
model = dict(
    criterion=dict(
        inst_criterion=dict(
            epco_supervision=dict(
                enabled=True,
                loss_weight=0.10,
                contact_sigma_m=0.60,
                min_contact_weight=0.05,
                tau_contact=0.75,
                tau_uncertain=1.50,
                max_points_per_sample=4096))))

teacher_benchmark = dict(
    arm='epco',
    seed=20260819,
    initialization='original_published_baseline_checkpoint',
    acpe_enabled=False,
    epco_enabled=True,
    epco_loss_weight=0.10)

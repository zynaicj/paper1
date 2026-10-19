_base_ = ['./segformer_mit-b0_suim_boundary.py']

# SMBAS input-level ablation C:
#   BoundaryHead input = C1 + C2 + C3
#   C2 and C3 are resized to C1 resolution before concatenation.
model = dict(
    auxiliary_head=dict(
        in_channels=[32, 64, 160],
        in_index=[0, 1, 2],
        loss_decode=dict(loss_weight=1.0)))

# Train from MiT-B0 ImageNet-1K initialization only. Do not fine-tune from the
# saved SUIM baseline checkpoint inherited by the default boundary config.
load_from = None
resume = False
randomness = dict(seed=0, deterministic=False)

work_dir = './work_dirs/module_a/segformer_b0_suim_boundary_c1c2c3'

custom_hooks = [
    dict(
        type='ArchiveExperimentHook',
        out_dir='experiments/module_a/segformer_b0_suim_boundary_c1c2c3',
        config='configs/segformer/segformer_mit-b0_suim_boundary_c1c2c3.py',
        name='SUIM SMBAS C1+C2+C3 Boundary Input Ablation',
        note=(
            'BoundaryHead input-level ablation: use C1+C2+C3 instead of '
            'default C1+C2; train from MiT-B0 ImageNet initialization.'))
]

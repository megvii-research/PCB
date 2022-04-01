_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/lvis_v1_instance.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
model = dict(
    roi_head=dict(
        type='IterativeRoIHead',
        n_iter=3,
        mlp_layers=2,
        iter_loss_weight=[0.2, 0.2, 0.6],
        reg_iter=2,
        use_layer_norm=True,
        bbox_head=dict(num_classes=1203,
            loss_cls=dict(
                    type="EQLv2PCBLoss", num_classes=1203, use_sigmoid=False, loss_weight=1.0, momentum=0.99, start_epoch = 17, alpha_pcb = 0.2, n_iter=3)), 
        mask_head=dict(num_classes=1203)),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            # LVIS allows up to 300
            max_per_img=300)))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
data = dict(train=dict(
                oversample_thr= 0.0,
                dataset=dict(pipeline=train_pipeline))
    )

custom_hooks = [
    dict(type="LossEpochUpdateHook", priority="LOW")
]

checkpoint_config = dict(interval=4)
fp16 = dict(loss_scale='dynamic')
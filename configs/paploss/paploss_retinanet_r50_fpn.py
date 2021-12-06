_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco500_detection_augm.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    bbox_head=dict(type='PAPLossRetinaHead',
    	anchor_generator=dict(scales_per_octave=2),
    	bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
    	loss_bbox=dict(type='GIoULoss', reduction='none'),
        ctrl_points=[[0.0, 0.0, 0.4182, 0.0277, 0.4948, 0.2187, 0.6882, 0.2624, 0.802, 0.6688, 1.0, 1.0],
                     [0.0, 0.0, 0.1279, 0.3696, 0.3056, 0.4708, 0.5198, 0.529, 0.6983, 0.6709, 1.0, 1.0],
                     [0.0, 0.0, 0.0931, 0.0289, 0.3607, 0.5585, 0.4823, 0.7131, 0.6291, 0.8162, 1.0, 1.0],
                     [0.0, 0.0, 0.4914, 0.1499, 0.5492, 0.1664, 0.6259, 0.3051, 0.7945, 0.6195, 1.0, 1.0],
                     [0.0, 0.0, 0.413, 0.0798, 0.4356, 0.3571, 0.5361, 0.7063, 0.8097, 0.9083, 1.0, 1.0]],
        reg_weight=4.1231))
# optimizer
optimizer = dict(type='SGD', lr=0.016, momentum=0.9, weight_decay=0.0001)

# learning policy
lr_config = dict(step=[60, 80])
total_epochs = 100

dist_params = dict(backend='nccl')

checkpoint_config = dict(interval=10)
workflow = [('train', 1)]

test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.5, # set high threshold for fast eval during training, change it back to 0.05 for accurate result
    nms=dict(type='nms', iou_thr=0.6),
    max_per_img=100)
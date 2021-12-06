_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco500_detection_augm.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    rpn_head=dict(
        type='PAPLossRPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        ctrl_points=[[0.0000, 0.0000, 0.9058, 0.0484, 0.9068, 0.5120, 0.9232, 0.7286, 0.9766, 0.9677, 1.0000, 1.0000],
                     [0.0000, 0.0000, 0.3200, 0.1084, 0.3493, 0.2482, 0.4924, 0.3869, 0.7231, 0.6510, 1.0000, 1.0000],
                     [0.0000, 0.0000, 0.0935, 0.0419, 0.2177, 0.1168, 0.5857, 0.3748, 0.7962, 0.6683, 1.0000, 1.0000],
                     [0.0000, 0.0000, 0.0045, 0.3599, 0.1295, 0.4603, 0.7662, 0.5140, 0.8483, 0.5259, 1.0000, 1.0000],
                     [0.0000, 0.0000, 0.1764, 0.1081, 0.5245, 0.4907, 0.6956, 0.5668, 0.7642, 0.8645, 1.0000, 1.0000]],
        reg_weight=1.6950,
        reg_input='giou'),
    roi_head=dict(
        bbox_head=dict(
            type='PAPLossShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            ctrl_points=[[0.0000, 0.0000, 0.3344, 0.0268, 0.5460, 0.2461, 0.8196, 0.6466, 0.9467, 0.8617, 1.0000, 1.0000],
                         [0.0000, 0.0000, 0.0978, 0.2019, 0.4073, 0.3367, 0.6337, 0.4689, 0.7708, 0.6633, 1.0000, 1.0000],
                         [0.0000, 0.0000, 0.2406, 0.0515, 0.3358, 0.5259, 0.6047, 0.6647, 0.8575, 0.7792, 1.0000, 1.0000],
                         [0.0000, 0.0000, 0.2574, 0.2363, 0.5442, 0.5999, 0.7165, 0.6548, 0.8728, 0.8331, 1.0000, 1.0000],
                         [0.0000, 0.0000, 0.1551, 0.2675, 0.4902, 0.3957, 0.7245, 0.5095, 0.8068, 0.8720, 1.0000, 1.0000]],
            reg_weight=7.1324,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            reg_input='giou',)))
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=False,
            ignore_iof_thr=-1),
        sampler=dict(type='PseudoSampler'),
        pos_weight=-1,
        debug=False))
# optimizer
optimizer = dict(type='SGD', lr=0.024, momentum=0.9, weight_decay=0.0001)

# learning policy
lr_config = dict(step=[75, 95])
total_epochs = 100

test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.5, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)
        # set high threshold for fast eval during training, change it back to 0.05 for accurate result
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
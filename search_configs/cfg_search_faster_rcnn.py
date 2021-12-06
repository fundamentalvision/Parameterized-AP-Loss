### model setting
_base_ = [
    '../configs/_base_/models/faster_rcnn_r50_fpn.py',
    '../configs/_base_/datasets/coco500_detection_augm.py',
    '../configs/_base_/schedules/schedule_1x.py',
    '../configs/_base_/default_runtime.py'
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
        ctrl_points=[0.0],
        reg_weight=1.0,
        reg_input='giou'),
    roi_head=dict(
        bbox_head=dict(
            type='PAPLossShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            ctrl_points=[0.0],
            reg_weight=1.0,
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

checkpoint_config = dict(interval=1000)
dist_params = dict(backend='nccl', port=25590)

data_root = 'data/coco/'
data = dict(
    train=dict(ann_file=data_root + 'annotations/search_train2017.json'),
    val=dict(ann_file=data_root + 'annotations/search_val2017.json',
            img_prefix=data_root + 'train2017/'),
    test=dict(ann_file=data_root + 'annotations/search_val2017.json',
             img_prefix=data_root + 'train2017/'))

total_epochs = 1

optimizer = dict(type='SGD', lr=0.024, momentum=0.9, weight_decay=0.0001) # 0.012 for 4gpu * 8

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[])


### search process setting
# num of sample rounds
sample_times = 40

# num of sampled loss functions each round, should be multiple of world_size
num_samples = 8

# num of models on each gpu
num_models_per_gpu = 1

# eps for PPO
clip_eps = 0.1

# lr for updating mu
mu_lr = 0.01

# lr schedule for updating mu
update_per_sample = 100


### search function setting
func_types = ['H2', 'I2', 'I3', 'I1', 'H1']  # function order should correspond to paploss.py
fixed_func_types = []
fixed_ctrl_points = []

search_func_num = len(func_types) * 2  # different heads different losses
num_theta = 8 * search_func_num # one function 12 mu and sigma each, 4 control points

mu_ = [0.2, 0.2, 0.25, 0.25, 0.3333, 0.3333, 0.5, 0.5]
sigma_ = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
constraint_index_ = [True, True, True, True, True, True, True, True]
mu = []
sigma = []
constraint_index = []
for i in range(search_func_num):
    mu = mu + mu_
    sigma = sigma + sigma_
    constraint_index = constraint_index + constraint_index_

# two different gradient scales for two different heads
mu = mu + [0.5] + [0.5]
sigma = sigma + [0.2] + [0.2]
num_theta = num_theta + 2
constraint_index = constraint_index + [True]
### model setting
_base_ = [
    '../configs/_base_/models/retinanet_r50_fpn.py',
    '../configs/_base_/datasets/coco500_detection_augm.py',
    '../configs/_base_/schedules/schedule_1x.py',
    '../configs/_base_/default_runtime.py'
]

model = dict(
    bbox_head=dict(type='PAPLossRetinaHead',
    	anchor_generator=dict(scales_per_octave=2),
    	bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
    	loss_bbox=dict(type='GIoULoss', reduction='none')))

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

optimizer = dict(type='SGD', lr=0.016, momentum=0.9, weight_decay=0.0001) # 0.008 for 4*8
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
func_types = ['H2', 'I2', 'I3', 'I1', 'H1'] # function order should correspond to paploss.py
fixed_func_types = []
fixed_ctrl_points = []

search_func_num = len(func_types)
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

# extra parameter for gradient scale
mu = mu + [0.5]
sigma = sigma + [0.2]
num_theta = num_theta + 1
constraint_index = constraint_index + [True]
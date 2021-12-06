_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco500_detection_augm.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    bbox_head=dict(
    	anchor_generator=dict(scales_per_octave=2),
        reg_decoded_bbox=True,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)))
# optimizer batch_size=8*8
optimizer = dict(type='SGD', lr=0.016, momentum=0.9, weight_decay=0.0001)

# learning policy
lr_config = dict(step=[60, 80])
total_epochs = 100
checkpoint_config = dict(interval=20)
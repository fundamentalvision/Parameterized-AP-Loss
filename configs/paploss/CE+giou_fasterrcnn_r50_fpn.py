_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco500_detection_augm.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            reg_decoded_bbox=True,
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0))))

# optimizer
optimizer = dict(type='SGD', lr=0.04, momentum=0.9, weight_decay=0.0001)

# learning policy
lr_config = dict(step=[75, 95])
total_epochs = 100
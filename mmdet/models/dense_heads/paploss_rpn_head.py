import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import bias_init_with_prob, normal_init

from mmdet.ops import batched_nms
from ..builder import HEADS
from .anchor_head import AnchorHead
from .rpn_test_mixin import RPNTestMixin
from mmdet.models.losses import paploss
from mmdet.core import giou, iou
import collections
import numpy as np


@HEADS.register_module()
class PAPLossRPNHead(RPNTestMixin, AnchorHead):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
    """  # noqa: W605

    def __init__(self, in_channels, ctrl_points=[0.0], reg_weight=2.0, num_topk=1e5, reg_input='giou', loss_form='searched', **kwargs):
        super(PAPLossRPNHead, self).__init__(
            1, in_channels, background_label=0, **kwargs)
        self.ctrl_points = torch.tensor(ctrl_points).cuda()
        self.reg_weight = reg_weight
        self.num_topk = num_topk
        self.reg_input = reg_input
        self.loss_form = loss_form

    def _init_layers(self):
        """Initialize layers of the head."""
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.rpn_conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.rpn_cls, std=0.01, bias=bias_cls)
        normal_init(self.rpn_reg, std=0.01)

    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        rpn_cls_score, rpn_bbox_pred = GradScale.apply(rpn_cls_score, rpn_bbox_pred, self.reg_weight)
        return rpn_cls_score, rpn_bbox_pred

    def flatten_labels(self, flat_labels, label_weights):
        prediction_number = flat_labels.shape[0]
        labels = torch.zeros( [prediction_number], dtype=flat_labels.dtype, device=flat_labels.device)
        labels[flat_labels == 1] = 1
        labels[label_weights == 0] = -1
        return labels.reshape(-1)

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=None,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        all_labels=[]
        all_label_weights=[]
        all_cls_scores=[]
        all_bbox_targets=[]
        all_bbox_weights=[]
        all_bbox_preds=[]
        for labels, label_weights, cls_score, bbox_targets, bbox_weights, bbox_pred in zip(labels_list, label_weights_list,cls_scores, bbox_targets_list, bbox_weights_list, bbox_preds):
            all_labels.append(labels.reshape(-1))
            all_label_weights.append(label_weights.reshape(-1))
            all_cls_scores.append(cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels))
            
            all_bbox_targets.append(bbox_targets.reshape(-1, 4))
            all_bbox_weights.append(bbox_weights.reshape(-1, 4))
            all_bbox_preds.append(bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4))

        labels = torch.cat(all_labels)
        label_weights = torch.cat(all_label_weights)
        cls_scores = torch.cat(all_cls_scores)
        bbox_targets = torch.cat(all_bbox_targets)
        bbox_weights = torch.cat(all_bbox_weights)
        bbox_preds = torch.cat(all_bbox_preds)
        
        pos_idx = (labels == 1)
        if pos_idx.sum() > 0:
            num_anchors, num_cls = cls_scores.shape

            # get one_hot labels
            one_hot_labels = torch.zeros([num_anchors, num_cls]).to(labels)
            one_hot_labels[pos_idx, :] = 1

            ignore_idx = label_weights == 0
            one_hot_labels[ignore_idx, :] = -1

            # get iou_scores
            bbox_preds = self.delta2bbox(bbox_preds[pos_idx])
            bbox_targets = self.delta2bbox(bbox_targets[pos_idx])

            # giou
            if self.reg_input == 'giou':
                pos_ious = giou(bbox_preds, bbox_targets)
                pos_ious = (pos_ious + 1)/2
            # iou
            if self.reg_input == 'iou':
                pos_ious = iou(bbox_preds, bbox_targets)
            # l1
            if self.reg_input == 'l1':
                pos_ious = 1 - torch.abs(bbox_preds - bbox_targets).sum(-1)
                pos_ious = torch.clamp(pos_ious, min=0, max=1)

            ious = torch.zeros([num_anchors]).to(bbox_preds)
            ious[pos_idx] = pos_ious
            ious = ious.view(-1, 1).repeat(1, num_cls)
            ious = ious * one_hot_labels
            
            loss = paploss(cls_scores, ious, one_hot_labels, self.ctrl_points, self.num_topk, self.loss_form)
        else:
            loss = 0 * torch.cat(all_cls_scores) + 0 * torch.cat(all_bbox_preds)
        return dict(loss=loss)

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for each scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # we set FG labels to [0, num_class-1] and BG label to
                # num_class in other heads since mmdet v2.0, However we
                # keep BG label as 0 and FG label as 1 in rpn head
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = mlvl_anchors[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size > 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_inds = torch.nonzero(
                (w >= cfg.min_bbox_size)
                & (h >= cfg.min_bbox_size),
                as_tuple=False).squeeze()
            if valid_inds.sum().item() != len(proposals):
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
                ids = ids[valid_inds]

        # TODO: remove the hard coded nms type
        nms_cfg = dict(type='nms', iou_thr=cfg.nms_thr)
        dets, keep = batched_nms(proposals, scores, ids, nms_cfg)
        return dets[:cfg.nms_post]

    def delta2bbox(self, deltas, means=[0., 0., 0., 0.], stds=[0.1, 0.1, 0.2, 0.2], max_shape=None, wh_ratio_clip=16/1000):

        wx, wy, ww, wh = stds
        dx = deltas[:, 0] * wx
        dy = deltas[:, 1] * wy
        dw = deltas[:, 2] * ww
        dh = deltas[:, 3] * wh
        
        max_ratio = np.abs(np.log(wh_ratio_clip))

        dw = dw.clamp(min=-max_ratio, max=max_ratio)
        dh = dh.clamp(min=-max_ratio, max=max_ratio)

        pred_ctr_x = dx
        pred_ctr_y = dy
        pred_w = torch.exp(dw)
        pred_h = torch.exp(dh)

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h
        
        return torch.stack([x1, y1, x2, y2], dim=-1)


class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cls_scores, bbox_preds, reg_weight):
        ctx.weight = reg_weight
        return cls_scores, bbox_preds
    
    @staticmethod
    def backward(ctx, grad_cls, grad_bbox):
        return grad_cls, ctx.weight * grad_bbox, None
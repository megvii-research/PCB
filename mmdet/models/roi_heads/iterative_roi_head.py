# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection (https://github.com/open-mmlab/mmdetection)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from mmdet.models.utils import MLP
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
from .standard_roi_head import StandardRoIHead


@HEADS.register_module()
class IterativeRoIHead(StandardRoIHead):
    def __init__(self, 
        n_iter=3,
        mlp_layers=2,
        box_reg_iter=-1,
        iter_loss_weight=None,
        reg_iter=-1,
        use_layer_norm=False,
        **kwargs):
        super(IterativeRoIHead, self).__init__(**kwargs)
        self.n_iter = n_iter
        self.box_reg_iter = box_reg_iter
        
        if self.bbox_head.custom_cls_channels:
            cls_input_dim = self.bbox_head.loss_cls.get_cls_channels(self.bbox_head.num_classes)
        else:
            cls_input_dim = self.bbox_head.num_classes + 1
        
        self.cls_sub_module = MLP(input_dim=cls_input_dim, hidden_dim=512, output_dim=256, n_layers=mlp_layers, use_layer_norm=use_layer_norm)
        self.loc_sub_module = MLP(input_dim= 4 * self.bbox_head.num_classes, hidden_dim=512, output_dim= 7 * 7, n_layers=mlp_layers)
        
        self.iter_loss_weight = iter_loss_weight
        self.reg_iter = reg_iter
    
    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    """Simplest base roi head including one bbox head and one mask head."""
    def _bbox_forward(self, x, rois, train=False):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        bbox_feats_list = [bbox_feats]
        predictions_list = []
        for i in range(self.n_iter):
            if self.with_shared_head:
                raise NotImplementedError
                bbox_feats = self.shared_head(bbox_feats_list[-1])

            cls_score, bbox_pred = self.bbox_head(bbox_feats_list[-1])

            channel_bias = self.cls_sub_module(cls_score).view(-1, 256, 1, 1)
            spatial_bias = self.loc_sub_module(bbox_pred).view(-1, 1, 7, 7) 
            
            last = bbox_feats_list[-1]
            bbox_feats_list.append(last * spatial_bias + channel_bias)
            predictions_list.append((cls_score, bbox_pred))

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        if train:
            return bbox_results, predictions_list
        return bbox_results
    
    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results, predictions_list = self._bbox_forward(x, rois, train=True)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_box_list = []
        for i in range(self.n_iter):
            loss_box_list.append(self.bbox_head.loss(predictions_list[i][0],
                                                    predictions_list[i][1], rois,
                                                    *bbox_targets,iter_id = i))
        loss_bbox = dict()
        for i, loss_box_item in enumerate(loss_box_list):
            for key, val in loss_box_item.items():
                if key in loss_bbox:
                    loss_bbox[key] = loss_bbox[key] + val * self.iter_loss_weight[i]
                else:
                    loss_bbox[key] = val * self.iter_loss_weight[i]
        if self.reg_iter != -1:
            loss_bbox["loss_bbox"] = loss_box_list[self.reg_iter]["loss_bbox"]

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results
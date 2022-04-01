# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection (https://github.com/open-mmlab/mmdetection)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .accuracy import accuracy
from .cross_entropy_loss import cross_entropy
from .utils import weight_reduce_loss


def seesaw_ce_loss(cls_score,
                   labels,
                   label_weights,
                   cum_samples,
                   num_classes,
                   p,
                   q,
                   eps,
                   reduction='mean',
                   avg_factor=None,
                   soft_target=None):
    """Calculate the Seesaw CrossEntropy loss.

    Args:
        cls_score (torch.Tensor): The prediction with shape (N, C),
             C is the number of classes.
        labels (torch.Tensor): The learning label of the prediction.
        label_weights (torch.Tensor): Sample-wise loss weight.
        cum_samples (torch.Tensor): Cumulative samples for each category.
        num_classes (int): The number of classes.
        p (float): The ``p`` in the mitigation factor.
        q (float): The ``q`` in the compenstation factor.
        eps (float): The minimal value of divisor to smooth
             the computation of compensation factor
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert cls_score.size(-1) == num_classes
    assert len(cum_samples) == num_classes

    onehot_labels = F.one_hot(labels, num_classes)

    seesaw_weights = cls_score.new_ones(onehot_labels.size())

    # mitigation factor
    if p > 0:
        sample_ratio_matrix = cum_samples[None, :].clamp(
            min=1) / cum_samples[:, None].clamp(min=1)
        index = (sample_ratio_matrix < 1.0).float()
        sample_weights = sample_ratio_matrix.pow(p) * index + (1 - index)
        mitigation_factor = sample_weights[labels.long(), :]
        seesaw_weights = seesaw_weights * mitigation_factor

    # compensation factor
    if q > 0:
        scores = F.softmax(cls_score.detach(), dim=1)
        self_scores = scores[
            torch.arange(0, len(scores)).to(scores.device).long(),
            labels.long()]
        score_matrix = scores / self_scores[:, None].clamp(min=eps)
        index = (score_matrix > 1.0).float()
        compensation_factor = score_matrix.pow(q) * index + (1 - index)
        seesaw_weights = seesaw_weights * compensation_factor

    cls_score = cls_score + (seesaw_weights.log() * (1 - onehot_labels))

    if soft_target is not None:
        # loss = F.cross_entropy(cls_score, soft_target, weight=None, reduction='none')
        log_prob = F.log_softmax(cls_score)
        loss = - (soft_target * log_prob).sum(dim=1)
    else:
        loss = F.cross_entropy(cls_score, labels, weight=None, reduction='none')

    if label_weights is not None:
        label_weights = label_weights.float()
    loss = weight_reduce_loss(
        loss, weight=label_weights, reduction=reduction, avg_factor=avg_factor)
    return loss


@LOSSES.register_module()
class SeesawPCBLoss(nn.Module):
    """
    Seesaw Loss for Long-Tailed Instance Segmentation (CVPR 2021)
    arXiv: https://arxiv.org/abs/2008.10032

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
             of softmax. Only False is supported.
        p (float, optional): The ``p`` in the mitigation factor.
             Defaults to 0.8.
        q (float, optional): The ``q`` in the compenstation factor.
             Defaults to 2.0.
        num_classes (int, optional): The number of classes.
             Default to 1203 for LVIS v1 dataset.
        eps (float, optional): The minimal value of divisor to smooth
             the computation of compensation factor
        reduction (str, optional): The method that reduces the loss to a
             scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
        return_dict (bool, optional): Whether return the losses as a dict.
             Default to True.
    """

    def __init__(self,
                 use_sigmoid=False,
                 p=0.8,
                 q=2.0,
                 num_classes=1203,
                 eps=1e-2,
                 reduction='mean',
                 loss_weight=1.0,
                 return_dict=True,
                 momentum=0.99,
                 start_epoch = 17,
                 alpha = 0.0,
                 n_iter=3):
        super(SeesawPCBLoss, self).__init__()
        assert not use_sigmoid
        self.use_sigmoid = False
        self.p = p
        self.q = q
        self.num_classes = num_classes
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.return_dict = return_dict

        # 0 for pos, 1 for neg
        self.cls_criterion = seesaw_ce_loss

        # cumulative samples for each category
        self.register_buffer(
            'cum_samples',
            torch.zeros(self.num_classes + 1, dtype=torch.float))

        # custom output channels of the classifier
        self.custom_cls_channels = True
        # custom activation of cls_score
        self.custom_activation = True
        # custom accuracy of the classsifier
        self.custom_accuracy = True

        _epoch = torch.zeros(1).to("cuda")
        _epoch.requires_grad = False
        torch.distributed.broadcast(_epoch, src=0)
        self.register_buffer('_epoch', _epoch)

        fg_confusion_matrix = torch.zeros((self.num_classes, self.num_classes)).to("cuda")
        fg_confusion_matrix.requires_grad = False
        torch.distributed.broadcast(fg_confusion_matrix, src=0)
        self.register_buffer('fg_confusion_matrix', fg_confusion_matrix)

        num_inst_cnt = torch.zeros((self.num_classes,)).to("cuda")
        num_inst_cnt.requires_grad = False
        torch.distributed.broadcast(num_inst_cnt, src=0)
        self.register_buffer('num_inst_cnt', num_inst_cnt)

        self.momentum = momentum
        self.start_epoch = start_epoch
        self.alpha = alpha
        self.n_iter = n_iter

    def _split_cls_score(self, cls_score):
        # split cls_score to cls_score_classes and cls_score_objectness
        assert cls_score.size(-1) == self.num_classes + 2
        cls_score_classes = cls_score[..., :-2]
        cls_score_objectness = cls_score[..., -2:]
        return cls_score_classes, cls_score_objectness

    def get_cls_channels(self, num_classes):
        """Get custom classification channels.

        Args:
            num_classes (int): The number of classes.

        Returns:
            int: The custom classification channels.
        """
        assert num_classes == self.num_classes
        return num_classes + 2

    def get_activation(self, cls_score):
        """Get custom activation of cls_score.

        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C + 2).

        Returns:
            torch.Tensor: The custom activation of cls_score with shape
                 (N, C + 1).
        """
        cls_score_classes, cls_score_objectness = self._split_cls_score(
            cls_score)
        score_classes = F.softmax(cls_score_classes, dim=-1)
        score_objectness = F.softmax(cls_score_objectness, dim=-1)
        score_pos = score_objectness[..., [0]]
        score_neg = score_objectness[..., [1]]
        score_classes = score_classes * score_pos
        scores = torch.cat([score_classes, score_neg], dim=-1)
        return scores

    def get_accuracy(self, cls_score, labels):
        """Get custom accuracy w.r.t. cls_score and labels.

        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C + 2).
            labels (torch.Tensor): The learning label of the prediction.

        Returns:
            Dict [str, torch.Tensor]: The accuracy for objectness and classes,
                 respectively.
        """
        pos_inds = labels < self.num_classes
        obj_labels = (labels == self.num_classes).long()
        cls_score_classes, cls_score_objectness = self._split_cls_score(
            cls_score)
        acc_objectness = accuracy(cls_score_objectness, obj_labels)
        acc_classes = accuracy(cls_score_classes[pos_inds], labels[pos_inds])
        acc = dict()
        acc['acc_objectness'] = acc_objectness
        acc['acc_classes'] = acc_classes
        return acc

    def forward(self,
                cls_score,
                labels,
                label_weights=None,
                avg_factor=None,
                reduction_override=None,
                iter_id=2):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C + 2).
            labels (torch.Tensor): The learning label of the prediction.
            label_weights (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                 the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                 Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor | Dict [str, torch.Tensor]:
                 if return_dict == False: The calculated loss |
                 if return_dict == True: The dict of calculated losses
                 for objectness and classes, respectively.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        assert cls_score.size(-1) == self.num_classes + 2
        pos_inds = labels < self.num_classes
        # 0 for pos, 1 for neg
        obj_labels = (labels == self.num_classes).long()

        # accumulate the samples for each category
        unique_labels = labels.unique()
        for u_l in unique_labels:
            inds_ = labels == u_l.item()
            self.cum_samples[u_l] += inds_.sum()

        if label_weights is not None:
            label_weights = label_weights.float()
        else:
            label_weights = labels.new_ones(labels.size(), dtype=torch.float)

        cls_score_classes, cls_score_objectness = self._split_cls_score(
            cls_score)

        if pos_inds.sum() > 0:
            pred_fg_distri = F.softmax(cls_score_classes[pos_inds], dim=1)

            fg_confusion_matrix_tmp = torch.zeros_like(self.fg_confusion_matrix).scatter_add_(0, labels[pos_inds].view(-1,1).repeat(1,self.num_classes), pred_fg_distri)

            fg_confusion_matrix_tmp_pool = [torch.zeros_like(fg_confusion_matrix_tmp) for i in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(fg_confusion_matrix_tmp_pool, fg_confusion_matrix_tmp)
            fg_confusion_matrix_tmp = sum(fg_confusion_matrix_tmp_pool)

            num_inst_cnt_tmp = torch.zeros_like(self.num_inst_cnt).scatter_add_(0, labels[pos_inds], torch.ones(pos_inds.sum()).to(self.num_inst_cnt.device))
            num_inst_cnt_tmp_pool = [torch.zeros_like(num_inst_cnt_tmp) for i in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(num_inst_cnt_tmp_pool, num_inst_cnt_tmp)
            num_inst_cnt_tmp = sum(num_inst_cnt_tmp_pool)

            fg_confusion_matrix_tmp[num_inst_cnt_tmp != 0] = fg_confusion_matrix_tmp[num_inst_cnt_tmp != 0] / num_inst_cnt_tmp[num_inst_cnt_tmp != 0].view(-1,1)

            self.fg_confusion_matrix[num_inst_cnt_tmp != 0] = self.fg_confusion_matrix[num_inst_cnt_tmp != 0] * self.momentum + \
                    fg_confusion_matrix_tmp[num_inst_cnt_tmp != 0] * (1 - self.momentum)
            self.fg_confusion_matrix[num_inst_cnt_tmp != 0] /= self.fg_confusion_matrix[num_inst_cnt_tmp != 0].sum(1, keepdim=True)
            
            self.num_inst_cnt = self.num_inst_cnt + num_inst_cnt_tmp
        
        # calculate loss_cls_classes (only need pos samples)
        if pos_inds.sum() > 0:
            loss_cls_classes = self.loss_weight * self.cls_criterion(
                cls_score_classes[pos_inds], labels[pos_inds],
                label_weights[pos_inds], self.cum_samples[:self.num_classes],
                self.num_classes, self.p, self.q, self.eps, reduction,
                avg_factor)
        else:
            loss_cls_classes = cls_score_classes[pos_inds].sum()
        # calculate loss_cls_objectness
        loss_cls_objectness = self.loss_weight * cross_entropy(
            cls_score_objectness, obj_labels, label_weights, reduction,
            avg_factor)

        if self.return_dict:
            loss_cls = dict()
            loss_cls['loss_cls_objectness'] = loss_cls_objectness
            if self._epoch >= self.start_epoch and (self.num_inst_cnt == 0).sum() == 0:
                alpha = self.alpha / (self.n_iter - 1)* iter_id

                loss_cls['loss_cls_classes'] = loss_cls_classes * (1 - alpha)

                # C x C (GT vs. Predict)
                cm = self.fg_confusion_matrix.clone().detach()
                cm /= cm.sum(0, keepdim=True)
                p_t = cm[:self.num_classes,labels[pos_inds]].t() 

                if pos_inds.sum() > 0:
                    loss_cls['loss_pcb_classes'] = self.loss_weight * self.cls_criterion(
                        cls_score_classes[pos_inds], labels[pos_inds],
                        label_weights[pos_inds], self.cum_samples[:self.num_classes],
                        self.num_classes, self.p, self.q, self.eps, reduction,
                        avg_factor, soft_target=p_t) * alpha
                else:
                    loss_cls['loss_pcb_classes'] = cls_score_classes[pos_inds].sum() * alpha
            else:
                loss_cls['loss_cls_classes'] = loss_cls_classes
        else:
            raise NotImplementedError
            loss_cls = loss_cls_classes + loss_cls_objectness
        return loss_cls

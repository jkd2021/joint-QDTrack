import numpy as np
import torch

import mmcv
from mmdet.core import bbox2roi
from mmdet.models import HEADS, build_head

from .joint_roi_head_mask import JointMaskQuasiDenseRoIHead


@HEADS.register_module()
class JointQuasiDenseSegRoIHeadRefine(JointMaskQuasiDenseRoIHead):  # PCAN joint model's RoI head (not developed)

    def __init__(self, refine_head=None, amodal_refine_head=None, double_train=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert refine_head is not None
        assert amodal_refine_head is not None
        self.init_refine_head(refine_head)
        self.init_amodal_refine_head(amodal_refine_head)
        self.double_train = double_train

    @property
    def with_refine(self):
        return hasattr(self, 'refine_head') and self.refine_head is not None and \
               hasattr(self, 'amodal_refine_head') and self.amodal_refine_head is not None

    def init_refine_head(self, refine_head):
        self.refine_head = build_head(refine_head)

    def init_amodal_refine_head(self, amodal_refine_head):
        self.amodal_refine_head = build_head(amodal_refine_head)

    def init_weights(self, *args, **kwargs):
        super().init_weights(*args, **kwargs)
        self.refine_head.init_weights()
        self.amodal_refine_head.init_weights()

    def _refine_forward(self, key_feats, key_masks, key_labels, ref_feats,
                        ref_masks):
        num_rois = key_masks.size(0)
        inds = torch.arange(0, num_rois, device=key_masks.device).long()
        key_masks = key_masks[inds, key_labels].unsqueeze(dim=1)
        ref_masks = ref_masks[inds, key_labels].unsqueeze(dim=1)

        if self.double_train and self.training:
            ref_masks = self.refine_head(ref_feats, ref_masks, ref_feats,
                                         ref_masks).detach()
        #print('ref feats:', ref_feats.shape)
        #print('key feats:', key_feats.shape)
        refine_pred = self.refine_head(ref_feats, ref_masks, key_feats,     # use ref img's feat&mask to refine the mask
                                       key_masks)                           # inst_lvl crs-attent in em_match_head.py
        refine_results = dict(refine_pred=refine_pred)
        return refine_results

    def _amodal_refine_forward(self, key_feats, key_masks, key_labels, ref_feats,
                        ref_masks):
        num_rois = key_masks.size(0)
        inds = torch.arange(0, num_rois, device=key_masks.device).long()
        key_masks = key_masks[inds, key_labels].unsqueeze(dim=1)
        ref_masks = ref_masks[inds, key_labels].unsqueeze(dim=1)

        if self.double_train and self.training:
            ref_masks = self.amodal_refine_head(ref_feats, ref_masks, ref_feats,
                                         ref_masks).detach()
        #print('ref feats:', ref_feats.shape)
        #print('key feats:', key_feats.shape)
        amodal_refine_pred = self.amodal_refine_head(ref_feats, ref_masks, key_feats,     # use ref img's feat&mask to refine the mask
                                       key_masks)                                  # inst_lvl crs-attent in em_match_head.py
        amodal_refine_results = dict(amodal_refine_pred=amodal_refine_pred)
        return amodal_refine_results


    def _refine_forward_train(self, key_sampling_results, ref_sampling_results,
                              key_mask_results, ref_mask_results, x, ref_x,
                              gt_match_inds):
        num_key_rois = [len(res.pos_bboxes) for res in key_sampling_results]
        key_pos_pids = [
            gt_match_ind[res.pos_assigned_gt_inds]
            for res, gt_match_ind in zip(key_sampling_results, gt_match_inds)]
        key_pos_bboxes = [res.pos_bboxes for res in key_sampling_results]
        key_embeds = torch.split(
            self._track_forward(x, key_pos_bboxes), num_key_rois)

        num_ref_rois = [len(res.pos_bboxes) for res in ref_sampling_results]
        ref_pos_pids = [
            res.pos_assigned_gt_inds for res in ref_sampling_results]
        ref_pos_bboxes = [res.pos_bboxes for res in ref_sampling_results]
        ref_embeds = torch.split(
            self._track_forward(ref_x, ref_pos_bboxes), num_ref_rois)

        valids, ref_inds = self.refine_head.match(
            key_embeds, ref_embeds, key_pos_pids, ref_pos_pids)

        def valid_select(inputs, num_splits, inds):
            inputs = torch.split(inputs, num_splits)
            inputs = torch.cat(
                [input_[ind] for input_, ind in zip(inputs, inds)])
            return inputs

        key_feats = valid_select(
            key_mask_results['mask_feats'], num_key_rois, valids)
        key_masks = valid_select(
            key_mask_results['mask_pred'], num_key_rois, valids)
        key_targets = valid_select(
            key_mask_results['mask_targets'], num_key_rois, valids)
        key_labels = torch.cat(
            [res.pos_gt_labels[valid]
            for res, valid in zip(key_sampling_results, valids)])
        ref_feats = valid_select(
            ref_mask_results['mask_feats'], num_ref_rois, ref_inds)
        ref_masks = valid_select(
            ref_mask_results['mask_pred'], num_ref_rois, ref_inds)

        if key_masks.size(0) == 0:
            key_feats = key_mask_results['mask_feats']
            key_masks = key_mask_results['mask_pred']
            key_targets = key_mask_results['mask_targets']
            key_labels = torch.cat([
                res.pos_gt_labels for res in key_sampling_results])
            ref_feats = key_feats.detach()
            ref_masks = key_masks.detach()

        refine_results = self._refine_forward(key_feats, key_masks, key_labels,
                                              ref_feats, ref_masks)
        refine_targets = key_targets
        loss_refine = self.refine_head.loss_mask(
            refine_results['refine_pred'].squeeze(dim=1), refine_targets)  # CE between the current img's mask GT
        refine_results.update(loss_refine=loss_refine,                     # and the refined mask
                              refine_targets=refine_targets)
        return refine_results


    def _amodal_refine_forward_train(self, key_sampling_results, ref_sampling_results,
                              key_mask_results, ref_mask_results, x, ref_x,
                              gt_match_inds):
        pass



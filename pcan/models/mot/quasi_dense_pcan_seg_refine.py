import torch
from mmdet.core import bbox2result
import numpy as np
import math

from pcan.core import segtrack2result
from ..builder import MODELS
from .quasi_dense_pcan_seg import QuasiDenseMaskRCNN
from torch import nn
import torch.nn.functional as F

@MODELS.register_module()
class EMQuasiDenseMaskRCNNRefine(QuasiDenseMaskRCNN):  # PCAN model

    def fix_modules(self):
        fixed_modules = [
            self.backbone,
            self.neck,
            self.rpn_head,
            self.roi_head.bbox_roi_extractor,
            self.roi_head.bbox_head,
            self.roi_head.track_roi_extractor,
            self.roi_head.track_head,
            self.roi_head.mask_roi_extractor,
            self.roi_head.mask_head]
        
        for module in fixed_modules:
            for name, param in module.named_parameters():
                param.requires_grad = False

    @torch.no_grad()
    def _em_iter(self, x, mu):                                      # in 3.2(2)of paper, x is i-th layer's features embed in memo(features), mu is key protos in memo
        R, C, H, W = x.size()
        x = x.view(R, C, H * W)                                     # r * c * n    # n is num of pixel-lvl embed=H*W=184*288, k is num of key protos
        for _ in range(self.stage_num):
            z = torch.einsum('rcn,rck->rnk', (x, mu))               # r * n * k    # in 3.2(3) of paper, Bayes prpbabilities of potential protos
            z = F.softmax(20 * z, dim=2)                            # r * n * k    # in 3.2(3) of paper, SoftMax Operation to get Bayes prpbabilities of potential protos (prpbability of a key K to the j-th proto)
            z = self._l1norm(z, dim=1)                              # r * n * k    # in 3.2(4) of paper, z is the Bayes probablities
            mu = torch.einsum('rcn,rnk->rck', (x, z))               # r * c * k    # in 3.2(4) of paper, x is i-th layer's features in memo, to get value of each proto
            mu = self._l2norm(mu, dim=1)                            # r * c * k
        return mu                                                              # new key protos, v_mu in 3.2(4)

    def _prop(self, feat, mu):                                      # feat = query key encode(feature) of cur frame in i-th layer, mu = key protos
        B, C, H, W = feat.size()
        x = feat.view(B, C, -1)                                     # B * C * N     # feat is transfered into x(i-th layer's query key encoding)
        z = torch.einsum('bcn,bck->bnk', (x, mu))                   # B * N * K     # 3.3.1(5=6) in paper, projection(contribution) of features from a past frame to the current frame
        z = F.softmax(z, dim=2)                                     # B * N * K     # 3.3.1(6), use SoftMax to get the contribution weights of past frames (3.3 in paper) Temporal feature aggregation
        return z

    def forward_test(self, img, img_metas, rescale=False):          # this is PCAN main test function
        # TODO inherit from a base tracker
        assert self.roi_head.with_track, 'Track head must be implemented.'
        img_metas = img_metas[0]
        frame_id = img_metas[0].get('frame_id', -1)
        x = self.extract_feat(img[0])

        # frame-level Prototypical Cross-Attention
        if frame_id == 0:
            self.init_tracker()
            # self.memo_banks = [x[0], x[1], x[2]]                          # features maps in layer P3,P4,P5
            self.memo_banks = [x[0], x[1], x[2], x[3], x[4]]
            # self.mus = [self.mu0, self.mu1, self.mu2]                     # keys protos in layer P3,P4,P5
            self.mus = [self.mu0, self.mu1, self.mu2, self.mu3, self.mu4]

        x = list(x)                                                         #  x[i] is features in cur frame in diff layers
        for i in range(2):                                                  # multi-layer prototypical feature fusion P5,P4,P3
            B, C, H, W = self.memo_banks[i].size()
            protos = self._em_iter(self.memo_banks[i], self.mus[i])         # 3.3.1(3) mus is N key protos, memo_banks is the features saved in memo iter by iter, protos = new key protos k_mu in 3.2(4), EM clustering in the key space (mus are original key protos)
            ref_z = self._prop(x[i], protos)                                # 3.3.1(5=6) in paper, ref_z is N value protos
            ref_r = torch.einsum('bck,bnk->bcn', (protos, ref_z))           # 3.3.1(6) in paper, ref_r is the value y (Temporal feature aggregation?), The contribution of each frame should be summed with weights (from key size to feat size)
            ref_r = ref_r.view(B, C, H, W)                                  # ref_r is transferred from HW into H*W aggregated features of past frames
                                                                            # aggregated features containing past and cur info
            x[i] = x[i] * 0.75 + ref_r * 0.25                               # ori momentum is 0.25
            self.memo_banks[i] = x[i] * 0.75 + self.memo_banks[i] * 0.25    # update value (5 feature maps containing the past and cur)
            self.mus[i] = self.mus[i] * 0.5 + protos * 0.5                  # update key protos
        x = tuple(x)

        proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        det_bboxes, det_labels, det_masks, track_feats = (
            self.roi_head.simple_test(x, img_metas, proposal_list, rescale))
        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.roi_head.bbox_head.num_classes)
        segm_result, ori_segms, labels_ori = self.roi_head.get_seg_masks(
            img_metas, det_bboxes, det_labels, det_masks, rescale=rescale)

        update_cls_segms = [[] for _ in range(self.roi_head.bbox_head.num_classes)]
        if track_feats is None:
            from collections import defaultdict
            track_result = defaultdict(list)
            refine_bbox_result = bbox_result
            update_cls_segms = segm_result
        else:
            bboxes, labels, masks, ids, embeds, ref_feats, ref_masks, inds, valids = (
                self.tracker.match(
                    bboxes=det_bboxes,
                    labels=det_labels,
                    masks=det_masks,
                    track_feats=track_feats,
                    frame_id=frame_id))

            mask_preds, mask_feats = masks['mask_pred'], masks['mask_feats']

            refine_segm_result, segms, refine_preds = self.roi_head.simple_test_refine(
                img_metas, mask_feats, mask_preds, bboxes, labels, ref_feats,
                ref_masks, rescale=rescale)

            ori_segms = np.array(ori_segms)
            ori_segms = ori_segms[list(inds.cpu().numpy()), :]
           
            labels_ori = labels_ori[inds]
            valids = list(valids.cpu().numpy()) 
            valids_new = [ind2 for ind2 in range(len(valids)) if valids[ind2] == True]

            ori_segms[valids_new,:] = segms
            ori_segms = list(ori_segms)
            
            for i1 in range(len(ori_segms)):
                update_cls_segms[labels_ori[i1]].append(ori_segms[i1])

            self.tracker.update_memo(ids, bboxes, mask_preds, mask_feats,
                                     refine_preds, embeds, labels, frame_id)

            track_result = segtrack2result(bboxes, labels, segms, ids)

        return dict(bbox_result=bbox_result, segm_result=update_cls_segms,
                    track_result=track_result)



        # class DotProductSimilarity(nn.Module):                        # no use
        #     def __init__(self, scale_output=False):
        #         super(DotProductSimilarity, self).__init__()
        #         self.scale_output = scale_output
        #
        #     def forward(self, tensor_1, tensor_2):
        #         result = (tensor_1 * tensor_2).sum(dim=-1)
        #         if self.scale_output:
        #             # TODO why allennlp do multiplication at here ?
        #             result /= math.sqrt(tensor_1.size(-1))
        #         return result
        #
        # self.temp_len = 4
        # self.memo_feat_frame = []
        # self.memo_feat_frame.append(x)
        # if len(self.memo_feat_frame) > self.temp_len:
        #     self.memo_feat_frame.pop(0)
        # self.temp_mus = []
        # self.temp_mus.append(self.mus)
        #
        # x = list(x)
        # for i in range(2):
        #     B, C, H, W = self.memo_banks[i].size()
        #     self.temp_protos=[]
        #     self.temp_ref_r = []
        #     for j in range(len(self.memo_feat_frame)):
        #         self.temp_protos.append(self._em_iter(self.memo_feat_frame[j][i], self.temp_mus[j][i]))
        #         ref_z = self._prop(x[i], self.temp_protos[j])
        #         ref_r = torch.einsum('bck,bnk->bcn', (self.temp_protos[j], ref_z))
        #         ref_r = ref_r.view(B, C, H, W)
        #         self.temp_ref_r.append(ref_r)
        #     for m in range(len(self.temp_ref_r)):
        #         DotProductSimilarity(x[i], self.temp_ref_r[m])
        #     self.proj_weight_lst=[]
        #
        #         x[i] = x[i] * 0.75 + ref_r * 0.25
        #         self.memo_banks[i] = x[i] * 0.75 + self.memo_banks[i] * 0.25
        #         self.mus[i] = self.mus[i] * 0.5 + protos * 0.5
        # x = tuple(x)
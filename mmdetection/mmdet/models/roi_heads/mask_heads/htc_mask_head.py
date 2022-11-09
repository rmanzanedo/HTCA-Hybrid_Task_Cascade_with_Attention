# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .fcn_mask_head import FCNMaskHead
from .SAM_centermask import SpatialAttention
from ...utils.visualize import show_image_from_tensor
import torch
@HEADS.register_module()
class HTCMaskHead(FCNMaskHead):

    def __init__(self, with_conv_res=True, *args, **kwargs):
        super(HTCMaskHead, self).__init__(*args, **kwargs)
        self.with_conv_res = with_conv_res
        if self.with_conv_res:
            self.conv_res = ConvModule(
                self.conv_out_channels,
                self.conv_out_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        ############################################Mi codigo#####################################
        self.spatialAtt = SpatialAttention()

    def forward(self, x, res_feat=None, return_logits=True, return_feat=True):
        for i in range(x.shape[0]):
            show_image_from_tensor(x[i][0].unsqueeze(0).cpu(), 'RoI')
        # quit()
        if res_feat is not None:
            assert self.with_conv_res

            ############################################Mi codigo#####################################
            # avg_out = torch.max(res_feat, dim=1, keepdim=True)
            # show_image_from_tensor(avg_out[0][0][0].unsqueeze(0).cpu(), 'maxbeforesam')
            # avg_out = torch.mean(res_feat, dim=1, keepdim=True)
            # show_image_from_tensor(avg_out[0][0].unsqueeze(0).cpu(), 'avebeforesam')
            # show_image_from_tensor(res_feat[0][0].unsqueeze(0).cpu(), 'beforesam')
            # res_feat = self.spatialAtt(res_feat)
            # show_image_from_tensor(res_feat[0][0].unsqueeze(0).cpu(), 'aftersam')
            # avg_out = torch.mean(res_feat, dim=1, keepdim=True)
            # show_image_from_tensor(avg_out[0][0].unsqueeze(0).cpu(), 'aveaftersam')
            # avg_out = torch.max(res_feat, dim=1, keepdim=True)
            # show_image_from_tensor(avg_out[0][0][0].unsqueeze(0).cpu(), 'maxaftersam')
            ############################################Original######################################
            res_feat = self.conv_res(res_feat)
            x = x + res_feat
        for conv in self.convs:
            x = conv(x)
        ############################################Mi codigo#####################################
        # avg_out = torch.max(x, dim=1, keepdim=True)
        # show_image_from_tensor(avg_out[0][0][0].unsqueeze(0).cpu(), 'maxbeforesam')
        # avg_out = torch.mean(x, dim=1, keepdim=True)
        # show_image_from_tensor(avg_out[0][0].unsqueeze(0).cpu(), 'avebeforesam')
        # show_image_from_tensor(x[0][0].unsqueeze(0).cpu(), 'beforesam')
        # x = self.spatialAtt(x)
        # show_image_from_tensor(x[0][0].unsqueeze(0).cpu(), 'aftersam')
        # avg_out = torch.mean(x, dim=1, keepdim=True)
        # show_image_from_tensor(avg_out[0][0].unsqueeze(0).cpu(), 'aveaftersam')
        # avg_out = torch.max(x, dim=1, keepdim=True)
        # show_image_from_tensor(avg_out[0][0][0].unsqueeze(0).cpu(), 'maxaftersam')
        ############################################Original######################################
        res_feat = x
        outs = []
        if return_logits:
            ############################################Mi codigo#####################################
            # avg_out = torch.max(x, dim=1, keepdim=True)
            # show_image_from_tensor(avg_out[0][0][0].unsqueeze(0).cpu(), 'maxbeforesam')
            # avg_out = torch.mean(x, dim=1, keepdim=True)
            # show_image_from_tensor(avg_out[0][0].unsqueeze(0).cpu(), 'avebeforesam')
            # show_image_from_tensor(x[0][0].unsqueeze(0).cpu(), 'beforesam')
            x= self.spatialAtt(x)
            # show_image_from_tensor(x[0][0].unsqueeze(0).cpu(), 'beforedecov')
            # avg_out = torch.mean(x, dim=1, keepdim=True)
            # show_image_from_tensor(avg_out[0][0].unsqueeze(0).cpu(), 'avebeforedeconv')
            # avg_out = torch.max(x, dim=1, keepdim=True)
            # show_image_from_tensor(avg_out[0][0][0].unsqueeze(0).cpu(), 'maxbeforedeconv')

            ############################################Original######################################

            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
            mask_pred = self.conv_logits(x)
            outs.append(mask_pred)
            for i in range(mask_pred.shape[0]):
                show_image_from_tensor(mask_pred[i][0].unsqueeze(0).cpu(), 'mask')
            quit()

            # show_image_from_tensor(mask_pred[0][0].unsqueeze(0).cpu(), 'RoI')
        if return_feat:
            outs.append(res_feat)
        return outs if len(outs) > 1 else outs[0]

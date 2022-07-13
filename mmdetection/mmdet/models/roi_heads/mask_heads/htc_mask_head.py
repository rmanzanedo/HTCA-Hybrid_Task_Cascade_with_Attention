# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .fcn_mask_head import FCNMaskHead
from .SAM_centermask import SpatialAttention
from ...utils.visualize import show_image_from_tensor

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
        # self.spatialAtt = SpatialAttention()

    def forward(self, x, res_feat=None, return_logits=True, return_feat=True):
        # for i in range(x.shape[0]):
        #     show_image_from_tensor(x[i][0].unsqueeze(0).cpu(), 'RoI')
        # quit()
        if res_feat is not None:
            assert self.with_conv_res
            ############################################Mi codigo#####################################
            # res_feat = self.spatialAtt(res_feat)
            ############################################Original######################################
            res_feat = self.conv_res(res_feat)
            x = x + res_feat
        for conv in self.convs:
            x = conv(x)
        ############################################Mi codigo#####################################
        # x = self.spatialAtt(x)
        ############################################Original######################################
        res_feat = x
        outs = []
        if return_logits:
            ############################################Mi codigo#####################################
            # x = self.spatialAtt(x)
            ############################################Original######################################
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
            mask_pred = self.conv_logits(x)
            outs.append(mask_pred)
        if return_feat:
            outs.append(res_feat)
        return outs if len(outs) > 1 else outs[0]

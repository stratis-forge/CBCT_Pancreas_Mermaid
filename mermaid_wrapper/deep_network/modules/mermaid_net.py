from modules.layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../mermaid'))
import mermaid.module_parameters as pars
import mermaid.model_factory as py_mf
import mermaid.utils as py_utils
import mermaid.similarity_measure_factory as smf
from mermaid.data_wrapper import AdaptVal

from utils.registration_method import _get_low_res_size_from_size, _get_low_res_spacing_from_spacing, \
    get_resampled_image
from functools import partial


class CustLNCCSimilarity(smf.SimilarityMeasureSingleImage):

    def __init__(self, spacing, params):
        super(CustLNCCSimilarity, self).__init__(spacing, params)
        self.dim = len(spacing)
        self.resol_bound = params['similarity_measure']['lncc'][
            ('resol_bound', [128, 64], "resolution bound for using different strategy")]
        self.kernel_size_ratio = params['similarity_measure']['lncc'][(
            'kernel_size_ratio', [[1. / 16, 1. / 8, 1. / 4], [1. / 4, 1. / 2], [1. / 2]],
            "kernel size, ratio of input size")]
        self.kernel_weight_ratio = params['similarity_measure']['lncc'][
            ('kernel_weight_ratio', [[0.1, 0.3, 0.6], [0.3, 0.7], [1.]], "kernel size, ratio of input size")]
        self.strides = params['similarity_measure']['lncc'][(
            'stride', [[1. / 4, 1. / 4, 1. / 4], [1. / 4, 1. / 4], [1. / 4]],
            "step size, responded with ratio of kernel size")]
        self.dilations = params['similarity_measure']['lncc'][
            ('dilation', [[2, 2, 2], [2, 2], [1]], "dilation param, responded with ratio of kernel size")]
        if self.resol_bound[0] > -1:
            assert len(self.resol_bound) + 1 == len(self.kernel_size_ratio)
            assert len(self.resol_bound) + 1 == len(self.kernel_weight_ratio)
            assert len(self.resol_bound) + 1 == len(self.strides)
            assert len(self.resol_bound) + 1 == len(self.dilations)

    def __stepup(self, img_sz):
        max_scale = min(img_sz)
        for i, bound in enumerate(self.resol_bound):
            if max_scale >= bound:
                self.kernel = [int(max_scale * kz) for kz in self.kernel_size_ratio[i]]
                self.weight = self.kernel_weight_ratio[i]
                self.stride = self.strides[i]
                self.dilation = self.dilations[i]
                break
        if max_scale < self.resol_bound[-1]:
            self.kernel = [int(max_scale * kz) for kz in self.kernel_size_ratio[-1]]
            self.weight = self.kernel_weight_ratio[-1]
            self.stride = self.strides[-1]
            self.dilation = self.dilations[-1]

        self.num_scale = len(self.kernel)
        self.kernel_sz = [[k for _ in range(self.dim)] for k in self.kernel]
        self.step = [[max(int((ksz + 1) * self.stride[scale_id]), 1) for ksz in self.kernel_sz[scale_id]] for scale_id
                     in range(self.num_scale)]
        self.filter = [torch.ones([1, 1] + self.kernel_sz[scale_id]).cuda() for scale_id in range(self.num_scale)]

        if self.dim == 1:
            self.conv = F.conv1d
        elif self.dim == 2:
            self.conv = F.conv2d
        elif self.dim == 3:
            self.conv = F.conv3d
        else:
            raise ValueError(" Only 1-3d support")

    def compute_similarity_multiC(self, I0, I1, I0Source=None, phi=None):
        sz0 = I0.size()[0]
        sz1 = I1.size()[0]
        assert (sz0 == sz1 - 1)

        # last channel of target image is similarity mask
        num_of_labels = sz0 - 1
        mask = I1[-1, ...]

        sim = 0.0
        sim = sim + self.compute_similarity(I0[0, ...], I1[0, ...], isLabel=False, similarity_mask=None)
        if num_of_labels > 0:
            I0_labels = I0[1:, ...]
            I1_labels = I1[1:, ...]
            for nrL in range(num_of_labels):
                sim = sim + self.compute_similarity(I0_labels[nrL, ...], I1_labels[nrL, ...], isLabel=True,
                                                    similarity_mask=mask)
        return AdaptVal(sim / self.sigma ** 2)

    def compute_similarity(self, I0, I1, I0Source=None, phi=None, isLabel=False, similarity_mask=None):
        if isLabel:
            sim = (torch.mul((I0 - I1) ** 2, similarity_mask)).sum()
            sim = sim / ((torch.mul(I0 ** 2 + I1 ** 2, similarity_mask)).sum() + 1e-5)
        else:
            input = I0.view([1, 1] + list(I0.shape))
            target = I1.view([1, 1] + list(I1.shape))
            self.__stepup(img_sz=list(I0.shape))

            input_2 = input ** 2
            target_2 = target ** 2
            input_target = input * target
            sim = 0.
            for scale_id in range(self.num_scale):
                input_local_sum = self.conv(input, self.filter[scale_id], padding=0, dilation=self.dilation[scale_id],
                                            stride=self.step[scale_id]).view(input.shape[0], -1)
                target_local_sum = self.conv(target, self.filter[scale_id], padding=0, dilation=self.dilation[scale_id],
                                             stride=self.step[scale_id]).view(input.shape[0],
                                                                              -1)
                input_2_local_sum = self.conv(input_2, self.filter[scale_id], padding=0,
                                              dilation=self.dilation[scale_id],
                                              stride=self.step[scale_id]).view(input.shape[0],
                                                                               -1)
                target_2_local_sum = self.conv(target_2, self.filter[scale_id], padding=0,
                                               dilation=self.dilation[scale_id],
                                               stride=self.step[scale_id]).view(
                    input.shape[0], -1)
                input_target_local_sum = self.conv(input_target, self.filter[scale_id], padding=0,
                                                   dilation=self.dilation[scale_id], stride=self.step[scale_id]).view(
                    input.shape[0], -1)

                input_local_sum = input_local_sum.contiguous()
                target_local_sum = target_local_sum.contiguous()
                input_2_local_sum = input_2_local_sum.contiguous()
                target_2_local_sum = target_2_local_sum.contiguous()
                input_target_local_sum = input_target_local_sum.contiguous()

                numel = float(np.array(self.kernel_sz[scale_id]).prod())

                cross = input_target_local_sum - input_local_sum * target_local_sum / numel
                input_local_var = input_2_local_sum - input_local_sum ** 2 / numel
                target_local_var = target_2_local_sum - target_local_sum ** 2 / numel

                lncc = (cross * cross) / (input_local_var * target_local_var + 1e-5)
                torch.clamp(lncc, 0, 1)
                lncc = 1 - lncc.mean()
                sim += lncc * self.weight[scale_id]
        return sim


class MermaidNet(nn.Module):
    def __init__(self, model_config):
        super(MermaidNet, self).__init__()
        self.use_ct_labels_as_input = True
        self.use_bn = model_config['mermaid_net']['bn']
        print("Use Batch Normalization: {}".format(self.use_bn))
        self.use_dp = model_config['mermaid_net']['dp']
        self.dp_p = model_config['mermaid_net']['dp_p']
        self.n_of_feature = model_config['mermaid_net']['n_of_feature']
        self.dim = model_config['dim']
        self.img_sz = model_config['img_sz']
        self.batch_size = self.img_sz[0]

        self.mermaid_config_file = model_config['mermaid_config_file']
        self.mermaid_unit = None
        self.spacing = 1. / (np.array(self.img_sz[2:]) - 1)

        # members that will be set during mermaid initialization
        self.use_map = None
        self.map_low_res_factor = None
        self.lowResSize = None
        self.lowResSpacing = None
        self.identityMap = None
        self.lowResIdentityMap = None
        self.mermaid_criterion = None
        self.lowRes_fn = None
        self.sim_criterion = None

        self.init_mermaid_env(spacing=self.spacing)
        self.__setup_network_structure__()

        # results to return
        self.phi = None
        self.warped_moving_image = None
        self.warped_moving_labels = None

        self.loss_dict = {
            'all_loss': 0.,
            'mermaid_all_loss': 0.,
            'mermaid_sim_loss': 0.,
            'mermaid_reg_loss': 0.,
            'dice_SmLabel_in_CB': 0.,
            'dice_SdLabel_in_CB': 0.,
        }
        return

    def init_mermaid_env(self, spacing):
        params = pars.ParameterDict()
        params.load_JSON(self.mermaid_config_file)
        model_name = params['model']['registration_model']['type']
        self.use_map = params['model']['deformation']['use_map']
        self.map_low_res_factor = params['model']['deformation'][('map_low_res_factor', None, 'low_res_factor')]
        assert self.map_low_res_factor == 0.5
        compute_similarity_measure_at_low_res = params['model']['deformation'][
            ('compute_similarity_measure_at_low_res', False, 'to compute Sim at lower resolution')]

        # Currently Must use map_low_res_factor = 0.5
        if self.map_low_res_factor is not None:
            self.lowResSize = _get_low_res_size_from_size(self.img_sz, self.map_low_res_factor)
            self.lowResSpacing = _get_low_res_spacing_from_spacing(spacing, self.img_sz, self.lowResSize)
            if compute_similarity_measure_at_low_res:
                mf = py_mf.ModelFactory(self.lowResSize, self.lowResSpacing, self.lowResSize, self.lowResSpacing)
            else:
                mf = py_mf.ModelFactory(self.img_sz, self.spacing, self.lowResSize, self.lowResSpacing)
        else:
            raise ValueError("map_low_res_factor not defined")
        model, criterion = mf.create_registration_model(model_name, params['model'], compute_inverse_map=False)

        criterion.add_similarity_measure('custlncc', CustLNCCSimilarity)
        # Currently Must use map
        if self.use_map:
            # create the identity map [0,1]^d, since we will use a map-based implementation
            _id = py_utils.identity_map_multiN(self.img_sz, spacing)
            self.identityMap = torch.from_numpy(_id).cuda()
            if self.map_low_res_factor is not None:
                # create a lower resolution map for the computations
                lowres_id = py_utils.identity_map_multiN(self.lowResSize, self.lowResSpacing)
                self.lowResIdentityMap = torch.from_numpy(lowres_id).cuda()

        # SVFVectorMometumMapNet, LDDMMShootingVectorMomentumMapNet
        self.mermaid_unit = model.cuda()
        self.mermaid_criterion = criterion
        print("Spacing: {}".format(self.spacing))
        print("LowResSize: {}".format(self.lowResSize))
        print("LowResIdentityMap Shape: {}".format(self.lowResIdentityMap.shape))
        self.lowRes_fn = partial(get_resampled_image, spacing=self.spacing, desired_size=self.lowResSize,
                                 zero_boundary=False, identity_map=self.lowResIdentityMap)
        return

    def __calculate_dice_score(self, predicted_label, target_label, mask=None):
        if mask is None:
            predicted_in_mask = predicted_label
            target_in_mask = target_label
        else:
            roi_indices = (mask > 0.5)
            predicted_in_mask = predicted_label[roi_indices]
            target_in_mask = target_label[roi_indices]
        intersection = (predicted_in_mask * target_in_mask).sum()
        smooth = 1.
        dice = (2 * intersection + smooth) / (predicted_in_mask.sum() + target_in_mask.sum() + smooth)
        return dice

    def __calculate_dice_score_multiN(self, predicted_labels, target_labels, mask=None):
        sm_label_dice = 0.0
        sd_label_dice = 0.0
        for batch in range(self.batch_size):
            if mask is None:
                sm_label_dice += self.__calculate_dice_score(predicted_labels[batch, 0, ...],
                                                             target_labels[batch, 0, ...])
                sd_label_dice += self.__calculate_dice_score(predicted_labels[batch, 1, ...],
                                                             target_labels[batch, 1, ...])
            else:
                sm_label_dice += self.__calculate_dice_score(predicted_labels[batch, 0, ...],
                                                             target_labels[batch, 0, ...],
                                                             mask[batch, 0, ...])
                sd_label_dice += self.__calculate_dice_score(predicted_labels[batch, 1, ...],
                                                             target_labels[batch, 1, ...],
                                                             mask[batch, 0, ...])
        return sm_label_dice, sd_label_dice

    def calculate_train_loss(self, moving_image_and_label, target_image_and_label):
        mermaid_all_loss, mermaid_sim_loss, mermaid_reg_loss = self.mermaid_criterion(
            phi0=self.identityMap,
            phi1=self.phi,
            I0_source=moving_image_and_label,
            I1_target=target_image_and_label,
            lowres_I0=None,
            variables_from_forward_model=self.mermaid_unit.get_variables_to_transfer_to_loss_function(),
            variables_from_optimizer=None
        )

        return mermaid_all_loss, mermaid_sim_loss, mermaid_reg_loss

    def forward(self, ct_image, cb_image, roi_label, ct_sblabel, ct_sdlabel):
        # for mermaid purpose, convert to [0,1] (optional?)
        ct_image_n = (ct_image + 1) / 2.

        # concatenacte ct labels
        ct_labels = torch.cat((ct_sblabel, ct_sdlabel), dim=1)
        init_map = self.identityMap

        momentum = self.__network_forward__(ct_image, cb_image, ct_labels, roi_label)
        warped_moving_image_n = self.__mermaid_shoot__(moving_image_n=ct_image_n,
                                                       moving_labels=ct_labels,
                                                       momentum=momentum,
                                                       init_map=init_map)

        # warped_image_n : [0, 1], warped_image: [-1, 1]
        self.warped_moving_image = warped_moving_image_n * 2 - 1
        return

    def calculate_loss(self, ct_image, cb_image, roi_label, ct_sblabel, ct_sdlabel, cb_sblabel, cb_sdlabel):
        # cb_image: [-1, 1], cb_image_n: [0, 1]
        # ct: same
        # training (mermaid) loss and dice loss
        # only training (mermaid) loss backpropagates
        ct_image_n = (ct_image + 1) / 2.
        cb_image_n = (cb_image + 1) / 2.

        cb_labels = torch.cat((cb_sblabel, cb_sdlabel), dim=1)

        # moving: CT, target: CBCT
        cb_image_n_and_label = torch.cat((cb_image_n, cb_sblabel, cb_sdlabel), dim=1)
        ct_image_n_and_label = torch.cat((ct_image_n, ct_sblabel, ct_sdlabel), dim=1)

        # add ROI to target space
        cb_image_n_and_label = torch.cat((cb_image_n_and_label, roi_label), dim=1)

        all_loss, sim_loss, reg_loss = self.calculate_train_loss(moving_image_and_label=ct_image_n_and_label,
                                                                 target_image_and_label=cb_image_n_and_label)
        self.loss_dict['mermaid_all_loss'] = all_loss / self.batch_size
        self.loss_dict['mermaid_sim_loss'] = sim_loss / self.batch_size
        self.loss_dict['mermaid_reg_loss'] = reg_loss / self.batch_size

        self.loss_dict['all_loss'] = self.loss_dict['mermaid_all_loss']

        # dice evaluated in the cb space, roi
        sm_label_dice, sd_label_dice = self.__calculate_dice_score_multiN(self.warped_moving_labels.detach(), cb_labels,
                                                                          roi_label)
        self.loss_dict['dice_SmLabel_in_CB'] = sm_label_dice / self.batch_size
        self.loss_dict['dice_SdLabel_in_CB'] = sd_label_dice / self.batch_size
        return

    def __mermaid_shoot__(self, moving_image_n, moving_labels, momentum, init_map):
        # obtain transformation map from momentum
        # warp moving image and labels
        self.mermaid_unit.m = momentum
        self.mermaid_criterion.m = momentum
        # low resolution phi
        low_res_phi = self.mermaid_unit(self.lowRes_fn(init_map), I0_source=moving_image_n)
        desired_sz = self.identityMap.size()[2:]
        # upsample phi to get original phi (match image size)
        self.phi = get_resampled_image(low_res_phi,
                                       self.lowResSpacing,
                                       desired_sz, 1,
                                       zero_boundary=False,
                                       identity_map=self.identityMap)

        # warp moving image and labels
        warped_moving_image_n = py_utils.compute_warped_image_multiNC(moving_image_n,
                                                                      self.phi,
                                                                      self.spacing,
                                                                      spline_order=1,
                                                                      zero_boundary=True)
        self.warped_moving_labels = py_utils.compute_warped_image_multiNC(moving_labels,
                                                                          self.phi,
                                                                          self.spacing,
                                                                          spline_order=0,
                                                                          zero_boundary=True)

        return warped_moving_image_n

    def __network_forward__(self, ct_image, cb_image, ct_labels, roi_label):
        x1 = torch.cat((ct_image, ct_labels, roi_label), dim=1)
        x1 = self.ec_1(x1)
        x2 = self.ec_2(cb_image)
        x_l1 = torch.cat((x1, x2), dim=1)
        x = self.ec_3(x_l1)
        x = self.ec_4(x)
        x_l2 = self.ec_5(x)
        x = self.ec_6(x_l2)
        x = self.ec_7(x)
        x_l3 = self.ec_8(x)
        x = self.ec_9(x_l3)
        x = self.ec_10(x)

        # Decode Momentum
        x = self.dc_17(x)
        x = torch.cat((x_l3, x), dim=1)
        x = self.dc_18(x)
        x = self.dc_19(x)
        x = self.dc_20(x)
        x = torch.cat((x_l2, x), dim=1)
        x = self.dc_21(x)
        x = self.dc_22(x)
        output = self.dc_23(x)
        return output

    def __setup_network_structure__(self):
        k = self.n_of_feature
        self.ec_1 = ConBnRelDp(4, k, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                               use_bn=self.use_bn, use_dp=self.use_dp, p=0.1)
        self.ec_2 = ConBnRelDp(1, k, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                               use_bn=self.use_bn, use_dp=self.use_dp, p=0.1)
        self.ec_3 = ConBnRelDp(2*k, 2*k, kernel_size=3, stride=2, dim=self.dim, activate_unit='leaky_relu',
                               use_bn=self.use_bn, use_dp=self.use_dp, p=self.dp_p)
        self.ec_4 = ConBnRelDp(2*k, 4*k, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                               use_bn=self.use_bn, use_dp=self.use_dp, p=self.dp_p)
        self.ec_5 = ConBnRelDp(4*k, 4*k, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                               use_bn=self.use_bn, use_dp=self.use_dp, p=self.dp_p)
        self.ec_6 = MaxPool(2, dim=self.dim)
        self.ec_7 = ConBnRelDp(4*k, 8*k, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                               use_bn=self.use_bn, use_dp=self.use_dp, p=self.dp_p)
        self.ec_8 = ConBnRelDp(8*k, 8*k, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                               use_bn=self.use_bn, use_dp=self.use_dp, p=self.dp_p)
        self.ec_9 = MaxPool(2, dim=self.dim)
        self.ec_10 = ConBnRelDp(8*k, 16*k, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp, p=self.dp_p)
        # decoder for momentum
        self.dc_17 = ConBnRelDp(16*k, 8*k, kernel_size=2, stride=2, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp, p=self.dp_p, reverse=True)
        self.dc_18 = ConBnRelDp(16*k, 8*k, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp, p=self.dp_p)
        self.dc_19 = ConBnRelDp(8*k, 8*k, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp, p=self.dp_p)
        self.dc_20 = ConBnRelDp(8*k, 4*k, kernel_size=2, stride=2, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp, p=self.dp_p, reverse=True)
        self.dc_21 = ConBnRelDp(8*k, 4*k, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp, p=self.dp_p)
        self.dc_22 = ConBnRelDp(4*k, 2*k, kernel_size=3, stride=1, dim=self.dim, activate_unit='None')
        self.dc_23 = ConBnRelDp(2*k, self.dim, kernel_size=3, stride=1, dim=self.dim, activate_unit='None')

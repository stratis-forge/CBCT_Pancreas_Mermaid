import os
import sys
import numpy as np
import torch

mermaid_path = os.path.join(os.path.dirname(__file__), '../mermaid')
sys.path.append(mermaid_path)
sys.path.append(os.path.join(mermaid_path, 'mermaid'))
sys.path.append(os.path.join(mermaid_path, 'mermaid/libraries'))

import mermaid.fileio as py_fio
import mermaid.image_sampling as py_is
import mermaid.model_factory as py_mf
import mermaid.simple_interface as py_si
import mermaid.utils as py_utils
import mermaid.visualize_registration_results as py_visreg
import mermaid.module_parameters as pars
from mermaid.data_wrapper import AdaptVal


def evaluate_momentum(moving_image, target_image, target_spacing, momentum, registration_param_file, islabel=False):
    shared_parameters=dict()
    params=pars.ParameterDict()
    print(sys.path[0])
    print(registration_param_file)
    params.load_JSON(registration_param_file)

    momentum_dict=dict()
    #momentum_dict['m'] = torch.from_numpy(momentum)
    momentum_dict['m'] = momentum
    warped_image,map_from_momentum, _ = evaluate_model(
        ISource_in = moving_image,
        ITarget_in = target_image,
        sz = np.shape(target_image),
        spacing=target_spacing,
        individual_parameters=momentum_dict,
        shared_parameters=shared_parameters,
        params=params,
        visualize=False,
        compute_inverse_map=False,
        islabel=islabel
    )

    return warped_image, map_from_momentum


def image_pair_registration(moving_images_w_masks, target_images_w_masks, target_image_spacing, map_resolution, result_folder, registration_param_file):
    """
    :param moving_image_paths: path for moving image
    :param target_image_paths: path for target image
    :param result_image_paths: path for result warped image, default None
    :param result_deformation_paths: path for deformation, default None
    :return: moving_images, target_images, momentum
    """
    si = py_si.RegisterImagePair()
    si.register_images(moving_images_w_masks, target_images_w_masks, target_image_spacing,
                       visualize_step=None,
                       model_name='svf_vector_momentum_map',
                       use_multi_scale=False,
                       use_batch_optimization=False,
                       json_config_out_filename=os.path.join(result_folder, 'mermaid_config.json'),
                       params=registration_param_file,
                       map_low_res_factor=map_resolution)

    warped_images_w_masks = si.get_warped_image()
    deformation_map = si.get_map()
    model_pars = si.get_model_parameters()
    momentum = model_pars['m'].cpu().data.numpy()
    return warped_images_w_masks, deformation_map, momentum


def _compute_low_res_image(I,spacing,low_res_size, lowResId):
    sampler = py_is.ResampleImage()
    low_res_image, _ = sampler.downsample_image_to_size(I, spacing, low_res_size[2::], lowResId ,1)
    return low_res_image


def _get_low_res_size_from_size(sz, factor):
    """
    Returns the corresponding low-res size from a (high-res) sz
    :param sz: size (high-res)
    :param factor: low-res factor (needs to be <1)
    :return: low res size
    """
    if (factor is None) or (factor>=1):
        print('WARNING: Could not compute low_res_size as factor was ' + str( factor ))
        return sz
    else:
        lowResSize = np.array(sz)
        lowResSize[2::] = (np.ceil((np.array(sz[2::]) * factor))).astype('int16')

        #if lowResSize[-1]%2!=0:
        #    lowResSize[-1]-=1
        #    print('\n\nWARNING: forcing last dimension to be even: fix properly in the Fourier transform later!\n\n')

        return lowResSize


def _get_low_res_spacing_from_spacing(spacing, sz, lowResSize):
    """
    Computes spacing for the low-res parameterization from image spacing
    :param spacing: image spacing
    :param sz: size of image
    :param lowResSize: size of low re parameterization
    :return: returns spacing of low res parameterization
    """
    #todo: check that this is the correct way of doing it
    return spacing * (np.array(sz[2::])-1) / (np.array(lowResSize[2::])-1)


def resample_image(I, spacing, desired_size, spline_order=1, zero_boundary=False, identity_map=None):
    """
    Resample an image to a given desired size
    :param I: Input image (expected to be of BxCxXxYxZ format)
    :param spacing: array describing the spatial spacing
    :param desired_size: array for the desired size (excluding B and C, i.e, 1 entry for 1D, 2 for 2D, and 3 for 3D)
    :return: returns a tuple: the downsampled image, the new spacing after downsampling
    """
    desiredSize = desired_size[2:]
    sz = np.array(list(I.size()))
    # check that the batch size and the number of channels is the same
    nrOfI = sz[0]
    nrOfC = sz[1]
    desired_size_NC = np.array([nrOfI, nrOfC] + list(desiredSize))

    new_spacing = spacing * ((sz[2::].astype('float') - 1.) / (
                desired_size_NC[2::].astype('float') - 1.))  ###########################################
    if identity_map is not None:
        idDes = identity_map
    else:
        idDes = AdaptVal(torch.from_numpy(py_utils.identity_map_multiN(desired_size_NC, new_spacing)))
    # now use this map for resampling
    ID = py_utils.compute_warped_image_multiNC(I, idDes, new_spacing, spline_order, zero_boundary)

    return ID, new_spacing


def get_resampled_image(I, spacing, desired_size, spline_order=1, zero_boundary=False, identity_map=None):
    """
    :param I:  B C X Y Z
    :param spacing: spx spy spz
    :param desiredSize: B C X Y Z
    :param spline_order:
    :param zero_boundary:
    :param identity_map:
    :return:
    """
    if spacing is None:
        img_sz = I.shape[2:]
        spacing = 1. / (np.array(img_sz) - 1)
    if identity_map is not None:  # todo  will remove, currently fix for symmetric training
        if I.shape[0] != identity_map.shape[0]:
            n_batch = I.shape[0]
            desiredSize = desired_size.copy()
            desiredSize[0] = n_batch
            identity_map = identity_map[:n_batch]
    resampled, new_spacing = resample_image(I, spacing, desired_size, spline_order=spline_order,
                                            zero_boundary=zero_boundary, identity_map=identity_map)
    return resampled

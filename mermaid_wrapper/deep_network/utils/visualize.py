import os
import sys
import torch
import numpy as np
import torchvision.utils as vision_utils
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2

plt.rcParams.update({'figure.max_open_warning': 0})


def generate_deform_grid(transform, slice_axis=0, background_image=None, dim=3):
    if isinstance(transform, torch.Tensor):
        transform = transform.cpu().numpy()
    if background_image is not None:
        if isinstance(background_image, torch.Tensor):
            background_image = background_image.cpu().numpy()
        assert background_image.shape[1:] == transform.shape[1:]

    if dim == 2:
        left_axis = [0, 1]
    elif dim == 3:
        left_axis = [0, 1, 2]
        left_axis.remove(slice_axis)
    else:
        raise ValueError("Dimension error")

    fig = plt.figure(figsize=np.array(transform.shape[1:]) / 5, dpi=10)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.axis('equal')
    if background_image is not None:
        ax.imshow(background_image.squeeze(), cmap='gray')
    for i, axis in enumerate(left_axis):
        T_slice = transform[axis, :, :]
        ax.contour(T_slice, colors=['red'], linewidths=10.0, linestyles='solid', levels=np.linspace(-1, 1, 40))
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3) / 255

    return np.transpose(image, [2, 0, 1])


def make_image_summary(images_to_show, labels_to_show, phis_to_show, n_samples=1):
    n_samples = min(n_samples, images_to_show['ct_image'].size()[0])
    dim = len(images_to_show['ct_image'].shape) - 2

    image_slices_to_show = []
    label_slices_to_show = []
    grid_slices_to_show = []
    grids = {}
    for n in range(n_samples):
        if dim == 3:
            for axis in range(1, 4):
                slice_idx = images_to_show['ct_image'].size()[axis + 1] // 2
                image_slices = []
                grid_slices = []
                label_slices = []

                image_dict = {}
                contour_dict = {}
                for image_name, image in images_to_show.items():
                    image_slice = torch.select(image[n, :, :, :, :], axis, slice_idx)
                    image_dict[image_name] = torch.squeeze(image_slice)
                    image_slices.append(image_slice)

                for label_name, label in labels_to_show.items():
                    label_slice = torch.squeeze(torch.select(label[n, :, :, :, :], axis, slice_idx))
                    _, contours, _ = cv2.findContours(label_slice.numpy().astype(np.uint8), mode=cv2.RETR_EXTERNAL,
                                                      method=cv2.CHAIN_APPROX_SIMPLE)
                    contour_dict[label_name] = contours

                for label_name, contours in contour_dict.items():
                    if label_name == 'roi_label':
                        continue
                    img = label_name.split('_')[0]
                    # label_slice = torch.squeeze(image_dict[img + '_image']).numpy().copy()
                    roi_contour = contour_dict['roi_label'] + contours
                    label_slice = cv2.drawContours(
                        cv2.cvtColor(
                            cv2.normalize(
                                torch.squeeze(image_dict[img + '_image']).numpy(),
                                None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1),
                            cv2.COLOR_GRAY2RGB)
                        , roi_contour, -1, (255, 255, 255), 1)
                    label_slices.append(torch.from_numpy(
                        cv2.cvtColor(label_slice, cv2.COLOR_RGB2GRAY)[np.newaxis, :].astype(np.float32)
                    ))
                for phi in phis_to_show:
                    phi_slice = torch.select(phi[n, :, :, :, :], axis, slice_idx)
                    grid_slice = torch.from_numpy(
                        generate_deform_grid(phi_slice, axis - 1, image_slices[2], dim=3)
                    )
                    grid_slices.append(grid_slice)
                label_slices_to_show += label_slices
                image_slices_to_show += image_slices
                grid_slices_to_show += grid_slices
        else:
            raise ValueError("dimension not supported")
        grids['labels'] = vision_utils.make_grid(label_slices_to_show, pad_value=1, nrow=len(label_slices),
                                                 normalize=True, range=None)
        grids['images'] = vision_utils.make_grid(image_slices_to_show, pad_value=1, nrow=len(image_slices),
                                                 normalize=True, range=None)
        if len(grid_slices_to_show) > 0:
            grids['grid'] = vision_utils.make_grid(grid_slices_to_show, pad_value=1, nrow=dim)
    return grids


def make_image_summary_old(moving_image, target_image, moving_warped, moving_warped_recons, deform_field=None,
                           n_samples=1):
    n_samples = min(n_samples, moving_image.size()[0])
    grids = {}
    dim = len(moving_image.shape) - 2

    image_slices = []
    deform_grid_slices = []
    grids = {}
    for n in range(n_samples):
        if dim == 2:
            moving_image_slice = moving_image[n, :, :, :]
            target_image_slice = target_image[n, :, :, :]
            moving_warped_slice = moving_warped[n, :, :, :]
            moving_warped_recons_slice = moving_warped_recons[n, :, :, :]
            diff_image_slice = torch.abs(moving_warped_slice - moving_warped_recons_slice)
            if deform_field is not None:
                deform_field_slice = deform_field[n, :, :, :]
                deform_grid_slice = torch.from_numpy(
                    generate_deform_grid(deform_field_slice, background_image=moving_warped_slice)
                )
                deform_grid_slices += [deform_grid_slice]

            image_slices += [moving_image_slice, target_image_slice, moving_warped_slice, moving_warped_recons_slice,
                             diff_image_slice]
        elif dim == 3:
            for axis in range(1, 4):
                slice_indx = moving_image.size()[axis + 1] // 2
                moving_image_slice = torch.flip(torch.select(moving_image[n, :, :, :, :], axis, slice_indx), dims=[1])
                target_image_slice = torch.flip(torch.select(target_image[n, :, :, :, :], axis, slice_indx), dims=[1])
                moving_warped_slice = torch.flip(torch.select(moving_warped[n, :, :, :, ], axis, slice_indx), dims=[1])
                moving_warped_recons_slice = torch.flip(
                    torch.select(moving_warped_recons[n, :, :, :, ], axis, slice_indx), dims=[1])
                diff_image_slice = torch.abs(moving_warped_slice - moving_warped_recons_slice)
                image_slices += [moving_image_slice, target_image_slice, moving_warped_slice,
                                 moving_warped_recons_slice, diff_image_slice]
        else:
            raise ValueError("dimension not supported")
        grids['images'] = vision_utils.make_grid(image_slices, pad_value=1, nrow=5, normalize=True, range=(0, 1))
        grids['images_un'] = vision_utils.make_grid(image_slices, pad_value=1, nrow=5)
        # grids['deform_grid'] = vision_utils.make_grid(deform_grid_slices, pad_value=1, nrow=1)
    return grids


def get_identity_transform_batch(size, normalize=True):
    """
    generate an identity transform for given image size (NxCxDxHxW)
    :param size: Batch, D,H,W size
    :param normalize: normalized index into [-1,1]
    :return: identity transform with size Nx3xDxHxW
    """
    _identity = get_identity_transform(size[2:], normalize)
    # return _identity.repeat(size[0], 1, 1, 1, 1)
    return _identity


def get_identity_transform(size, normalize=True):
    """

    :param size: D,H,W size
    :param normalize:
    :return: 3XDxHxW tensor
    """

    if normalize:
        xx, yy, zz = torch.meshgrid([torch.arange(0, size[k]).float() / (size[k] - 1) * 2.0 - 1 for k in [0, 1, 2]])
    else:
        xx, yy, zz = torch.meshgrid([torch.arange(0, size[k]) for k in [0, 1, 2]])
    _identity = torch.stack([zz, yy, xx])
    return _identity


def test_generate_deform_grid():
    transform3d = get_identity_transform_batch([1, 1, 180, 160, 160])
    idslice = transform3d[:, :, :, 80]
    bk_img = torch.ones(1, 180, 160) / 2
    image = generate_deform_grid(idslice, 0, bk_img)
    # scipy.misc.imsave('grid.jpg', image)
    print(image.shape)
    plt.figure()
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    test_generate_deform_grid()

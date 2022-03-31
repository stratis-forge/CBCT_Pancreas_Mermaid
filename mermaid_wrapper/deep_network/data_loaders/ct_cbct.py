import math
import os
import random

import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
from scipy import ndimage
import blosc
import multiprocessing
import progressbar as pb
import torch

blosc.set_nthreads(1)


def __nii2tensorarray__(data):
    [z, y, x] = data.shape
    new_data = np.reshape(data, [1, z, y, x])
    new_data = new_data.astype("float32")
    return new_data


def __random_center_crop__(data, label):
    from random import random
    """
    Random crop
    """
    target_indexs = np.where(label > 0)
    [img_d, img_h, img_w] = data.shape
    [max_D, max_H, max_W] = np.max(np.array(target_indexs), axis=1)
    [min_D, min_H, min_W] = np.min(np.array(target_indexs), axis=1)
    [target_depth, target_height, target_width] = np.array([max_D, max_H, max_W]) - np.array([min_D, min_H, min_W])
    Z_min = int((min_D - target_depth * 1.0 / 2) * random())
    Y_min = int((min_H - target_height * 1.0 / 2) * random())
    X_min = int((min_W - target_width * 1.0 / 2) * random())

    Z_max = int(img_d - ((img_d - (max_D + target_depth * 1.0 / 2)) * random()))
    Y_max = int(img_h - ((img_h - (max_H + target_height * 1.0 / 2)) * random()))
    X_max = int(img_w - ((img_w - (max_W + target_width * 1.0 / 2)) * random()))

    Z_min = np.max([0, Z_min])
    Y_min = np.max([0, Y_min])
    X_min = np.max([0, X_min])

    Z_max = np.min([img_d, Z_max])
    Y_max = np.min([img_h, Y_max])
    X_max = np.min([img_w, X_max])

    Z_min = int(Z_min)
    Y_min = int(Y_min)
    X_min = int(X_min)

    Z_max = int(Z_max)
    Y_max = int(Y_max)
    X_max = int(X_max)

    return data[Z_min: Z_max, Y_min: Y_max, X_min: X_max], label[Z_min: Z_max, Y_min: Y_max, X_min: X_max]


class R21RegDataset(Dataset):
    def __init__(self, settings, mode):
        super(R21RegDataset, self).__init__()
        self.mode = mode
        if self.mode == 'train':
            print("Reading training list{}".format(settings.train_list))
            with open(settings.train_list, 'r') as f:
                img_list = [line.strip() for line in f]
                if settings.use_val:
                    fold_num = int(len(img_list)/8)
                    self.img_list = img_list[0: 6*fold_num]
                else:
                    self.img_list = img_list
        elif self.mode == 'validate':
            print("Reading validate list {}".format(settings.train_list))
            with open(settings.train_list, 'r') as f:
                img_list = [line.strip() for line in f]
                fold_num = int(len(img_list)/8)
                self.img_list = img_list[6*fold_num:]
        elif self.mode == 'test':
            print("Reading test list {}".format(settings.test_list))
            with open(settings.test_list, 'r') as f:
                self.img_list = [line.strip() for line in f]
        self.num_of_workers = min(len(self.img_list), 16)
        print("Processing {} datas".format(len(self.img_list)))
        self.input_D = settings.input_D
        self.input_H = settings.input_H
        self.input_W = settings.input_W

        self.img_dict = {}
        if self.num_of_workers == 16:
            multi_threads = True
        else:
            multi_threads = False
        if not multi_threads:
            self.__single_thread_loading()
        else:
            self.__multi_threads_loading__()

    def __split_files__(self):
        index_list = list(range(len(self.img_list)))
        index_split = np.array_split(np.array(index_list), self.num_of_workers)
        split_list = []
        for i in range(self.num_of_workers):
            current_list = self.img_list[index_split[i][0]:index_split[i][0] + len(index_split[i])]
            split_list.append(current_list)
        return split_list

    def __load_images_and_compress__(self, image_list):
        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(image_list)).start()
        count = 0
        for idx in range(len(image_list)):
            ith_info = image_list[idx].split(" ")
            ct_img_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', ith_info[0])
            cb_img_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', ith_info[1])
            roi_lbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', ith_info[2])
            ct_sblbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..',  ith_info[3])
            ct_sdlbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..',  ith_info[4])
            cb_sblbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', ith_info[5])
            cb_sdlbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..',  ith_info[6])
            assert os.path.isfile(ct_img_name), "{} not exist".format(ct_img_name)
            assert os.path.isfile(cb_img_name), "{} not exist".format(cb_img_name)
            assert os.path.isfile(roi_lbl_name), "{} not exist".format(roi_lbl_name)
            assert os.path.isfile(ct_sblbl_name), "{} not exist".format(ct_sblbl_name)
            assert os.path.isfile(ct_sdlbl_name), "{} not exist".format(ct_sdlbl_name)
            assert os.path.isfile(cb_sblbl_name), "{} not exist".format(cb_sblbl_name)
            assert os.path.isfile(cb_sdlbl_name), "{} not exist".format(cb_sdlbl_name)

            ct_img_itk = sitk.ReadImage(ct_img_name)
            cb_img_itk = sitk.ReadImage(cb_img_name)
            roi_lbl_itk = sitk.ReadImage(roi_lbl_name)
            ct_sblbl_itk = sitk.ReadImage(ct_sblbl_name)
            ct_sdlbl_itk = sitk.ReadImage(ct_sdlbl_name)
            cb_sblbl_itk = sitk.ReadImage(cb_sblbl_name)
            cb_sdlbl_itk = sitk.ReadImage(cb_sdlbl_name)

            # data processing
            ct_img_arr = sitk.GetArrayFromImage(ct_img_itk)
            cb_img_arr = sitk.GetArrayFromImage(cb_img_itk)
            roi_lbl_arr = sitk.GetArrayFromImage(roi_lbl_itk)
            ct_sblbl_arr = sitk.GetArrayFromImage(ct_sblbl_itk)
            ct_sdlbl_arr = sitk.GetArrayFromImage(ct_sdlbl_itk)
            cb_sblbl_arr = sitk.GetArrayFromImage(cb_sblbl_itk)
            cb_sdlbl_arr = sitk.GetArrayFromImage(cb_sdlbl_itk)

            if self.mode == 'train':
                ct_img_arr, cb_img_arr, roi_lbl_arr, \
                ct_sblbl_arr, ct_sdlbl_arr, cb_sblbl_arr, cb_sdlbl_arr = \
                    self.__processing_training_data__(ct_img_arr, cb_img_arr, roi_lbl_arr,
                                                      ct_sblbl_arr, ct_sdlbl_arr, cb_sblbl_arr, cb_sdlbl_arr)
            elif self.mode == 'test' or self.mode == 'validate':
                ct_img_arr, cb_img_arr, roi_lbl_arr, \
                ct_sblbl_arr, ct_sdlbl_arr, cb_sblbl_arr, cb_sdlbl_arr = \
                    self.__processing_testing_data__(ct_img_arr, cb_img_arr, roi_lbl_arr,
                                                     ct_sblbl_arr, ct_sdlbl_arr, cb_sblbl_arr, cb_sdlbl_arr)
            else:
                raise ValueError("Mode Wrong! Only train and test are supported!")

            ct_img_arr = __nii2tensorarray__(ct_img_arr)
            cb_img_arr = __nii2tensorarray__(cb_img_arr)
            roi_lbl_arr = __nii2tensorarray__(roi_lbl_arr)
            ct_sblbl_arr = __nii2tensorarray__(ct_sblbl_arr)
            ct_sdlbl_arr = __nii2tensorarray__(ct_sdlbl_arr)
            cb_sblbl_arr = __nii2tensorarray__(cb_sblbl_arr)
            cb_sdlbl_arr = __nii2tensorarray__(cb_sdlbl_arr)

            self.img_dict[ct_img_name] = blosc.pack_array(ct_img_arr)
            self.img_dict[cb_img_name] = blosc.pack_array(cb_img_arr)
            self.img_dict[roi_lbl_name] = blosc.pack_array(roi_lbl_arr)
            self.img_dict[ct_sblbl_name] = blosc.pack_array(ct_sblbl_arr)
            self.img_dict[ct_sdlbl_name] = blosc.pack_array(ct_sdlbl_arr)
            self.img_dict[cb_sblbl_name] = blosc.pack_array(cb_sblbl_arr)
            self.img_dict[cb_sdlbl_name] = blosc.pack_array(cb_sdlbl_arr)
            count += 1
            pbar.update(count)
        pbar.finish()
        return

    def __single_thread_loading(self):
        for idx in range(len(self.img_list)):
            ith_info = self.img_list[idx].split(" ")
            ct_img_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', ith_info[0])
            cb_img_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', ith_info[1])
            roi_lbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', ith_info[2])
            ct_sblbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..',  ith_info[3])
            ct_sdlbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..',  ith_info[4])
            cb_sblbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', ith_info[5])
            cb_sdlbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..',  ith_info[6])
            assert os.path.isfile(ct_img_name), "{} not exist".format(ct_img_name)
            assert os.path.isfile(cb_img_name), "{} not exist".format(cb_img_name)
            assert os.path.isfile(roi_lbl_name), "{} not exist".format(roi_lbl_name)
            assert os.path.isfile(ct_sblbl_name), "{} not exist".format(ct_sblbl_name)
            assert os.path.isfile(ct_sdlbl_name), "{} not exist".format(ct_sdlbl_name)
            assert os.path.isfile(cb_sblbl_name), "{} not exist".format(cb_sblbl_name)
            assert os.path.isfile(cb_sdlbl_name), "{} not exist".format(cb_sdlbl_name)

            ct_img_itk = sitk.ReadImage(ct_img_name)
            cb_img_itk = sitk.ReadImage(cb_img_name)
            roi_lbl_itk = sitk.ReadImage(roi_lbl_name)
            ct_sblbl_itk = sitk.ReadImage(ct_sblbl_name)
            ct_sdlbl_itk = sitk.ReadImage(ct_sdlbl_name)
            cb_sblbl_itk = sitk.ReadImage(cb_sblbl_name)
            cb_sdlbl_itk = sitk.ReadImage(cb_sdlbl_name)

            # data processing
            ct_img_arr = sitk.GetArrayFromImage(ct_img_itk)
            cb_img_arr = sitk.GetArrayFromImage(cb_img_itk)
            roi_lbl_arr = sitk.GetArrayFromImage(roi_lbl_itk)
            ct_sblbl_arr = sitk.GetArrayFromImage(ct_sblbl_itk)
            ct_sdlbl_arr = sitk.GetArrayFromImage(ct_sdlbl_itk)
            cb_sblbl_arr = sitk.GetArrayFromImage(cb_sblbl_itk)
            cb_sdlbl_arr = sitk.GetArrayFromImage(cb_sdlbl_itk)

            if self.mode == 'train':
                ct_img_arr, cb_img_arr, roi_lbl_arr, \
                ct_sblbl_arr, ct_sdlbl_arr, cb_sblbl_arr, cb_sdlbl_arr = \
                    self.__processing_training_data__(ct_img_arr, cb_img_arr, roi_lbl_arr,
                                                      ct_sblbl_arr, ct_sdlbl_arr, cb_sblbl_arr, cb_sdlbl_arr)
            elif self.mode == 'test' or self.mode == 'validate':
                ct_img_arr, cb_img_arr, roi_lbl_arr, \
                ct_sblbl_arr, ct_sdlbl_arr, cb_sblbl_arr, cb_sdlbl_arr = \
                    self.__processing_testing_data__(ct_img_arr, cb_img_arr, roi_lbl_arr,
                                                     ct_sblbl_arr, ct_sdlbl_arr, cb_sblbl_arr, cb_sdlbl_arr)
            else:
                raise ValueError("Mode Wrong! Only train and test are supported!")

            ct_img_arr = __nii2tensorarray__(ct_img_arr)
            cb_img_arr = __nii2tensorarray__(cb_img_arr)
            roi_lbl_arr = __nii2tensorarray__(roi_lbl_arr)
            ct_sblbl_arr = __nii2tensorarray__(ct_sblbl_arr)
            ct_sdlbl_arr = __nii2tensorarray__(ct_sdlbl_arr)
            cb_sblbl_arr = __nii2tensorarray__(cb_sblbl_arr)
            cb_sdlbl_arr = __nii2tensorarray__(cb_sdlbl_arr)

            self.img_dict[ct_img_name] = blosc.pack_array(ct_img_arr)
            self.img_dict[cb_img_name] = blosc.pack_array(cb_img_arr)
            self.img_dict[roi_lbl_name] = blosc.pack_array(roi_lbl_arr)
            self.img_dict[ct_sblbl_name] = blosc.pack_array(ct_sblbl_arr)
            self.img_dict[ct_sdlbl_name] = blosc.pack_array(ct_sdlbl_arr)
            self.img_dict[cb_sblbl_name] = blosc.pack_array(cb_sblbl_arr)
            self.img_dict[cb_sdlbl_name] = blosc.pack_array(cb_sdlbl_arr)

    def __multi_threads_loading__(self):
        manager = multiprocessing.Manager()
        self.img_dict = manager.dict()
        split_list = self.__split_files__()
        process = []
        for i in range(self.num_of_workers):
            p = multiprocessing.Process(target=self.__load_images_and_compress__, args=(split_list[i],))
            p.start()
            process.append(p)

        for p in process:
            p.join()
        self.img_dict = dict(self.img_dict)
        print("Finish loading images")
        return

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        if self.mode == "train":
            # read image and labels
            ith_info = self.img_list[idx].split(" ")
            ct_img_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', ith_info[0])
            cb_img_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', ith_info[1])
            roi_lbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', ith_info[2])
            ct_sblbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..',  ith_info[3])
            ct_sdlbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..',  ith_info[4])
            cb_sblbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', ith_info[5])
            cb_sdlbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..',  ith_info[6])

            ct_img_arr = blosc.unpack_array(self.img_dict[ct_img_name])
            cb_img_arr = blosc.unpack_array(self.img_dict[cb_img_name])
            roi_lbl_arr = blosc.unpack_array(self.img_dict[roi_lbl_name])
            ct_sblbl_arr = blosc.unpack_array(self.img_dict[ct_sblbl_name])
            ct_sdlbl_arr = blosc.unpack_array(self.img_dict[ct_sdlbl_name])
            cb_sblbl_arr = blosc.unpack_array(self.img_dict[cb_sblbl_name])
            cb_sdlbl_arr = blosc.unpack_array(self.img_dict[cb_sdlbl_name])

            # add random noise to images
            ct_img_arr += np.random.normal(0, 0.01, ct_img_arr.shape)
            cb_img_arr += np.random.normal(0, 0.01, cb_img_arr.shape)
            
            # random shift roi
            #shift = np.random.randint(-5, 6, (3,))
            #roi_lbl_arr = np.roll(roi_lbl_arr, shift=tuple(shift), axis=(0, 1, 2))

            return ct_img_arr, cb_img_arr, roi_lbl_arr, ct_sblbl_arr, ct_sdlbl_arr, cb_sblbl_arr, cb_sdlbl_arr

        elif self.mode == 'validate' or self.mode == "test":
            # read image and labels
            ith_info = self.img_list[idx].split(" ")
            ct_img_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', ith_info[0])
            cb_img_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', ith_info[1])
            roi_lbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', ith_info[2])
            ct_sblbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..',  ith_info[3])
            ct_sdlbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..',  ith_info[4])
            cb_sblbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', ith_info[5])
            cb_sdlbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..',  ith_info[6])

            ct_img_arr = blosc.unpack_array(self.img_dict[ct_img_name])
            cb_img_arr = blosc.unpack_array(self.img_dict[cb_img_name])
            roi_lbl_arr = blosc.unpack_array(self.img_dict[roi_lbl_name])
            ct_sblbl_arr = blosc.unpack_array(self.img_dict[ct_sblbl_name])
            ct_sdlbl_arr = blosc.unpack_array(self.img_dict[ct_sdlbl_name])
            cb_sblbl_arr = blosc.unpack_array(self.img_dict[cb_sblbl_name])
            cb_sdlbl_arr = blosc.unpack_array(self.img_dict[cb_sdlbl_name])

            return ct_img_arr, cb_img_arr, roi_lbl_arr, ct_sblbl_arr, ct_sdlbl_arr, cb_sblbl_arr, cb_sdlbl_arr

    def __drop_invalid_range__(self, volume, label=None):
        """
        Cut off the invalid area
        """
        zero_value = volume[0, 0, 0]
        non_zeros_idx = np.where(volume != zero_value)

        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)

        if label is not None:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w], label[min_z:max_z, min_h:max_h, min_w:max_w]
        else:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w]

    def __resize_data__(self, data, order=0):
        """
        Resize the data to the input size
        """
        [depth, height, width] = data.shape
        scale = [self.input_D * 1.0 / depth, self.input_H * 1.0 / height, self.input_W * 1.0 / width]
        data = ndimage.interpolation.zoom(data, scale, order=order)

        return data

    def __crop_data__(self, data, label):
        """
        Random crop with different methods:
        """
        # random center crop
        data, label = __random_center_crop__(data, label)

        return data, label

    def __processing_training_data__(self, ct_img_arr, cb_img_arr, roi_lbl_arr, ct_sblbl_arr, ct_sdlbl_arr,
                                     cb_sblbl_arr, cb_sdlbl_arr):
        # drop out the invalid range
        # data, label = self.__drop_invalid_range__(data, label)

        # crop data
        # data, label = self.__crop_data__(data, label)

        # resize data
        ct_img_arr = self.__resize_data__(ct_img_arr, order=3)
        cb_img_arr = self.__resize_data__(cb_img_arr, order=3)

        roi_lbl_arr = self.__resize_data__(roi_lbl_arr)
        ct_sblbl_arr = self.__resize_data__(ct_sblbl_arr)
        ct_sdlbl_arr = self.__resize_data__(ct_sdlbl_arr)
        cb_sblbl_arr = self.__resize_data__(cb_sblbl_arr)
        cb_sdlbl_arr = self.__resize_data__(cb_sdlbl_arr)

        return ct_img_arr, cb_img_arr, roi_lbl_arr, ct_sblbl_arr, ct_sdlbl_arr, cb_sblbl_arr, cb_sdlbl_arr

    def __processing_testing_data__(self, ct_img_arr, cb_img_arr, roi_lbl_arr, ct_sblbl_arr, ct_sdlbl_arr, cb_sblbl_arr,
                                    cb_sdlbl_arr):
        # resize data
        ct_img_arr = self.__resize_data__(ct_img_arr, order=3)
        cb_img_arr = self.__resize_data__(cb_img_arr, order=3)
        roi_lbl_arr = self.__resize_data__(roi_lbl_arr)
        ct_sblbl_arr = self.__resize_data__(ct_sblbl_arr)
        ct_sdlbl_arr = self.__resize_data__(ct_sdlbl_arr)
        cb_sblbl_arr = self.__resize_data__(cb_sblbl_arr)
        cb_sdlbl_arr = self.__resize_data__(cb_sdlbl_arr)

        return ct_img_arr, cb_img_arr, roi_lbl_arr, ct_sblbl_arr, ct_sdlbl_arr, cb_sblbl_arr, cb_sdlbl_arr

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


class R21RegDataset(Dataset):
    def __init__(self, settings):
        super(R21RegDataset, self).__init__()

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
            ct_sblbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', ith_info[3])
            ct_sdlbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', ith_info[4])
            assert os.path.isfile(ct_img_name), "{} not exist".format(ct_img_name)
            assert os.path.isfile(cb_img_name), "{} not exist".format(cb_img_name)
            assert os.path.isfile(roi_lbl_name), "{} not exist".format(roi_lbl_name)
            assert os.path.isfile(ct_sblbl_name), "{} not exist".format(ct_sblbl_name)
            assert os.path.isfile(ct_sdlbl_name), "{} not exist".format(ct_sdlbl_name)

            ct_img_itk = sitk.ReadImage(ct_img_name)
            cb_img_itk = sitk.ReadImage(cb_img_name)
            roi_lbl_itk = sitk.ReadImage(roi_lbl_name)
            ct_sblbl_itk = sitk.ReadImage(ct_sblbl_name)
            ct_sdlbl_itk = sitk.ReadImage(ct_sdlbl_name)

            # data processing
            ct_img_arr = sitk.GetArrayFromImage(ct_img_itk)
            cb_img_arr = sitk.GetArrayFromImage(cb_img_itk)
            roi_lbl_arr = sitk.GetArrayFromImage(roi_lbl_itk)
            ct_sblbl_arr = sitk.GetArrayFromImage(ct_sblbl_itk)
            ct_sdlbl_arr = sitk.GetArrayFromImage(ct_sdlbl_itk)

            ct_img_arr, cb_img_arr, roi_lbl_arr, \
            ct_sblbl_arr, ct_sdlbl_arr = \
                self.__processing_testing_data__(ct_img_arr, cb_img_arr, roi_lbl_arr,
                                                 ct_sblbl_arr, ct_sdlbl_arr)

            ct_img_arr = __nii2tensorarray__(ct_img_arr)
            cb_img_arr = __nii2tensorarray__(cb_img_arr)
            roi_lbl_arr = __nii2tensorarray__(roi_lbl_arr)
            ct_sblbl_arr = __nii2tensorarray__(ct_sblbl_arr)
            ct_sdlbl_arr = __nii2tensorarray__(ct_sdlbl_arr)

            self.img_dict[ct_img_name] = blosc.pack_array(ct_img_arr)
            self.img_dict[cb_img_name] = blosc.pack_array(cb_img_arr)
            self.img_dict[roi_lbl_name] = blosc.pack_array(roi_lbl_arr)
            self.img_dict[ct_sblbl_name] = blosc.pack_array(ct_sblbl_arr)
            self.img_dict[ct_sdlbl_name] = blosc.pack_array(ct_sdlbl_arr)
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
            ct_sblbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', ith_info[3])
            ct_sdlbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', ith_info[4])
            assert os.path.isfile(ct_img_name), "{} not exist".format(ct_img_name)
            assert os.path.isfile(cb_img_name), "{} not exist".format(cb_img_name)
            assert os.path.isfile(roi_lbl_name), "{} not exist".format(roi_lbl_name)
            assert os.path.isfile(ct_sblbl_name), "{} not exist".format(ct_sblbl_name)
            assert os.path.isfile(ct_sdlbl_name), "{} not exist".format(ct_sdlbl_name)

            ct_img_itk = sitk.ReadImage(ct_img_name)
            cb_img_itk = sitk.ReadImage(cb_img_name)
            roi_lbl_itk = sitk.ReadImage(roi_lbl_name)
            ct_sblbl_itk = sitk.ReadImage(ct_sblbl_name)
            ct_sdlbl_itk = sitk.ReadImage(ct_sdlbl_name)

            # data processing
            ct_img_arr = sitk.GetArrayFromImage(ct_img_itk)
            cb_img_arr = sitk.GetArrayFromImage(cb_img_itk)
            roi_lbl_arr = sitk.GetArrayFromImage(roi_lbl_itk)
            ct_sblbl_arr = sitk.GetArrayFromImage(ct_sblbl_itk)
            ct_sdlbl_arr = sitk.GetArrayFromImage(ct_sdlbl_itk)

            ct_img_arr, cb_img_arr, roi_lbl_arr, \
            ct_sblbl_arr, ct_sdlbl_arr = self.__processing_testing_data__(ct_img_arr, cb_img_arr, roi_lbl_arr,
                                                                          ct_sblbl_arr, ct_sdlbl_arr)

            ct_img_arr = __nii2tensorarray__(ct_img_arr)
            cb_img_arr = __nii2tensorarray__(cb_img_arr)
            roi_lbl_arr = __nii2tensorarray__(roi_lbl_arr)
            ct_sblbl_arr = __nii2tensorarray__(ct_sblbl_arr)
            ct_sdlbl_arr = __nii2tensorarray__(ct_sdlbl_arr)

            self.img_dict[ct_img_name] = blosc.pack_array(ct_img_arr)
            self.img_dict[cb_img_name] = blosc.pack_array(cb_img_arr)
            self.img_dict[roi_lbl_name] = blosc.pack_array(roi_lbl_arr)
            self.img_dict[ct_sblbl_name] = blosc.pack_array(ct_sblbl_arr)
            self.img_dict[ct_sdlbl_name] = blosc.pack_array(ct_sdlbl_arr)

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
        # read image and labels
        ith_info = self.img_list[idx].split(" ")
        ct_img_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', ith_info[0])
        cb_img_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', ith_info[1])
        roi_lbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', ith_info[2])
        ct_sblbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', ith_info[3])
        ct_sdlbl_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', ith_info[4])

        ct_img_arr = blosc.unpack_array(self.img_dict[ct_img_name])
        cb_img_arr = blosc.unpack_array(self.img_dict[cb_img_name])
        roi_lbl_arr = blosc.unpack_array(self.img_dict[roi_lbl_name])
        ct_sblbl_arr = blosc.unpack_array(self.img_dict[ct_sblbl_name])
        ct_sdlbl_arr = blosc.unpack_array(self.img_dict[ct_sdlbl_name])

        return ct_img_arr, cb_img_arr, roi_lbl_arr, ct_sblbl_arr, ct_sdlbl_arr

    def __resize_data__(self, data, order=0):
        """
        Resize the data to the input size
        """
        [depth, height, width] = data.shape
        scale = [self.input_D * 1.0 / depth, self.input_H * 1.0 / height, self.input_W * 1.0 / width]
        data = ndimage.interpolation.zoom(data, scale, order=order)
        return data

    def __processing_testing_data__(self, ct_img_arr, cb_img_arr, roi_lbl_arr, ct_sblbl_arr, ct_sdlbl_arr):
        # resize data
        ct_img_arr = self.__resize_data__(ct_img_arr, order=3)
        cb_img_arr = self.__resize_data__(cb_img_arr, order=3)
        roi_lbl_arr = self.__resize_data__(roi_lbl_arr)
        ct_sblbl_arr = self.__resize_data__(ct_sblbl_arr)
        ct_sdlbl_arr = self.__resize_data__(ct_sdlbl_arr)

        return ct_img_arr, cb_img_arr, roi_lbl_arr, ct_sblbl_arr, ct_sdlbl_arr

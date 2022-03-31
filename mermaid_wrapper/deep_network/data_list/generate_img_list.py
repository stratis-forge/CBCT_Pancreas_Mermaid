import os
import sys
import glob
import numpy as np
from sklearn.model_selection import KFold
from operator import itemgetter
import socket

def generate_img_list(patients, indices, mode='train'):
    patients = itemgetter(*indices)(patients)
    img_line = ""
    for patient in patients:
        ct_lists = sorted(glob.glob(os.path.join(patient, 'Processed', 'planCT*')))
        cb_lists = sorted(glob.glob(os.path.join(patient, 'Processed', 'CBCT*')))
        roi_label = os.path.join(patient, 'Processed', 'planCT_OG', 'roi_label.nii.gz')
        assert os.path.isfile(roi_label)
        for ct_list in ct_lists:
            ct_image = os.path.join(ct_list, 'normalized_image.nii.gz')
            ct_sblabel = os.path.join(ct_list, 'SmBowel_label.nii.gz')
            ct_sdlabel = os.path.join(ct_list, 'StomachDuo_label.nii.gz')
            assert os.path.isfile(ct_image) and os.path.isfile(ct_sblabel) and os.path.isfile(ct_sdlabel)
            for cb_list in cb_lists:
                cb_image = os.path.join(cb_list, 'normalized_image.nii.gz')
                if mode == 'val':
                    if not 'OG' in ct_image or not 'OG' in cb_image:
                        continue
                cb_sblabel = os.path.join(cb_list, 'SmBowel_label.nii.gz')
                cb_sdlabel = os.path.join(cb_list, 'StomachDuo_label.nii.gz')
                assert os.path.isfile(cb_image) and os.path.isfile(cb_sblabel) and os.path.isfile(cb_sdlabel)
                img_line += "{} {} {} {} {} {} {}\n".format(ct_image, cb_image, roi_label, ct_sblabel, ct_sdlabel, cb_sblabel, cb_sdlabel)

    return img_line


if __name__ == '__main__':
    root_folder = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data'))
    patients = sorted(glob.glob(os.path.join(root_folder, '18227??')))

    print(len(patients))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1

    for train_index, test_index in kf.split(patients):
        print(train_index, test_index)
        train_line = generate_img_list(patients, train_index, 'train')
        train_file = "train_{}.txt".format(fold)
        train_f = open(train_file, 'w')
        train_f.write(train_line)
        train_f.close()

        test_line = generate_img_list(patients, test_index, 'val')
        test_file = "val_{}.txt".format(fold)
        test_f = open(test_file, 'w')
        test_f.write(test_line)
        test_f.close()
        fold += 1

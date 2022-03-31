import SimpleITK as sitk
import os
import glob
import numpy as np
import cv2
import argparse
import subprocess
from scipy import ndimage
import pandas as pd

#niftyreg_bin = "/home/xhs400/install/niftyreg-git/install/bin" # niftyreg location
niftyreg_bin = '/software/niftyreg/bin' # niftyreg location
data_folder = '/scratch'


#data_folder = os.path.normpath(os.path.join(__file__, "../../data")) # change this to a specific folder


class ProcessR21(object):
    def __init__(self, patient):
        self.patient = patient  # patient ID 18227??

        self.image_root_folder = os.path.join(data_folder, patient)  # folder to save processed images
        self.original_folder = os.path.join(self.image_root_folder, 'Original')
        self.processed_folder = os.path.join(self.image_root_folder, 'Processed')

        self.dicom_root_folder = os.path.join(data_folder, 'dicoms', patient)  # dicom files folder

        self.zip_root_folder = os.path.join(data_folder, 'zips')  # zip folder received from MSK
        assert os.path.isdir(self.zip_root_folder)
        self.is_special = False  # where neither CB case is spotlight
        if patient in ["1822709", "1822710", "1822749"]:
            self.is_special = True
        self.cases = []
        self.data_csv = pd.read_csv('image_setup.csv', header=None, dtype=str)
        return

    def process(self):
        self.__extract_dicom_files__()
        self.__get_all_cases__()
        print(self.cases)
        self.__rename_label_files__()
        self.__generate_image_masks__()
        self.__crop_and_resample_images__()
        self.__fill_gas_pockets__()
        self.__normalize_images__()

    def __normalize_images__(self):
        intensity_window_filter = sitk.IntensityWindowingImageFilter()
        intensity_window_filter.SetOutputMaximum(1.0)
        intensity_window_filter.SetOutputMinimum(-1.0)
        for image_folder in self.cases:
            processed_folder = os.path.join(self.processed_folder, image_folder)
            filled_file = os.path.join(processed_folder, 'filled_image.nii.gz')
            filled_itk = sitk.ReadImage(filled_file)

            if "planCT" in image_folder:
                image_case = "00"
            else:
                image_case = image_folder[4:6]
            print("image case {}".format(image_case))

            window_minimum = None
            window_maximum = None
            num_of_images = len(self.data_csv)
            for idx in range(num_of_images):
                c_patient = self.data_csv.iloc[idx, 0]
                c_case = self.data_csv.iloc[idx, 1]
                if c_patient == self.patient and c_case == image_case:
                    window_minimum = int(self.data_csv.iloc[idx, 4])
                    window_maximum = int(self.data_csv.iloc[idx, 5])
                    break
            assert window_maximum is not None and window_minimum is not None
            intensity_window_filter.SetWindowMaximum(window_maximum)
            intensity_window_filter.SetWindowMinimum(window_minimum)

            normalized_file = os.path.join(processed_folder, 'normalized_image.nii.gz')
            normalized_itk = intensity_window_filter.Execute(filled_itk)
            sitk.WriteImage(normalized_itk, normalized_file)

    def __fill_gas_pockets__(self):
        merge_filter = sitk.MergeLabelMapFilter()
        merge_filter.SetMethod(1)  # aggregate
        invert_filter = sitk.InvertIntensityImageFilter()
        invert_filter.SetMaximum(1)

        regional_minimal_filter = sitk.RegionalMinimaImageFilter()
        regional_minimal_filter.SetFlatIsMinima(False)
        regional_minimal_filter.SetFullyConnected(True)

        threshold_filter = sitk.BinaryThresholdImageFilter()
        connected_filter = sitk.ConnectedComponentImageFilter()
        connected_filter.SetFullyConnected(True)
        label_shape_statistics_filter = sitk.LabelShapeStatisticsImageFilter()
        confidence_filter = sitk.ConfidenceConnectedImageFilter()
        confidence_filter.SetInitialNeighborhoodRadius(0)
        confidence_filter.SetNumberOfIterations(1)

        mask_filter = sitk.MaskImageFilter()
        binary_filter = sitk.BinaryDilateImageFilter()

        gaussian_filter = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian_filter.SetSigma(2)
        for image_folder in self.cases:
            print("Filling gas pocket {}".format(image_folder))
            processed_folder = os.path.join(self.processed_folder, image_folder)
            image_file = os.path.join(processed_folder, 'original_image.nii.gz')
            mask_file = os.path.join(processed_folder, 'original_mask.nii.gz')
            leftlung_file = os.path.join(processed_folder, "LeftLung_label.nii.gz")
            rightlung_file = os.path.join(processed_folder, "RightLung_label.nii.gz")

            image_itk = sitk.ReadImage(image_file)
            mask_itk = sitk.ReadImage(mask_file)

            # all air (except potential pockets): outside value + lung
            air = [
                sitk.Cast(sitk.ReadImage(leftlung_file), sitk.sitkLabelUInt8),
                sitk.Cast(sitk.ReadImage(rightlung_file), sitk.sitkLabelUInt8),
                sitk.Cast(invert_filter.Execute(mask_itk), sitk.sitkLabelUInt8)
            ]
            binary_filter.SetKernelRadius([10, 10, 10])
            air_itk = binary_filter.Execute(sitk.Cast(merge_filter.Execute(air), sitk.sitkUInt8))

            if "planCT" in image_folder:
                image_case = "00"
            else:
                image_case = image_folder[4:6]
            print("image case {}".format(image_case))

            threshold_upper = None
            multiplier = None
            num_of_images = len(self.data_csv)
            for idx in range(num_of_images):
                c_patient = self.data_csv.iloc[idx, 0]
                c_case = self.data_csv.iloc[idx, 1]
                if c_patient == self.patient and c_case == image_case:
                    threshold_upper = int(self.data_csv.iloc[idx, 3])
                    multiplier = float(self.data_csv.iloc[idx, 2])
                    break
            assert threshold_upper is not None and multiplier is not None
            confidence_filter.SetMultiplier(multiplier)
            threshold_filter.SetUpperThreshold(threshold_upper)

            threshold_filter.SetLowerThreshold(-2000)
            # temporarily modify image (air part set zero, so in next part, algorithm won't find these regions)
            mask_filter.SetOutsideValue(0)
            modified_image_itk = mask_filter.Execute(image_itk, invert_filter.Execute(air_itk))
            threshold_itk = threshold_filter.Execute(modified_image_itk)
            # sitk.WriteImage(threshold_itk, os.path.join(processed_folder, 'threshold.nii.gz'))

            modified_image2_itk = mask_filter.Execute(image_itk, threshold_itk)
            regional_minimal_itk = regional_minimal_filter.Execute(modified_image2_itk)
            label_shape_statistics_filter.Execute(regional_minimal_itk)
            # sitk.WriteImage(regional_minimal_itk, os.path.join(processed_folder, 'regional_minimal.nii.gz'))
            ind = label_shape_statistics_filter.GetIndexes(1)
            seed_points = []
            for i in range(0, len(ind), 3):
                seed_points.append((ind[i], ind[i + 1], ind[i + 2]))

            confidence_filter.ClearSeeds()
            confidence_filter.SetSeedList(seed_points)
            gas_itk = confidence_filter.Execute(modified_image_itk)
            binary_filter.SetKernelRadius([2, 2, 2])
            gas_itk = binary_filter.Execute(gas_itk)
            # sitk.WriteImage(gas_itk, os.path.join(processed_folder, 'gas.nii.gz'))
            gas_arr = sitk.GetArrayFromImage(gas_itk)
            image_arr = sitk.GetArrayFromImage(image_itk)

            z, _, _ = image_arr.shape
            filled_arr = image_arr.copy()
            for i in range(z):
                gas_slice = gas_arr[i, ...]
                image_slice = image_arr[i, ...]
                if np.sum(gas_slice) > 0:
                    filled_arr[i, ...] = cv2.inpaint(image_slice, gas_slice, 2, cv2.INPAINT_NS)

            unsmooth_filled_itk = sitk.GetImageFromArray(filled_arr)
            unsmooth_filled_itk.CopyInformation(image_itk)
            # sitk.WriteImage(unsmooth_filled_itk, os.path.join(processed_folder, 'unsmoothed_filled_image.nii.gz'))
            smoothed_filled_itk = gaussian_filter.Execute(unsmooth_filled_itk)

            smoothed_filled_arr = sitk.GetArrayFromImage(smoothed_filled_itk)
            filled_arr[np.where(gas_arr == 1)] = smoothed_filled_arr[np.where(gas_arr == 1)]
            filled_itk = sitk.GetImageFromArray(filled_arr)
            filled_itk.CopyInformation(image_itk)

            filled_file = os.path.join(processed_folder, 'filled_image.nii.gz')
            sitk.WriteImage(filled_itk, filled_file)
        return

    def __generate_image_masks__(self):
        connected_filter = sitk.ConnectedComponentImageFilter()
        connected_filter.FullyConnectedOn()
        relabel_component_filter = sitk.RelabelComponentImageFilter()
        threshold_filter = sitk.ThresholdImageFilter()
        threshold_filter.SetUpper(1.5)
        threshold_filter.SetLower(0.5)

        binary_opening_filter = sitk.BinaryMorphologicalOpeningImageFilter()
        binary_opening_filter.SetKernelRadius([5, 5, 5])
        binary_closing_filter = sitk.BinaryMorphologicalClosingImageFilter()
        binary_closing_filter.SetKernelRadius([5, 5, 5])

        for image_folder in self.cases:
            print("Get image mask {}".format(image_folder))
            original_folder = os.path.join(self.original_folder, image_folder)
            image_file = os.path.join(original_folder, 'original_image.nii.gz')
            mask_file = os.path.join(original_folder, 'original_mask.nii.gz')
            image_itk = sitk.ReadImage(image_file)
            image_arr = sitk.GetArrayFromImage(image_itk)
            mask_arr = np.zeros_like(image_arr)
            mask_arr[np.where(image_arr > -900)] = 1
            mask_itk = sitk.GetImageFromArray(mask_arr)

            mask_itk = connected_filter.Execute(sitk.Cast(mask_itk, sitk.sitkUInt8))
            mask_itk = relabel_component_filter.Execute(mask_itk)
            mask_itk = threshold_filter.Execute(mask_itk)
            mask_itk = binary_opening_filter.Execute(mask_itk)
            mask_itk = binary_closing_filter.Execute(mask_itk)
            # mask_itk = binary_fillhole_filter.Execute(mask_itk)
            # fill hole 2D
            mask_arr = sitk.GetArrayFromImage(mask_itk)
            z, x, y = mask_arr.shape
            for i in range(z):
                mask_slice = mask_arr[i, :, :]
                mask_arr[i, :, :] = ndimage.morphology.binary_fill_holes(mask_slice)
            mask_itk = sitk.GetImageFromArray(mask_arr)
            mask_itk.CopyInformation(image_itk)
            sitk.WriteImage(mask_itk, mask_file)

        return

    def __crop_and_resample_images__(self):
        if not self.is_special:
            self.__align_template_to_cbct__()
        else:
            self.__get_special_templates__()
        template_folder = os.path.join(self.original_folder, 'template')
        bbox_file = os.path.join(template_folder, 'shifted_bbox.nii.gz')
        bbox_itk = sitk.ReadImage(bbox_file)

        statistic_filter = sitk.LabelShapeStatisticsImageFilter()
        statistic_filter.Execute(bbox_itk)
        bbox = np.array(statistic_filter.GetBoundingBox(1))

        resample_filter = sitk.ResampleImageFilter()
        for image_folder in self.cases:
            original_folder = os.path.join(self.original_folder, image_folder)
            processed_folder = os.path.join(self.processed_folder, image_folder)
            os.system('mkdir -p {}'.format(processed_folder))
            images_to_process = glob.glob(os.path.join(original_folder, '*.nii.gz'))
            for image in images_to_process:
                image_itk = sitk.ReadImage(image)
                cropped_itk = image_itk[bbox[0] + 1:bbox[0] + bbox[3] - 2, bbox[1] + 1:bbox[1] + bbox[4] - 2,
                              bbox[2] + 1:bbox[2] + bbox[5] - 2]
                spacing = cropped_itk.GetSpacing()
                sz = cropped_itk.GetSize()
                origin = cropped_itk.GetOrigin()
                physical_size = [round(a * b) for a, b in zip(spacing, sz)]

                resample_filter.SetSize(tuple(physical_size))
                resample_filter.SetOutputSpacing((1, 1, 1))
                resample_filter.SetOutputOrigin(origin)
                minmaxFilter = sitk.MinimumMaximumImageFilter()
                minmaxFilter.Execute(cropped_itk)
                resample_filter.SetDefaultPixelValue(minmaxFilter.GetMinimum())
                if "image.nii.gz" in os.path.basename(image):
                    resample_filter.SetInterpolator(sitk.sitkLinear)
                else:
                    resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
                resampled_image = resample_filter.Execute(cropped_itk)
                resampled_arr = sitk.GetArrayFromImage(resampled_image)
                resampled_arr = resampled_arr[2:-2, 2:-2, 2:-2]
                resampled_image = sitk.GetImageFromArray(resampled_arr)
                resampled_image.SetOrigin([0, 0, 0])

                # expand roi further for certain case:
                if os.path.basename(image) == "roi_label.nii.gz":
                    if self.patient in ['1822709', '1822715']:
                        dilate_filter = sitk.BinaryDilateImageFilter()
                        dilate_filter.SetKernelRadius([10, 10, 10])
                        resampled_image = dilate_filter.Execute(resampled_image)
                    elif patient in ['1822708', '1822734']:
                        dilate_filter = sitk.BinaryDilateImageFilter()
                        dilate_filter.SetKernelRadius([5, 5, 5])
                        resampled_image = dilate_filter.Execute(resampled_image)

                sitk.WriteImage(resampled_image, os.path.join(processed_folder, os.path.basename(image)))

            # convert label files to contour files
            label_files = glob.glob(os.path.join(processed_folder, '*_label.nii.gz'))
            for label_file in label_files:
                label_itk = sitk.Cast(sitk.ReadImage(label_file), sitk.sitkUInt8)
                label_arr = sitk.GetArrayFromImage(label_itk)
                z, _, _ = label_arr.shape
                contour_arr = np.zeros_like(label_arr)

                for i in range(z):
                    label_slice = label_arr[i, :, :]
                    contour_slice = np.zeros_like(label_slice, np.uint8)
                    if np.sum(label_slice) > 0:
                        _, contours, _ = cv2.findContours(label_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        cv2.polylines(contour_slice, contours, isClosed=True, color=1)
                        contour_arr[i] = contour_slice

                contour_itk = sitk.GetImageFromArray(contour_arr)
                contour_itk.CopyInformation(label_itk)
                contour_file = label_file.replace('_label.nii.gz', '_contour.nii.gz')
                sitk.WriteImage(contour_itk, contour_file)

            # combine stomachduo and smbowel lable/contour
            merge_filter = sitk.MergeLabelMapFilter()
            merge_filter.SetMethod(Method=0)

            labels = []
            labels.append(sitk.Cast(sitk.ReadImage(os.path.join(processed_folder, 'SmBowel_label.nii.gz')),
                                    sitk.sitkLabelUInt8))
            labels.append(sitk.Cast(sitk.ReadImage(os.path.join(processed_folder, 'StomachDuo_label.nii.gz')),
                                    sitk.sitkLabelUInt8))
            labels = sitk.Cast(merge_filter.Execute(labels), sitk.sitkUInt8)
            sitk.WriteImage(labels, os.path.join(processed_folder, 'oar_label.nii.gz'))

            contours = []
            contours.append(sitk.Cast(sitk.ReadImage(os.path.join(processed_folder, 'SmBowel_contour.nii.gz')),
                                      sitk.sitkLabelUInt8))
            contours.append(sitk.Cast(sitk.ReadImage(os.path.join(processed_folder, 'StomachDuo_contour.nii.gz')),
                                      sitk.sitkLabelUInt8))
            contours = sitk.Cast(merge_filter.Execute(contours), sitk.sitkUInt8)
            sitk.WriteImage(contours, os.path.join(processed_folder, 'oar_contour.nii.gz'))

        return

    def __align_template_to_cbct__(self):
        # Rigidly align template to one cbct image and then resample the template bounding box
        template_folder = os.path.join(self.original_folder, 'template')
        all_cb_folder = glob.glob(os.path.join(self.original_folder, 'CBCT*_OG'))
        cb_folder = all_cb_folder[0]
        if self.patient == '1822713':
            cb_folder = all_cb_folder[1]
        # first get a cb_mask
        cb_mask_file = os.path.join(cb_folder, 'original_mask.nii.gz')
        # niftyreg rigid registration, template to cbct_mask
        ori_outer = os.path.join(template_folder, 'Outer.nii.gz')
        res_outer = os.path.join(template_folder, 'shifted_outer.nii.gz')
        res_aff = os.path.join(template_folder, 'rigid.txt')
        ori_bbox = os.path.join(template_folder, 'BoundBoxLess32mm.nii.gz')
        res_bbox = os.path.join(template_folder, 'shifted_bbox.nii.gz')

        cmd = ""
        cmd += "\n" + "{}/reg_aladin -rigOnly -ref {} -flo {} -res {} -aff {}".format(niftyreg_bin, cb_mask_file,
                                                                                      ori_outer, res_outer, res_aff)
        cmd += "\n" + "{}/reg_resample -ref {} -flo {} -res {} -trans {}".format(niftyreg_bin, cb_mask_file, ori_bbox,
                                                                                 res_bbox, res_aff)

        process = subprocess.Popen(cmd, shell=True)
        process.wait()
        return

    def __get_special_templates__(self):
        # threshold CT image
        template_folder = os.path.join(self.original_folder, 'template')
        ct_file = os.path.join(template_folder, 'templateCT.nii.gz')
        ct_itk = sitk.ReadImage(ct_file)
        threshold_filter = sitk.BinaryThresholdImageFilter()
        threshold_filter.SetUpperThreshold(10000)  # any
        threshold_filter.SetLowerThreshold(-800)
        connected_filter = sitk.ConnectedComponentImageFilter()
        connected_filter.FullyConnectedOn()

        mask_itk = threshold_filter.Execute(ct_itk)
        connected_itk = connected_filter.Execute(mask_itk)
        label = 1
        statistic_filter = sitk.LabelShapeStatisticsImageFilter()
        statistic_filter.Execute(connected_itk)
        largest_label = 0
        largest_size = 0
        while statistic_filter.HasLabel(label):
            if statistic_filter.GetNumberOfPixels(label) > largest_size:
                largest_size = statistic_filter.GetNumberOfPixels(label)
                largest_label = label
            label = label + 1
        bbox = np.array(statistic_filter.GetBoundingBox(largest_label))

        mask_arr = sitk.GetArrayFromImage(mask_itk)
        template_arr = np.zeros_like(mask_arr)
        template_arr[
            bbox[2]: bbox[2] + bbox[5],
            bbox[1]: bbox[1] + bbox[4],
            bbox[0]: bbox[0] + bbox[3]
        ] = 1
        template_itk = sitk.GetImageFromArray(template_arr)
        template_itk.CopyInformation(mask_itk)
        res_bbox = os.path.join(template_folder, 'shifted_bbox.nii.gz')
        sitk.WriteImage(template_itk, res_bbox)
        return

    def __rename_label_files__(self):
        for case_folder in self.cases:
            # Need to process if ROI label is not available but PTV labels are given
            # if case_folder == 'planCT_OG':
            #     ptv = []
            image_folder = os.path.join(self.original_folder, case_folder)
            all_images = glob.glob(os.path.join(image_folder, '*.nii.gz'))
            for image in all_images:
                if "image.nii.gz" in os.path.basename(image):
                    continue
                matched = False
                if "LUNG_L" in os.path.basename(image).upper():
                    os.system('mv "' + image + '" ' + os.path.join(image_folder, 'LeftLung_label.nii.gz'))
                    if matched:
                        raise ValueError("image name matched multiple label criterion")
                    matched = True
                if "LUNG_R" in os.path.basename(image).upper():
                    os.system('mv "' + image + '" ' + os.path.join(image_folder, 'RightLung_label.nii.gz'))
                    if matched:
                        raise ValueError("image name matched multiple label criterion")
                    matched = True
                if "BOWE" in os.path.basename(image).upper():
                    os.system('mv "' + image + '" ' + os.path.join(image_folder, 'SmBowel_label.nii.gz'))
                    if matched:
                        raise ValueError("image name matched multiple label criterion")
                    matched = True
                if "STOM" in os.path.basename(image).upper():
                    os.system('mv "' + image + '" ' + os.path.join(image_folder, 'StomachDuo_label.nii.gz'))
                    if matched:
                        raise ValueError("image name matched multiple label criterion")
                    matched = True
                if "ROI" in os.path.basename(image).upper():
                    os.system('mv "' + image + '" ' + os.path.join(image_folder, 'roi_label.nii.gz'))
                    if matched:
                        raise ValueError("image name matched multiple label criterion")
                    matched = True
                # if case_folder == 'planCT_OG' and "PTV" in os.path.basename(image).upper():
                #     ptv.append(os.path.basename(image))
                #     if matched:
                #         raise ValueError("image name matched multiple label criterion")
                #     matched = True
                if not matched:
                    print(
                        "Image {} does not matched any useful label criterion, removed".format(os.path.basename(image)))
                    os.system('rm "' + image + '"')

    def __get_all_cases__(self):
        self.cases = []
        for folder in os.listdir(self.original_folder):
            if folder == 'template':
                continue
            self.cases.append(folder)
        return

    def __extract_dicom_files__(self):
        os.system('rm -r {}'.format(self.dicom_root_folder))
        os.system('rm -r {}'.format(self.image_root_folder))
        os.system('mkdir -p {}'.format(self.dicom_root_folder))
        os.system('mkdir -p {}'.format(self.image_root_folder))
        os.system('mkdir -p {}'.format(self.original_folder))
        os.system('mkdir -p {}'.format(self.processed_folder))
        # extract original
        zip_file = os.path.join(self.zip_root_folder, self.patient + '.zip')
        assert os.path.isfile(zip_file)
        print("Unzipping files.")
        os.system('unzip -qq ' + zip_file + ' ' + '-d' + ' ' + os.path.join(self.dicom_root_folder, '..'))  # extract

        # extract simulated
        zip_file = os.path.join(self.zip_root_folder, self.patient + '_sim.zip')
        if os.path.isfile(zip_file):
            os.system('unzip -qq ' + zip_file + ' ' + '-d' + ' ' + os.path.join(self.dicom_root_folder, '..'))  # extract

        # convert contours to images
        # update all cases
        for folder in os.listdir(self.dicom_root_folder):
            print(folder)
            if 'Studies' in folder:
                # rename if format is 19xx-01__Studies
                image_case = folder.split('-')[0][2:]
                print(image_case)
                if image_case == '00':
                    os.system('mv {} {}'.format(os.path.join(self.dicom_root_folder, folder),
                                                os.path.join(self.dicom_root_folder, 'planCT_OG')))
                    folder = "planCT_OG"
                else:
                    os.system('mv {} {}'.format(os.path.join(self.dicom_root_folder, folder),
                                                os.path.join(self.dicom_root_folder, 'CBCT{}_OG'.format(image_case))))
                    folder = "CBCT{}_OG".format(image_case)
            self.cases.append(folder)
            dicom_case_folder = os.path.join(self.dicom_root_folder, folder)
            assert os.path.isdir(dicom_case_folder)

            # convert dicom file to images
            image_case_folder = os.path.join(self.original_folder, folder)
            os.system('mkdir -p ' + image_case_folder)

            referenced = glob.glob(os.path.join(dicom_case_folder, '*_CT_*'))[0]
            print(referenced)
            for f in os.listdir(dicom_case_folder):
                f = os.path.join(dicom_case_folder, f)
                cmd = "plastimatch convert --input '{}' " \
                      "--output-img {} " \
                      "--output-prefix {} " \
                      "--prefix-format nii.gz" \
                      " --referenced-ct '{}'".format(f,
                                                     os.path.join(image_case_folder, 'original_image.nii.gz'),
                                                     image_case_folder,
                                                     referenced
                                                     )
                os.system(cmd)

        if not self.is_special:
            template_zip_file = os.path.join(self.zip_root_folder, 'templates', 'template.zip')
            assert os.path.isfile(template_zip_file)
            os.system('unzip -qq ' + template_zip_file + ' ' + '-d' + ' ' + self.dicom_root_folder)
            dicom_template_folder = os.path.join(self.dicom_root_folder, '1822799_CBCTTemplate', '1999-01__Studies')
            cmd = ""
            for f in os.listdir(dicom_template_folder):
                f = os.path.join(dicom_template_folder, f)
                if os.path.isfile(f):
                    continue
                cmd += "\n" + "mv" + ' "' + f + '"/*.dcm "' + os.path.join(f, '../') + '"'
                cmd += "\n" + "rm" + ' ' + "-r" + ' "' + f + '"'
            os.system(cmd)

            image_template_folder = os.path.join(self.original_folder, 'template')
            os.system('mkdir -p {}'.format(image_template_folder))
            cmd = "plastimatch convert --input {} --output-img {} --output-prefix {} --prefix-format nii.gz".format(
                dicom_template_folder,
                os.path.join(image_template_folder, 'template.nii.gz'),
                image_template_folder
            )
            os.system(cmd)
        else:
            template_zip_file = os.path.join(self.zip_root_folder, 'templates', self.patient + '.zip')
            assert os.path.isfile(template_zip_file)
            os.system('unzip -qq ' + template_zip_file + ' ' + '-d' + ' ' + self.dicom_root_folder)

            dicom_template_folder = os.path.join(self.dicom_root_folder, self.patient + 'Template', 'planCT')

            image_template_folder = os.path.join(self.original_folder, 'template')
            os.system('mkdir -p {}'.format(image_template_folder))
            cmd = "plastimatch convert --input {} --output-img {}".format(
                dicom_template_folder,
                os.path.join(image_template_folder, 'templateCT.nii.gz')
            )
            os.system(cmd)

        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="preprocessing")
    parser.add_argument('--patients', nargs='+', dest='patients', help='list of patients', type=int)
    args = parser.parse_args(['--patients',
                              '5', '8', '16', '29', '30',
                              '32', '35', '37', '38', '40',
                              '41', '43', '44', '45', '47',
                              '48', '50', '51', '52', '53',
                              '9', '10', '11', '12', '13',
                              '14', '15', '17', '18', '19',
                              '20', '21', '22', '23', '24',
                              '27', '34', '42', '49', '96'])
    args = parser.parse_args(['--patients', '5'])
    print(args)
    patients = args.patients
    if patients is None:
        print("No patient given")
        for i in range(0, 100):
            print("Processing patient 18227{:02d}".format(i))
            patient = "18227{:02d}".format(i)
            preprocesser = ProcessR21(patient)
            preprocesser.process()
    else:
        for p in patients:
            patient = "18227{:02d}".format(p)
            preprocesser = ProcessR21(patient)
            print("Processing patient 18227{:02d}".format(p))
            preprocesser.process()

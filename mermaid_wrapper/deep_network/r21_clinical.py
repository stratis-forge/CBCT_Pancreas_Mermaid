import os
import sys
import datetime
import json
import socket

from settings.setting import parse_opts
import torch
import torch.nn.functional as F
import SimpleITK as sitk

torch.backends.cudnn.benchmark = True
sys.path.append(os.path.join(os.path.dirname(__file__), '../mermaid'))
from modules.mermaid_net import MermaidNet
from torch.utils.data import DataLoader
from data_loaders.ct_cbct_test import R21RegDataset
import numpy as np


class TestR21:
    def __init__(self):
        self.settings = parse_opts()
        torch.manual_seed(self.settings.manual_seed)
        self.root_folder = os.path.dirname(os.path.realpath(__file__))

        # set configuration file:
        self.network_config = None
        self.mermaid_config = None
        self.network_folder = None
        self.network_file = None
        self.network_config_file = None
        self.mermaid_config_file = None
        self.img_list = None
        self.test_folder = None
        self.__setup__()

        # load models
        self.test_data_loader = None
        self.model = None
        self.__load_models__()
        print("Finish Loading models")
        return

    def __create_test_model__(self):
        model = MermaidNet(self.network_config['model'])
        model.cuda()
        checkpoint = torch.load(self.network_file)
        print("Best eval epoch: {}".format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def __create_test_dataloader__(self):
        model_config = self.network_config['model']
        test_dataset = R21RegDataset(self.settings)

        batch_size = 1
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                     drop_last=False, num_workers=4)
        # add info to config
        model_config['img_sz'] = [batch_size, 1, self.settings.input_D, self.settings.input_H, self.settings.input_W]
        model_config['dim'] = 3
        return test_dataloader

    def __load_models__(self):
        self.test_data_loader = self.__create_test_dataloader__()
        self.model = self.__create_test_model__()

    def __setup__(self):
        # to continue, specify the model folder and model
        self.network_folder = os.path.dirname(self.settings.saved_model)
        self.network_config_file = os.path.join(self.network_folder, 'network_config.json')
        self.mermaid_config_file = os.path.join(self.network_folder, 'mermaid_config.json')
        self.network_file = self.settings.saved_model

        self.test_folder = os.path.join(self.network_folder, "test_result_{}".format(os.path.basename(self.settings.saved_model)))
        os.system('mkdir {}'.format(self.test_folder))

        print("Loading {}".format(self.network_config_file))
        print("Loading {}".format(self.mermaid_config_file))
        with open(self.network_config_file) as f:
            self.network_config = json.load(f)
        with open(self.mermaid_config_file) as f:
            self.mermaid_config = json.load(f)
        self.network_config['model']['mermaid_config_file'] = self.mermaid_config_file

        #print("Reading {}".format(self.settings.test_list))
        with open(self.settings.test_list, 'r') as f:
            self.img_list = [line.strip() for line in f]
        return

    def test_model(self):
        iters = len(self.test_data_loader.dataset)
        print('iters:', str(iters))

        mergeFilter = sitk.MergeLabelMapFilter()
        mergeFilter.SetMethod(Method=0)
        with torch.no_grad():
            for j, images in enumerate(self.test_data_loader, 0):
                ct_image_name = os.path.join('../', self.img_list[j].split(' ')[0])
                cb_image_name = os.path.join('../', self.img_list[j].split(' ')[1])
                roi_name = os.path.join('../', self.img_list[j].split(' ')[2])
                ct_labels_name = os.path.join('../', self.img_list[j].split(' ')[3].replace('SmBowel_label', 'oar_label'))
                #if not ("OG" in ct_image_name and "OG" in cb_image_name):
                #     print("Simulated case, ignore")
                #     continue
                #patient = ct_image_name.split('18227')[1][0:2]
                #cb_case = cb_image_name.split('CBCT')[1][0:2]
                ct_image = images[0].cuda()
                cb_image = images[1].cuda()
                roi_label = images[2].cuda()
                ct_sblabel = images[3].cuda()
                ct_sdlabel = images[4].cuda()

                self.model(ct_image, cb_image, roi_label, ct_sblabel, ct_sdlabel)
                warped_moving_image = self.model.warped_moving_image
                warped_moving_sblabel = self.model.warped_moving_labels[:, [0], ...]
                warped_moving_sdlabel = self.model.warped_moving_labels[:, [1], ...]

                phi = self.model.phi - self.model.identityMap

                result_folder = os.path.join(self.test_folder,
                                             '{}__to_{}'.format(ct_image_name.split('data/')[1].replace('/', '_'),
                                                                cb_image_name.split('data')[1].replace('/', '_')))
                print(ct_image_name, cb_image_name)
                os.system('mkdir -p {}'.format(result_folder))
                orig_image_itk = sitk.ReadImage(ct_image_name)

                copied_ct_image_file = os.path.join(result_folder, 'ct_image.nii.gz')
                copied_cb_image_file = os.path.join(result_folder, 'cb_image.nii.gz')
                copied_roi_file = os.path.join(result_folder, 'roi.nii.gz')
                copied_ct_labels_file = os.path.join(result_folder, 'ct_labels.nii.gz')

                os.system('cp {} {}'.format(ct_image_name, copied_ct_image_file))
                os.system('cp {} {}'.format(cb_image_name, copied_cb_image_file))
                os.system('cp {} {}'.format(roi_name, copied_roi_file))
                os.system('cp {} {}'.format(ct_labels_name, copied_ct_labels_file))

                orig_image_arr = sitk.GetArrayFromImage(orig_image_itk)
                spacing = 1. / (np.array(orig_image_arr.shape) - 1)

                [depth, height, width] = orig_image_arr.shape
                scale = [depth * 1.0 / self.settings.input_D,
                         height * 1.0 / self.settings.input_H * 1.0,
                         width * 1.0 / self.settings.input_W * 1.0]

                orig_phi_x = F.interpolate(phi[:, [0], ...], scale_factor=scale, mode='trilinear')
                orig_phi_y = F.interpolate(phi[:, [1], ...], scale_factor=scale, mode='trilinear')
                orig_phi_z = F.interpolate(phi[:, [2], ...], scale_factor=scale, mode='trilinear')
                orig_phi_x[0, 0, ...] = orig_phi_x[0, 0, ...] / spacing[0]
                orig_phi_y[0, 0, ...] = orig_phi_y[0, 0, ...] / spacing[1]
                orig_phi_z[0, 0, ...] = orig_phi_z[0, 0, ...] / spacing[2]

                orig_warped_moving_image = F.interpolate(warped_moving_image, scale_factor=scale, mode='trilinear')
                orig_warped_moving_sblabel = F.interpolate(warped_moving_sblabel, scale_factor=scale, mode='nearest')
                orig_warped_moving_sdlabel = F.interpolate(warped_moving_sdlabel, scale_factor=scale, mode='nearest')

                orig_warped_moving_image_itk = sitk.GetImageFromArray(torch.squeeze(orig_warped_moving_image).cpu().numpy())
                orig_warped_moving_image_itk.CopyInformation(orig_image_itk)
                orig_warped_moving_image_file = os.path.join(result_folder, 'warped_moving_image.nii.gz')
                sitk.WriteImage(orig_warped_moving_image_itk, orig_warped_moving_image_file)

                orig_warped_moving_sblabel_arr = torch.squeeze(orig_warped_moving_sblabel).cpu().numpy().astype(np.uint8)
                orig_warped_moving_sblabel_itk = sitk.GetImageFromArray(orig_warped_moving_sblabel_arr)
                orig_warped_moving_sblabel_itk.CopyInformation(orig_image_itk)
                orig_warped_moving_sblabel_file = os.path.join(result_folder, 'warped_moving_sblabel.nii.gz')
                sitk.WriteImage(orig_warped_moving_sblabel_itk, orig_warped_moving_sblabel_file)

                orig_warped_moving_sdlabel_arr = torch.squeeze(orig_warped_moving_sdlabel).cpu().numpy().astype(np.uint8)
                orig_warped_moving_sdlabel_itk = sitk.GetImageFromArray(orig_warped_moving_sdlabel_arr)
                orig_warped_moving_sdlabel_itk.CopyInformation(orig_image_itk)
                orig_warped_moving_sdlabel_file = os.path.join(result_folder, 'warped_moving_sdlabel.nii.gz')
                sitk.WriteImage(orig_warped_moving_sdlabel_itk, orig_warped_moving_sdlabel_file)

                all_moving_labels_itk = mergeFilter.Execute([sitk.Cast(orig_warped_moving_sblabel_itk, sitk.sitkLabelUInt8),
                                                             sitk.Cast(orig_warped_moving_sdlabel_itk, sitk.sitkLabelUInt8)])
                all_moving_labels_file = os.path.join(result_folder, 'warped_moving_labels.nii.gz')
                sitk.WriteImage(sitk.Cast(all_moving_labels_itk, sitk.sitkUInt8), all_moving_labels_file)

                orig_phi = torch.cat((orig_phi_x, orig_phi_y, orig_phi_z), dim=1).permute([0, 2, 3, 4, 1])
                orig_phi_itk = sitk.GetImageFromArray(torch.squeeze(orig_phi).cpu().numpy(), isVector=True)
                orig_phi_itk.CopyInformation(orig_image_itk)
                orig_phi_file = os.path.join(result_folder, 'phi.nii')
                sitk.WriteImage(orig_phi_itk, orig_phi_file)



if __name__ == '__main__':
    network = TestR21()
    network.test_model()

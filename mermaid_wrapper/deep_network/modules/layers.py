import torch
import torch.nn as nn


class MaxPool(nn.Module):
    def __init__(self, kernel_size=2, dim=3, return_indieces=False):
        super(MaxPool, self).__init__()
        if dim == 1:
            max_pool = nn.MaxPool1d
        elif dim == 2:
            max_pool = nn.MaxPool2d
        elif dim == 3:
            max_pool = nn.MaxPool3d
        else:
            raise ValueError("Dimension error")
        self.max_pool = max_pool(kernel_size, stride=2, return_indices=return_indieces)
        return

    def forward(self, x):
        return self.max_pool(x)


class ConBnRelDp(nn.Module):
    # conv + batch_normalize + relu + dropout
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, activate_unit='relu', same_padding=True,
                 use_bn=False, use_dp=False, p=0.2, reverse=False, group=1, dilation=1, dim=3):
        super(ConBnRelDp, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        if dim == 1:
            conv = nn.Conv1d
            batch_norm = nn.BatchNorm1d
            drop_out = nn.Dropout
            convT = nn.ConvTranspose1d
        elif dim == 2:
            conv = nn.Conv2d
            batch_norm = nn.BatchNorm2d
            drop_out = nn.Dropout2d
            convT = nn.ConvTranspose2d
        elif dim == 3:
            conv = nn.Conv3d
            batch_norm = nn.BatchNorm3d
            drop_out = nn.Dropout3d
            convT = nn.ConvTranspose3d
        else:
            raise ValueError("Dimension can only be 1, 2 or 3.")
        if not reverse:
            self.conv = conv(in_ch, out_ch, kernel_size, stride, padding, groups=group, dilation=dilation)
        else:
            self.conv = convT(in_ch, out_ch, kernel_size, stride, padding, groups=group, dilation=dilation)

        self.batch_norm = batch_norm(out_ch) if use_bn else False
        if activate_unit == 'relu':
            self.activate_unit = nn.ReLU(inplace=True)
        elif activate_unit == 'elu':
            self.activate_unit = nn.ELU(inplace=True)
        elif activate_unit == 'leaky_relu':
            self.activate_unit = nn.LeakyReLU(inplace=True)
        elif activate_unit == 'prelu':
            self.activate_unit = nn.PReLU(init=0.01)
        elif activate_unit == 'sigmoid':
            self.activate_unit = nn.Sigmoid()
        else:
            self.activate_unit = False
        self.drop_out = drop_out(p) if use_dp else False

        return

    def forward(self, x):
        x = self.conv(x)
        if self.activate_unit is not False:
            x = self.activate_unit(x)
        if self.batch_norm is not False:
            x = self.batch_norm(x)
        if self.drop_out is not False:
            x = self.drop_out(x)
        return x


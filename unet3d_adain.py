import torch
import torch.nn as nn

class adain_code_generator_seperate(nn.Module):
    """
    ref: https://github.com/YSerin/TMI_SwitchableCycleGAN/blob/train/networks/adain.py
    """
    def __init__(self, ch):
        super(adain_code_generator_seperate, self).__init__()

        self.fc_mean = nn.Linear(128, ch)
        self.fc_var = nn.Linear(128, ch)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, input, shared_code):
        N, C, h, w, d = input.size()

        fc_mean = self.fc_mean(shared_code)
        fc_var = self.fc_var(shared_code)      
        fc_var = self.ReLU(fc_var)

        fc_mean_np = fc_mean.view(N, C, 1, 1, 1).expand(N, C, h, w, d)
        fc_var_np = fc_var.view(N, C, 1, 1, 1).expand(N, C, h, w, d)

        return fc_mean_np, fc_var_np


class adain_code_generator_shared(nn.Module):
    """
    ref: https://github.com/YSerin/TMI_SwitchableCycleGAN/blob/train/networks/adain.py
    """
    def __init__(self, voxel_size=[0.982, 0.982, 1]):
        super(adain_code_generator_shared, self).__init__()
        self.voxel_size = torch.Tensor(voxel_size)
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)

    def forward(self, batch_size):
        self.ones_vec = (self.voxel_size * torch.ones((batch_size, 3))).cuda(self.fc1.weight.device)

        x = self.fc1(self.ones_vec)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


class one_conv_adain(nn.Module):
    """
    ref: https://github.com/YSerin/TMI_SwitchableCycleGAN/blob/train/networks/adain.py
    """
    def __init__(self, in_ch, out_ch):
        super(one_conv_adain, self).__init__()
        self.conv3d = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.instanceNorm3d = nn.InstanceNorm3d(out_ch)
        self.adain = adain_code_generator_seperate(out_ch)
        self.Leakyrelu = nn.LeakyReLU(inplace=True)
        # self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x_in, shared_code, alpha=1):
        x_in = self.conv3d(x_in)
        x_in = self.instanceNorm3d(x_in)
        x_out = self.Leakyrelu(x_in)

        # mean_y, sigma_y = self.adain(x_in, shared_code)
        # x_out = sigma_y * (x_in) + mean_y

        # x_out = x_out * (alpha) + x_in * (1 - alpha)
        # x_out = self.Leakyrelu(x_out)

        return x_out

class ConvAdaIN(nn.Module):
    """ convolution => adain => LeakyReLU """
    def __init__(self, in_channels, out_channels):
        super(ConvAdaIN, self).__init__()
        self.conv3d_adain = one_conv_adain(in_channels, out_channels)

    def forward(self, x, shared_code):
        x = self.conv3d_adain(x, shared_code)
        return x

class Down(nn.Module):
    """Downscaling with 3D conv and stride 2"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=2),
            # nn.InstanceNorm3d(out_channels)
        )

    def forward(self, x):
        return self.down_conv(x)

class Up(nn.Module):
    """Upsampling with nearest neighbor then 3d conv"""
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up_nn = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.InstanceNorm3d(out_channels)
        )

    def forward(self, x):
        return self.up_nn(x)

class OutConv(nn.Module):
    """last conv, with kernel size 1"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3DAdaINCodeGen(nn.Module):
    """main network class"""
    def __init__(self):
        super(UNet3DAdaINCodeGen, self).__init__()

        self.conv_adain1 = ConvAdaIN(1, 32)
        self.conv_adain2 = ConvAdaIN(32, 64)
        self.conv_adain3 = ConvAdaIN(64, 128)
        self.conv_adain4 = ConvAdaIN(128, 256)
        self.conv_adain5 = ConvAdaIN(256, 128)
        self.conv_adain6 = ConvAdaIN(128, 64)
        self.conv_adain7 = ConvAdaIN(64, 32)
        
        self.down1 = Down(32, 32)
        self.down2 = Down(64, 64)
        self.down3 = Down(128, 128)
        
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)

        self.outc = OutConv(32, 1)

        self.adain_shared = adain_code_generator_shared()

    def forward(self, x):
        # for adain
        shared_code = self.adain_shared(x.shape[0])

        # encoder
        enc32 = self.conv_adain1(x, shared_code)
        enc32_down = self.down1(enc32)
        enc64 = self.conv_adain2(enc32_down, shared_code)
        enc64_down = self.down2(enc64)
        enc128 = self.conv_adain3(enc64_down, shared_code)
        enc128_down = self.down3(enc128)
        enc256 = self.conv_adain4(enc128_down, shared_code)
        
        # decoder
        dec128_for_cat = self.up1(enc256)
        dec128 = self.conv_adain5(torch.cat([enc128, dec128_for_cat], axis=1), shared_code)
        dec64_for_cat = self.up2(dec128)
        dec64 = self.conv_adain6(torch.cat([enc64, dec64_for_cat], axis=1), shared_code)
        dec32_for_cat = self.up3(dec64)
        dec32 = self.conv_adain7(torch.cat([enc32, dec32_for_cat], axis=1), shared_code)
        
        out = self.outc(dec32)

        return out
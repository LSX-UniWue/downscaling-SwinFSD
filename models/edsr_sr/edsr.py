## From https://github.com/sanghyun-son/EDSR-PyTorch
## PyTorch version of the paper 'Enhanced Deep Residual Networks for Single Image Super-Resolution' (CVPRW 2017)

from models.edsr_sr import common

import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F


from models.lightning_model_template import LightningModelTemplate, LightningModelTemplateSidechannel, cyclicMSELoss, cyclicPSNR
from torchmetrics.image import PeakSignalNoiseRatio

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

def make_model(args, parent=False):
    return EDSR(args)

class EDSR(LightningModelTemplate):
    def __init__(self, 
                n_resblocks: int = 16,
                n_feats: int = 64,
                scale: int = 4,
                in_chans: int = 20,
                out_chans: int = 20,
                out_channels = None,
                res_scale: float = 1,
                dataset_metrics: dict = {}, 
                channel_names: list = [],
                learning_rate: float = 1e-4,
                loss_function: str = 'mse',
                weight_decay: float = 0.0,
                lr_scheduling: bool = False,
                conv=common.default_conv):
        """
        Initializes an instance of the EDSR model.

        Args:
            n_resblocks (int): Number of residual blocks in the model. Default is 16.
            n_feats (int): Number of feature maps in the model. Default is 64.
            scale (int): Upscaling factor of the model. Default is 4.
            n_colors (int): Number of color channels. Default is 3.
            res_scale (float): Residual scaling factor. Default is 1.
            conv (function): Convolution function to use. Default is common.default_conv.

        Returns:
            None
        """
        super().__init__()

        self.channel_names = np.array(channel_names)
        self.save_hyperparameters()

        self.seperate_dataset = ('si10' and 'wdir10' not in self.channel_names) and ('u10' and 'v10' in self.channel_names)

        n_colors = in_chans
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.n_feats = n_feats
        
        if out_channels is None:
            num_out_ch = out_chans
            self.out_channels = list(range(num_out_ch))
        else:
            num_out_ch = len(out_channels)
            self.out_channels = out_channels

        if len(dataset_metrics.keys()) != 0:
            print(f"Dataset metrics provided. Using provided values.")

            self.register_buffer("variable_mean", torch.tensor(dataset_metrics["variable_mean"].reshape(1, -1, 1, 1)).to(torch.float32))
            self.register_buffer("variable_std", torch.tensor(dataset_metrics["variable_std"].reshape(1, -1, 1, 1)).to(torch.float32))
            self.register_buffer("variable_max", torch.tensor(dataset_metrics["variable_max"].reshape(1, -1, 1, 1)).to(torch.float32))
            self.register_buffer("variable_min", torch.tensor(dataset_metrics["variable_min"].reshape(1, -1, 1, 1)).to(torch.float32))
        else:
            print("No dataset metrics provided. Using default values.")
            self.variable_mean = 0
            self.variable_std = 1
            self.variable_max = None
            self.variable_min = None

        self.valid_psnr = PeakSignalNoiseRatio()
        self.test_psnr = PeakSignalNoiseRatio()

        self.learning_rate = learning_rate
        self.sync_dist = False


        self.weight_decay = weight_decay
        self.lr_scheduling = lr_scheduling

        if loss_function == 'mse':
            self.loss_function = F.mse_loss
        elif loss_function == 'cyclic_loss':
            self.loss_function = cyclicMSELoss(np.where(self.channel_names == 'wdir10'), self.out_chans)
            self.mse_function = cyclicMSELoss(np.where(self.channel_names == 'wdir10'), self.out_chans, norm_data=False)
            self.psnr_cyclic = cyclicPSNR(np.where(self.channel_names == 'wdir10'))

        else:
            raise ValueError(f"Unknown loss function: {loss_function}")



        kernel_size = 3 
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None

        # self.sub_mean = common.MeanShift(rgb_range)
        # self.add_mean = common.MeanShift(rgb_range, sign=1)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
                
class EDSR_sidechannel(LightningModelTemplateSidechannel):
    def __init__(self, 
                n_resblocks: int = 16,
                n_feats: int = 64,
                scale: int = 4,
                in_chans: int = 20,
                out_chans: int = 20,
                out_channels = None,
                num_sidechannels: int = 2,
                embed_dim_sidechannels: int = 20,
                loss_beta: float = 0.5,
                res_scale: float = 1,
                dataset_metrics: dict = {}, 
                channel_names: list = [],
                learning_rate: float = 1e-4,
                loss_function: str = 'mse',
                weight_decay: float = 0.0,
                lr_scheduling: bool = False,
                conv=common.default_conv):
        """
        Initializes an instance of the EDSR model.

        Args:
            n_resblocks (int): Number of residual blocks in the model. Default is 16.
            n_feats (int): Number of feature maps in the model. Default is 64.
            scale (int): Upscaling factor of the model. Default is 4.
            rgb_range (int): Range of RGB values. Default is 255.
            n_colors (int): Number of color channels. Default is 3.
            res_scale (float): Residual scaling factor. Default is 1.
            conv (function): Convolution function to use. Default is common.default_conv.

        Returns:
            None
        """
        super().__init__()

        self.channel_names = np.array(channel_names)
        self.save_hyperparameters()

        self.seperate_dataset = ('si10' and 'wdir10' not in self.channel_names) and ('u10' and 'v10' in self.channel_names)

        n_colors = in_chans
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_sidechannels = num_sidechannels

        self.n_feats = n_feats
        self.embed_dim_sidechannels = embed_dim_sidechannels
        self.loss_beta = loss_beta

        if out_channels is None:
            num_out_ch = out_chans
            self.out_channels = list(range(num_out_ch))
        else:
            num_out_ch = len(out_channels)
            self.out_channels = out_channels

        if len(dataset_metrics.keys()) != 0:
            print(f"Dataset metrics provided. Using provided values.")

            self.register_buffer("variable_mean", torch.tensor(dataset_metrics["variable_mean"].reshape(1, -1, 1, 1)).to(torch.float32))
            self.register_buffer("variable_std", torch.tensor(dataset_metrics["variable_std"].reshape(1, -1, 1, 1)).to(torch.float32))
            self.register_buffer("variable_max", torch.tensor(dataset_metrics["variable_max"].reshape(1, -1, 1, 1)).to(torch.float32))
            self.register_buffer("variable_min", torch.tensor(dataset_metrics["variable_min"].reshape(1, -1, 1, 1)).to(torch.float32))
        else:
            print("No dataset metrics provided. Using default values.")
            self.variable_mean = 0
            self.variable_std = 1
            self.variable_max = None
            self.variable_min = None

        self.valid_psnr = PeakSignalNoiseRatio()
        self.test_psnr = PeakSignalNoiseRatio()

        self.learning_rate = learning_rate
        self.sync_dist = False

        self.weight_decay = weight_decay
        self.lr_scheduling = lr_scheduling

        if loss_function == 'mse':
            self.loss_function = F.mse_loss
        elif loss_function == 'cyclic_loss':
            self.loss_function = cyclicMSELoss(np.where(self.channel_names == 'wdir10'), self.out_chans)
            self.mse_function = cyclicMSELoss(np.where(self.channel_names == 'wdir10'), self.out_chans, norm_data=False)
            self.psnr_cyclic = cyclicPSNR(np.where(self.channel_names == 'wdir10'))

        else:
            raise ValueError(f"Unknown loss function: {loss_function}")


        kernel_size = 3 
        act = nn.ReLU(True)
       
        # Sidechannel modules
        self.sidechannel_conv_first = nn.Conv2d(num_sidechannels, embed_dim_sidechannels, 3, 1, 1)

        downsample_layer = []
        for _ in range(scale//2):
            layer = torch.nn.Conv2d(embed_dim_sidechannels, embed_dim_sidechannels, 3, 2, 1)
            downsample_layer.append(layer)

        self.downsample_sidechannel = torch.nn.Sequential(*downsample_layer)

        m_tail_sidechannel = [
            common.Upsampler(conv, scale, self.embed_dim_sidechannels, act=False),
            conv(self.embed_dim_sidechannels, self.num_sidechannels, kernel_size)
        ]

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, self.n_feats + self.embed_dim_sidechannels, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(self.n_feats + self.embed_dim_sidechannels, self.n_feats + self.embed_dim_sidechannels, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        self.tail_sidechannel = nn.Sequential(*m_tail_sidechannel)

    def forward(self, x):
        # x = self.sub_mean(x)
        x, x_sidechannel = x

        x = self.head(x)
        x_sidechannel = self.sidechannel_conv_first(x_sidechannel)
        x_sidechannel = self.downsample_sidechannel(x_sidechannel)


        x = torch.cat([x, x_sidechannel], dim=1)

        res = self.body(x)
        res += x

        res_main = res[:, :self.n_feats, :, :]
        res_sidechannel = res[:, self.n_feats:, :, :]

        x = self.tail(res_main)
        x_sidechannel = self.tail_sidechannel(res_sidechannel)
        # x = self.add_mean(x)

        return x, x_sidechannel

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))



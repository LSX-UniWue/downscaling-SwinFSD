import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L
import numpy as np

from torchmetrics.image import PeakSignalNoiseRatio

from models.lightning_model_template import LightningModelTemplate, cyclicMSELoss, cyclicPSNR


class InterpolationModel(LightningModelTemplate):
    def __init__(self, upscale: int = 4, dataset_metrics: dict = {}, channel_names: list = [], do_nothing: bool = False, cerra_test_step: bool = False, out_chans=20, out_channels=None, loss_function: str = 'mse' ) -> None:
        super().__init__()

        self.channel_names = channel_names
        self.do_nothing = do_nothing
        self.cerra_test_step = cerra_test_step
        self.out_chans = out_chans
    
        if self.cerra_test_step:
            print(f"Parameter cerra_test_step is enabled. The model use the CERRA target data for testing and will not perform any computation.\
                  The model will return the CERRA target data as output. (Only useful for callback functions (e.g., DWD-Callback))")

        if out_channels is None:
            num_out_ch = out_chans
            self.out_channels = list(range(num_out_ch))
        else:
            num_out_ch = len(out_channels)
            self.out_channels = out_channels

        if self.do_nothing:
            print("Do nothing mode is enabled. The model will not perform any computation.")

        self.seperate_dataset = ('si10' and 'wdir10' not in self.channel_names) and ('u10' and 'v10' in self.channel_names)

        self.save_hyperparameters()

        if len(dataset_metrics.keys()) != 0:
            print(f"Dataset metrics provided. Using provided values.")

            self.register_buffer("variable_mean", torch.tensor(dataset_metrics["variable_mean"].reshape(1, -1, 1, 1)))
            self.register_buffer("variable_std", torch.tensor(dataset_metrics["variable_std"].reshape(1, -1, 1, 1)))
            self.register_buffer("variable_max", torch.tensor(dataset_metrics["variable_max"].reshape(1, -1, 1, 1)))
            self.register_buffer("variable_min", torch.tensor(dataset_metrics["variable_min"].reshape(1, -1, 1, 1)))
        else:
            print("No dataset metrics provided. Using default values.")
            self.variable_mean = 0
            self.variable_std = 1
            self.variable_max = None
            self.variable_min = None

        self.valid_psnr = PeakSignalNoiseRatio()
        self.test_psnr = PeakSignalNoiseRatio()

        self.downscaling_factor = upscale
        self.sync_dist = False
        self.channel_names = np.array(channel_names)

        if loss_function == 'mse':
            self.loss_function = F.mse_loss
        elif loss_function == 'cyclic_loss':
            self.loss_function = cyclicMSELoss(np.where(self.channel_names == 'wdir10'), self.out_chans)
            self.mse_function = cyclicMSELoss(np.where(self.channel_names == 'wdir10'), self.out_chans, norm_data=False)
            self.psnr_cyclic = cyclicPSNR(np.where(self.channel_names == 'wdir10'))
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")

    def forward(self, x):
        if self.do_nothing:
            return x
        else:
            return F.interpolate(x, scale_factor=self.downscaling_factor, mode='bicubic', align_corners=False)
        
    
    def test_step(self, batch, batch_idx):
        if not self.cerra_test_step:
            return super(InterpolationModel, self).test_step(batch, batch_idx)
        
        else:
            x, y = batch[:2]
            y_s = y * self.variable_std[:, self.out_channels] + self.variable_mean[:, self.out_channels]
            return y_s
            



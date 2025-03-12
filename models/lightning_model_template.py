import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F

import lightning as L
from torchmetrics.image import PeakSignalNoiseRatio
from lightning import LightningModule


class LightningModelTemplate(LightningModule):
    def __init__(self) -> None:
        super().__init__()

    def training_step(self, batch, batch_idx):
        x, y = batch[0:2]
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0:2]
        y_hat = self(x)
        val_loss = self.loss_function(y_hat, y)

        y_hat_s = y_hat * self.variable_std[:, self.out_channels] + self.variable_mean[:, self.out_channels]
        y_s = y * self.variable_std[:, self.out_channels] + self.variable_mean[:, self.out_channels]

        val_mse = self.loss_function(y_hat_s, y_s)

        channelwise_metrics = self.perform_channelwise_evaluation(y_hat_s, y_s, y_hat, y, self.channel_names[self.out_channels])

        self.log_dict(channelwise_metrics, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        self.log_dict({"val_loss": val_loss, "val_psnr": self.valid_psnr(y_hat, y), "val_mse": val_mse}, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch[:2]
        y_hat = self(x)
        test_loss = self.loss_function(y_hat, y)
        
        y_hat_s = y_hat * self.variable_std[:, self.out_channels] + self.variable_mean[:, self.out_channels]
        y_s = y * self.variable_std[:, self.out_channels] + self.variable_mean[:, self.out_channels]

        if self.hparams['loss_function'] == 'cyclic_loss':
            test_mse = self.mse_function(y_hat_s, y_s)
            self.log_dict({"test_psnr_cyclic": self.psnr_cyclic(y_hat, y)}, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        else:
            test_mse = self.loss_function(y_hat_s, y_s)


        channelwise_metrics = self.perform_channelwise_evaluation(y_hat_s, y_s, y_hat, y, self.channel_names[self.out_channels])

        if self.seperate_dataset:
            si10 = torch.sqrt(y_s[:, 0]**2 + y_s[:, 1]**2)
            wdir10 = torch.arctan2(y_s[:, 0],y_s[:, 1])
            wdir10 = torch.remainder(180.0 + torch.rad2deg(wdir10), 360.0)

            si10_hat = torch.sqrt(y_hat_s[:, 0]**2 + y_hat_s[:, 1]**2)
            wdir10_hat = torch.arctan2(y_hat_s[:, 0],y_hat_s[:, 1])
            wdir10_hat = torch.remainder(180.0 + torch.rad2deg(wdir10_hat), 360.0)

            si10 = (si10 - 5.303656134765812) / 3.695860737416358
            wdir10 = (wdir10 - 183.88709676666727) / 107.8809532565079

            si10_hat = (si10_hat - 5.303656134765812) / 3.695860737416358
            wdir10_hat = (wdir10_hat - 183.88709676666727) / 107.8809532565079

            channelwise_metrics['mse_si10'] = F.mse_loss(si10_hat, si10).mean()
            channelwise_metrics['mse_wdir10'] = (torch.min(torch.abs(wdir10_hat-wdir10), (360 / 107.8809532565079) - torch.abs(wdir10_hat-wdir10)) ** 2).mean()



        self.log_dict(channelwise_metrics, on_step=False, on_epoch=True, sync_dist=self.sync_dist)

        self.log_dict({"test_loss": test_loss, "test_psnr": self.test_psnr(y_hat, y), "test_mse": test_mse}, on_step=False, on_epoch=True, sync_dist=self.sync_dist)

        return y_hat_s

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
    

    def perform_channelwise_evaluation(self, y_hat_s, y_s, y_hat, y, channel_names = [], loss_function = None):
        if loss_function is None:
            mse_channelwise = self.loss_function(y_hat, y, reduction='none').mean(dim=(0, 2, 3))
        else:
            mse_channelwise = loss_function(y_hat, y, reduction='none').mean(dim=(0, 2, 3))
        psnr_channelwise = torch.empty_like(mse_channelwise)

        for i in range(y_s.shape[1]):
            psnr_channelwise[i] = (self.valid_psnr(y_hat_s[:, i, :, :], y_s[:, i, :, :]))

        results = {}
        if len(channel_names) != 0:
            for i in range(len(channel_names)):
                results['mse_' + channel_names[i]] = mse_channelwise[i]
                results['psnr_' + channel_names[i]] = psnr_channelwise[i]            

        return results


class LightningModelTemplateSidechannel(LightningModule):
    def __init__(self) -> None:
        super().__init__()

    def training_step(self, batch, batch_idx):
        x, y = batch[:2]
        x, x_sidechannel = x
        y, y_sidechannel = y

        y_hat, y_hat_sidechannel = self((x, x_sidechannel))
        loss = self.loss_function(y_hat, y)
        loss_sidechannel = F.mse_loss(y_hat_sidechannel, y_sidechannel)

        
        combined_loss = loss + self.loss_beta * loss_sidechannel
        
        self.log_dict({'train_loss': loss, 'train_loss_sidechannel': loss_sidechannel, 'train_combined_loss': combined_loss})
        return combined_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]
        x, x_sidechannel = x
        y, y_sidechannel = y

        y_hat, y_hat_sidechannel = self((x, y_sidechannel))
        val_loss = self.loss_function(y_hat, y)
        

        y_hat_s = y_hat * self.variable_std[:, self.out_channels] + self.variable_mean[:, self.out_channels]
        y_s = y * self.variable_std[:, self.out_channels] + self.variable_mean[:, self.out_channels]

        val_mse = self.loss_function(y_hat_s, y_s)

        channelwise_metrics = self.perform_channelwise_evaluation(y_hat_s, y_s, y_hat, y, self.channel_names[self.out_channels])

        self.log_dict(channelwise_metrics, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        self.log_dict({"val_loss": val_loss, "val_psnr": self.valid_psnr(y_hat, y), "val_mse": val_mse}, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch[:2]
        x, x_sidechannel = x
        y, y_sidechannel = y

        y_hat, y_hat_sidechannel = self((x, y_sidechannel))
        test_loss = self.loss_function(y_hat, y)
        
        y_hat_s = y_hat * self.variable_std[:, self.out_channels] + self.variable_mean[:, self.out_channels]
        y_s = y * self.variable_std[:, self.out_channels] + self.variable_mean[:, self.out_channels]

        if self.hparams['loss_function'] == 'cyclic_loss':
            test_mse = self.mse_function(y_hat_s, y_s)
            self.log_dict({"test_psnr_cyclic": self.psnr_cyclic(y_hat, y)}, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        else:
            test_mse = self.loss_function(y_hat_s, y_s)

        channelwise_metrics = self.perform_channelwise_evaluation(y_hat_s, y_s, y_hat, y, self.channel_names[self.out_channels])

        if self.seperate_dataset:
            si10 = torch.sqrt(y_s[:, 0]**2 + y_s[:, 1]**2)
            wdir10 = torch.arctan2(y_s[:, 0],y_s[:, 1])
            wdir10 = torch.remainder(180.0 + torch.rad2deg(wdir10), 360.0)

            si10_hat = torch.sqrt(y_hat_s[:, 0]**2 + y_hat_s[:, 1]**2)
            wdir10_hat = torch.arctan2(y_hat_s[:, 0],y_hat_s[:, 1])
            wdir10_hat = torch.remainder(180.0 + torch.rad2deg(wdir10_hat), 360.0)

            si10 = (si10 - 5.303656134765812) / 3.695860737416358
            wdir10 = (wdir10 - 183.88709676666727) / 107.8809532565079

            si10_hat = (si10_hat - 5.303656134765812) / 3.695860737416358
            wdir10_hat = (wdir10_hat - 183.88709676666727) / 107.8809532565079

            channelwise_metrics['mse_si10'] = F.mse_loss(si10_hat, si10).mean()
            channelwise_metrics['mse_wdir10'] = (torch.min(torch.abs(wdir10_hat-wdir10), (360 / 107.8809532565079) - torch.abs(wdir10_hat-wdir10)) ** 2).mean()


        self.log_dict(channelwise_metrics, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        self.log_dict({"test_loss": test_loss, "test_psnr": self.test_psnr(y_hat, y), "test_mse": test_mse}, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
            
        return y_hat_s
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.lr_scheduling:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=7)
            return [optimizer], [scheduler]
        
        return [optimizer]
    

    def perform_channelwise_evaluation(self, y_hat_s, y_s, y_hat, y, channel_names = [], loss_function = None):
        if loss_function is None:
            mse_channelwise = self.loss_function(y_hat, y, reduction='none').mean(dim=(0, 2, 3))
        else:
            mse_channelwise = loss_function(y_hat, y, reduction='none').mean(dim=(0, 2, 3))
        psnr_channelwise = torch.empty_like(mse_channelwise)

        for i in range(y_s.shape[1]):
            psnr_channelwise[i] = (self.valid_psnr(y_hat_s[:, i, :, :], y_s[:, i, :, :]))

        results = {}
        if len(channel_names) != 0:
            for i in range(len(channel_names)):
                results['mse_' + channel_names[i]] = mse_channelwise[i]
                results['psnr_' + channel_names[i]] = psnr_channelwise[i]            

        return results
    

def get_outnorm(x:torch.Tensor, out_norm:str='') -> torch.Tensor:
    """ Common function to get a loss normalization value. Can
        normalize by either the batch size ('b'), the number of
        channels ('c'), the image size ('i') or combinations
        ('bi', 'bci', etc)
    """
    # b, c, h, w = x.size()
    img_shape = x.shape

    if not out_norm:
        return 1

    norm = 1
    if 'b' in out_norm:
        # normalize by batch size
        # norm /= b
        norm /= img_shape[0]
    if 'c' in out_norm:
        # normalize by the number of channels
        # norm /= c
        norm /= img_shape[-3]
    if 'i' in out_norm:
        # normalize by image/map size
        # norm /= h*w
        norm /= img_shape[-1]*img_shape[-2]

    return norm


class cyclicMSELoss(nn.Module):
    def __init__(self, cyclic_indices: list, output_length: int, reduction='mean', norm_data: bool = True):
        super(cyclicMSELoss, self).__init__()

        self.cyclic_indices = cyclic_indices
        self.cyclic_map = torch.zeros(output_length, dtype=torch.bool)
        self.cyclic_map[cyclic_indices] = True

        if norm_data:
            # self.converted_degrees = (360 - 183.88709676666727) / 107.8809532565079
            self.converted_degrees = 360 / 107.8809532565079
        else:
            self.converted_degrees = 360


        # self.cyclic_factor = len(cyclic_indices)
        # self.normal_factor = output_length - self.cyclic_factor

    def forward(self, predictions, targets, reduction: str = 'mean'):


        error = (predictions - targets)
        error[:, self.cyclic_map] = torch.min(torch.abs(error[:, self.cyclic_map]), self.converted_degrees - torch.abs(error[:, self.cyclic_map]))

        if reduction == 'mean':
            return torch.mean(error**2)
        elif reduction == 'none':
            return error**2
        else:
            raise ValueError("Invalid reduction type for cyclicMSELoss")
        
class cyclicPSNR(nn.Module):
    
    def __init__(self, cyclic_indices: list, norm_data: bool = True):
        super(cyclicPSNR, self).__init__()

        self.psnr_func = PeakSignalNoiseRatio()
        self.cyclic_indices = cyclic_indices
        
        if norm_data:
            # self.converted_degrees = (360 - 183.88709676666727) / 107.8809532565079
            self.converted_degrees = 360 / 107.8809532565079
        else:
            self.converted_degrees = 360


    def forward(self, predictions, target, reduction: str = 'mean'):

        modified_pred = predictions.clone()

        for i in self.cyclic_indices:
            error = torch.min(torch.abs(modified_pred[:, i] - target[:, i]), self.converted_degrees - torch.abs(modified_pred[:, i] - target[:, i]))
            modified_pred[:, i] = target[:, i] - error

        return self.psnr_func(modified_pred, target)






class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6, out_norm:str='bci'):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.out_norm = out_norm

    def forward(self, x, y):
        norm = get_outnorm(x, self.out_norm)
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps**2))
        return loss*norm
  
def grad_loss(v, v_gt):
    # Gradient loss
    loss = torch.mean(torch.abs(v - v_gt), dim=[1,2,3])
    jy = v[..., 1:, :, :] - v[..., :-1, :, :]
    jx = v[..., :, 1:, :] - v[..., :, :-1, :]
    jy_ = v_gt[..., 1:, :, :] - v_gt[..., :-1, :, :]
    jx_ = v_gt[..., :, 1:, :] - v_gt[..., :, :-1, :]
    loss += torch.mean(torch.abs(jy - jy_), dim=[1,2,3])
    loss += torch.mean(torch.abs(jx - jx_), dim=[1,2,3])
    
    return loss.mean()
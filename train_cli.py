import torch
import wandb
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


torch.set_float32_matmul_precision('high')


def my_main():
    cli = MyLightningCLI(save_config_callback=None)

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(EarlyStopping, "default_early_stopping")
        parser.add_lightning_class_args(ModelCheckpoint, "default_checkpoint")
        parser.set_defaults({"default_checkpoint.monitor": "val_loss", "default_checkpoint.save_top_k": 1, "default_checkpoint.mode": "min", "default_checkpoint.dirpath": "output/models"})
        parser.set_defaults({"default_early_stopping.monitor": "val_loss", "default_early_stopping.patience": 3, "default_early_stopping.mode": "min"})
        
        # Linking arguments between data and model
        parser.link_arguments("data.init_args.downscaling_factor", "model.init_args.downscaling_factor")
        parser.link_arguments("data.init_args.downscaling_factor", "model.init_args.upscale")
        parser.link_arguments("data.init_args.dataset_metrics", "model.init_args.dataset_metrics", apply_on="instantiate")
        parser.link_arguments("data.init_args.channel_names", "model.init_args.channel_names")
    

if __name__ == '__main__':
    print("Starting PyTorch Lightning CLI...")
    my_main()


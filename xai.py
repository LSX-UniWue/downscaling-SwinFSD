import sys
sys.path.append('..')

import numpy as np
from tqdm import tqdm

import torch
import plotly.express as px
import captum.attr as attr
import itertools

import wandb
import random


#Models
from models.swin_fir.swin_fir import SwinFIR_sidechannel

#Data
from dataset_utils.LightningDataModules import Era2CerraDataModule


def single_output_forward(channel, out_y, out_x):
    # Forward function that only returns the output for a single pixel (needed for the attribution)
    def forward(x_main, x_side):
        yhat = model((x_main, x_side))[0][:, channel, out_y, out_x]
        return yhat.unsqueeze(0)

    return forward


##### Configurations #####


# Model Path
model_path = "artifacts/model-wj8yuabe:v8/model.ckpt"
# model_path = "artifacts/model-gq9tyr49:v5/model.ckpt" # no sp
# model_path = "artifacts/model-ek0doq54:v6/model.ckpt" # no sp no wind

model_class = SwinFIR_sidechannel
constant_channels = True

data_path = "/anvme/workspace/b214cb13-ecodata/downscaling/"
num_workers = 32
crop_size = 256
batch_size = 1

num_batches = 300

xai_channel = 2
in_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# in_channels = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# in_channels = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

area = 'germany'
test_years = (2020, 2020)

geo_eval = False
deterministic_eval = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run = wandb.init(project='masterthesis', entity = "shiemn", group="era2cerra", job_type="xai", name="xai")

print(f"Downloading model checkpoint from: masterthesis/{model_path.split('/')[1]}")
artifact = run.use_artifact(f"masterthesis/{model_path.split('/')[1]}")
artifact.download(root=f"artifacts/{model_path.split('/')[1]}")


model = model_class.load_from_checkpoint(model_path)
model = model.to(device)
model.eval()

channel_names = np.array(model.hparams.channel_names)
out_channel_names = channel_names[model.hparams.out_channels]
in_channel_names = channel_names[in_channels]
channel_names = channel_names.squeeze()
print(f"Out channels: {out_channel_names}, In channels: {in_channel_names}")


wandb.config['data_path'] = data_path
wandb.config['model_path'] = model_path
wandb.config['model'] = model_class.__name__
wandb.config['constant_channels'] = constant_channels
wandb.config['crop_size'] = crop_size
wandb.config['batch_size'] = batch_size
wandb.config['test_years'] = test_years
wandb.config['channel_names'] = channel_names.tolist()
wandb.config['in_channels'] = in_channels
wandb.config['area'] = area
wandb.config['xai_channel'] = xai_channel #out_channel_names[xai_channel]

separate_dataset = True


# Load the data
data_module = Era2CerraDataModule(data_path, batch_size=batch_size, crop_size=crop_size, cropping=area, test_years=test_years, return_era_original=True, return_offset=True, downscaling_factor=model.hparams.upscale, constant_channels=constant_channels, num_workers=num_workers, out_channels=model.hparams.out_channels, in_channels=in_channels, use_separate_dataset=separate_dataset)
data_module.setup('test')

dataloader = data_module.test_dataloader()

attributions_main = np.zeros((len(in_channels), crop_size//model.hparams.upscale, crop_size//model.hparams.upscale))
attributions_side = np.zeros((2, crop_size, crop_size))

attributions_side_sum = np.zeros((2, crop_size, crop_size))


if geo_eval:
    print("Geo evaluation")
    num_batches = 1
    positions = list(itertools.product(range(0, crop_size, 4), range(0, crop_size, 4)))
elif deterministic_eval:
    print("Deterministic evaluation")
    positions = list(itertools.product(range(32, crop_size - 32, 12), range(32, crop_size-32, 12)))
else:
    positions = random.sample(list(itertools.product(range(32, crop_size - 32, 1), range(32, crop_size-32, 1))), 200)
    
    #positions = random.sample(list(itertools.product(range(32, crop_size - 32, 1), range(32, crop_size-32, 1))), 50)
    #positions = [(77, 197), (53, 135), (119, 44), (72, 79), (201, 52), (185, 113), (193, 69), (161, 142), (103, 159), (64, 39), (198, 132), (41, 161), (165, 42), (179, 169), (32, 170), (184, 36), (122, 206), (110, 48), (66, 203), (140, 99), (42, 116), (39, 150), (40, 163), (216, 186), (35, 59), (162, 54), (105, 211), (176, 47), (41, 207), (212, 50), (107, 160), (181, 121), (201, 77), (220, 168), (111, 139), (149, 223), (110, 186), (106, 162), (188, 200), (130, 207), (39, 96), (174, 42), (221, 211), (66, 57), (95, 119), (133, 64), (73, 82), (145, 139), (202, 212), (205, 89)]
    print(f"Random evaluation: {positions}")



exception_count = 0
count = 0
batch_count = 0
# Get the first batch
for batch in dataloader:
    batch_count += 1
    print(f"Starting with batch {batch_count}")

    for x, y in tqdm(positions):
    #for x, y in list(itertools.product(range(0, crop_size, 16), range(0, crop_size, 16))):
        count += 1
        position = (xai_channel, x, y)
        ig = attr.IntegratedGradients(single_output_forward(*position))
        
        data = batch
        input = data[0]

        input_main = input[0].to(device)
        input_side = input[1].to(device)

        target_data = data[1][0].to(device)
        target = target_data[:, *position]

        try:

            attributions, delta = ig.attribute((input_main, input_side), target=target.long(), return_convergence_delta=True, n_steps=25)
            
            

            attributions_main = (attributions_main * (count - 1) + abs(attributions[0]).detach().cpu().numpy().mean(axis=(0)).squeeze()) / count
            attributions_side = (attributions_side * (count - 1) + abs(attributions[1]).detach().cpu().numpy().mean(axis=(0)).squeeze()) / count

            attributions_side_sum[:, x, y] += abs(attributions[1]).detach().cpu().numpy().mean(axis=(0,2,3)).squeeze()

        
        except Exception as e:
            exception_count += 1
            continue
        
    if batch_count == num_batches:
        break


print(f"An exception occured in {exception_count} of {count} iterations")

# Save the attributions
np.save("attribution_main", attributions_main)
np.save("attribution_side", attributions_side)
np.save("attribution_side_sum", attributions_side_sum)
wandb.save("attribution_main.npy")
wandb.save("attribution_side.npy")
wandb.save("attribution_side_sum.npy")

fig = px.imshow(attributions_main[3])
fig.write_image('attribution_map.png')

channel_relevance = np.abs(attributions_main).sum(axis=(1, 2)).squeeze()
data = {'names': in_channel_names, 'relevance': channel_relevance}
wandb.log(data)
fig = px.bar(data, y='relevance', x='names')
fig.write_image('channel_attribution.png')

channel_relevance_side = np.abs(attributions_side).sum(axis=(1, 2)).squeeze()
data = {'names_side': ['land-sea mask', 'orography'], 'relevance_side': channel_relevance_side}
wandb.log(data)
fig = px.bar(data, y='relevance_side', x='names_side')
fig.write_image('channel_attribution_side.png')

wandb.log({"attribution_map": wandb.Image("attribution_map.png")})
wandb.log({"channel_attribution": wandb.Image("channel_attribution.png")})
wandb.log({"channel_attribution_side": wandb.Image("channel_attribution_side.png")})
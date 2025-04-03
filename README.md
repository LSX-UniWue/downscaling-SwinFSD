

# Statistical Downscaling of Multiple Atmospheric Variables over Europe using Super-Resolution Transformer

This repository contains the code for the research paper titled "Statistical Downscaling of Multiple Atmospheric Variables over Europe using Super-Resolution Transformer", which is currently under review. This study introduces a Super-Resolution Transformer for downscaling of multiple atmospheric variables.

To obtain example preprocessed ERA5 and CERRA data and model weights, please head to our [Zenodo-Project](https://doi.org/10.5281/zenodo.15012510).

## Demo

![Demo Animation](notebooks/animation.gif)

## Abstract
High-resolution climate projections are crucial for understanding regional climate impacts, but current global climate models are limited by their coarse resolution. Downscaling methods address this limitation by bridging the gap between global-scale and regional-scale data. Recently, empirical downscaling approaches utilizing deep learning have shown promising results.

This study introduces SwinFSD, an empirical downscaling model adapted from the state-of-the-art super-resolution model SwinFIR, designed to downscale 20 atmospheric variables from the ERA5 reanalysis dataset to the high-resolution CERRA dataset. Additionally, we present DECADL, a downscaling dataset that enables training and evaluation of SwinFSD across Europe using both gridded CERRA data and station-based observational data.
SwinFSD achieves good results, demonstrating strong performance in evaluations against CERRA data and station-based observations across Europe.

Furthermore, we evaluate the model’s generalization capabilities on regions outside the training domain, assessing its ability to partially generalize to new regions.

## Getting Started

To get started with the project, clone the repository to your local machine.
Then install the required dependencies using, e.g., the provided [dockerfile](slurm_jobs/Dockerfile).

For a quick start, download the example preprocessed data from our [Zenodo-Project](https://doi.org/10.5281/zenodo.15012510) and extract it to the `data` folder.

Alternatively, you can download the raw data and preprocess it using the scripts in the `data_processing` folder. For this you will need a free account at the [CDS API](https://cds.climate.copernicus.eu/#!/home).

After that, configure the training and evaluation settings in the `configs` folder and run the training script using the PyTorch Lightning CLI.
For automatic logging, configer the `wandb` logger by setting the configuration to match your own account in the configuration file.

To run the training, use the following command:

```bash
python train_cli.py fit --config configs/example.yaml
```

or run the evaluation as follows:

```bash
python train_cli.py test --config configs/example.yaml
```

## Folder Structure

- **configs**: Contains PyTorch Lightning CLI configuration files for running experiments and training models.
- **data**: Used to store raw and processed datasets for the project. This includes climate data from various sources. Place the downloaded data from zenodo in the respective subfolders.
  - **data_processing**: Contains the scripts to download and preprocess the relevant datasets.
- **dataset_utils**: Contains PyTorch Datasets for loading and processing datasets.
- **slurm_jobs**: Contains utility scripts running experiments on a slurm cluster.
- **models**: Contains the implementation of different machine learning models used for downscaling. Each subfolder/file within `models` represents a specific model or approach.
  - **lightning_model_template**: Contains template classes for PyTorch Lightning models adapted to our use case.
  - **edsr_sr**: Contains the implementation of the Enhanced Deep Super-Resolution (EDSR) model.
  - **swin_fir**: Contains the implementation of the SwinFIR model.
  - **interpolation_models**: Contains the implementation of simple interpolation models used as baselines.
- **notebooks**: Contains a example Jupyter notebooks used to create the demo animation.
- **utils**: Contains custom callbacks for running customized evaluations.
- **train_cli.py**: The main script for training models using PyTorch Lightning CLI.
- **xai.py**: The main script for running XAI experiments.

## Hyperparameter-Search

To run a hyperparameter search, you can use tools like W&B Sweeps. Below is the search space used for our HP-Search:

| Parameter                | Search Space                     | Selected Value |
|--------------------------|-----------------------------------|----------------|
| Model-Embedding-Dimension | [60, 90, 120, 180, **240**]     | **240**        |
| Side-Embedding-Dimension  | [10, **20**, 30]                | **20**         |
| Model-Depth              | [4, **5**, 6, 7, 8]             | **5**          |
| Model-Layers             | [4, **5**, 6, 7, 8]             | **5**          |
| Model-MLP-Ratio          | [1, **2**, 3, 4]                | **2**          |
| Model-Window-Size        | [4, **8**, 16]                  | **8**          |
| Model-Heads              | [3, **6**]                      | **6**          |
| Loss-Beta                | (0.1-0.9)                       | **0.6**        |
| Learning-Rate            | (1e-2-1e-4)                     | **1.5e-4**     |
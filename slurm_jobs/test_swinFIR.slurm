#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --time=7:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --output=output/job-%j.out


# Load modules

module load cuda/12.1.1
module load python/3.9-anaconda
module load openmpi/4.1.6-gcc11.2.0-cuda
module load hdf5/1.10.7-gcc11.2.0-openmpi-cuda

# Set up environment
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

conda activate downscaling

export WANDB__SERVICE_WAIT=120


config="config/paper/swinFIR_4x.yaml"
artifact="model-wj8yuabe:v8"

wandb artifact get "masterthesis/${artifact}" --root "artifacts/${artifact}"
python -u train_cli.py test --config $config --ckpt_path "artifacts/${artifact}/model.ckpt"  \
        --data.init_args.data_path "/anvme/workspace/b214cb13-ecodata/downscaling/" \
        --data.init_args.return_era_original true \
        --data.init_args.batch_size 16 \
        --trainer.logger.init_args.job_type "test" \
        --model.init_args.loss_function "cyclic_loss"


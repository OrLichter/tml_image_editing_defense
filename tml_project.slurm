#! /bin/sh
#SBATCH --job-name=test
#SBATCH --output=/home/dcor/orlichter/slurm/test.out
#SBATCH --error=/home/dcor/orlichter/slurm/test.err
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-a100-killable
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=1
_
cd /home/dcor/orlichter/TML_project
accelerate launch train_noise.py
--pretrained_model_name_or_path latent-consistency/lcm-sdxl
--pretrained_vae_model_name_or_path adebyollin/sdxl-vae-fp16-fix
--train_data_dir /home/dcor/orlichter/TML_project/data/single_image_dataset
--output_dir experiments/test
--num_train_epochs 100
--report_to wandb
--mixed_precision fp16
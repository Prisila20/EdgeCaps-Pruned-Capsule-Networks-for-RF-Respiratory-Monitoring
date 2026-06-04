#!/bin/bash
#SBATCH --job-name=train_sweep
#SBATCH --array=0-107
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=/home/postgrads/2803323I/edge_exp/EdgeCaps-Pruned-Capsule-Networks-for-RF-Respiratory-Monitoring/logs/output_train_%A_%a.log

# Activate env
module load python
source ~/torch-env/bin/activate
export PYTHONPATH=$PWD

# Hyperparameters
p_vals=(0.2 0.3 0.4)
t_vals=(3 5 7)
a_vals=(0.1 0.3 0.5 0.7)
l_vals=(1e-3 1e-4 1e-5)

i=$SLURM_ARRAY_TASK_ID

p=${p_vals[$(( i / 36 % 3 ))]}
t=${t_vals[$(( i / 12 % 3 ))]}
a=${a_vals[$(( i / 3  % 4 ))]}
l=${l_vals[$(( i % 3 ))]}

echo "Running job $i"
echo "Params: p=$p t=$t a=$a lr=$l"

# python run_experiment.py \
python -m scripts.run_experiment \
    --prune_ratio $p \
    --temperature $t \
    --alpha $a \
    --lr $l \
    --batch_size 16 \
    --data_path data/EdgeCaps_datasets
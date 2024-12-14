#!/usr/bin/env bash
#SBATCH --mem=30000
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --output=slurm_out/slurm-%j.out

set -e

probe="mlamaf"
langs=('zh' 'tr')  # language array to use
sizes=(1000 5000 10000)  # dataset sizes
out_dir="./evl_mbert_focus_continue_pretraining/"  # dir to save output
args="${@:1}"

mkdir -p ${out_dir}

for lang in "${langs[@]}"; do
  for size in "${sizes[@]}"; do
    # Define the model dynamically based on the language and size
    model="mbert_focus_${lang}_${size}"
    # Keep the probe unchanged
    
    echo "========== Language: $lang, Size: $size, Model: $model, Probe: $probe, Args: ${args} =========="
    filename=${out_dir}/${lang}_${size}.out
    pred_dir=${out_dir}/${lang}_${size}/
    mkdir -p ${pred_dir}
    
    echo "python scripts/probe.py --probe ${probe} --model ${model} --lang ${lang} --pred_dir $pred_dir ${args} &> $filename" > $filename
    python scripts/probe.py --probe ${probe} --model ${model} --lang ${lang} --pred_dir $pred_dir "${@:1}" &>> $filename
    tail -n 1 $filename
  done
done
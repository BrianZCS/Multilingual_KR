#!/usr/bin/env bash
#SBATCH --mem=30000
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --output=slurm_out/slurm-%j.out

set -e

probe="mlamaf"
model="bert-base-multilingual-cased"  # model to use
langs=('tr' 'zh')  # language array to use
#fact_file=$4  # fact file
#facts=$5  # a list of facts joined by ","
out_dir="./evl_mbert/"  # dir to save output
args="${@:7}"

mkdir -p ${out_dir}

for lang in "${langs[@]}"; do
    echo "========== Language: $lang, Args: ${args} =========="
    filename=${out_dir}/${lang}.out
    pred_dir=${out_dir}/${lang}/
    mkdir -p ${pred_dir}
    
    echo "python scripts/probe_original.py --probe ${probe} --model ${model} --lang ${lang} --pred_dir $pred_dir ${args} &> $filename" > $filename
    python scripts/probe_original.py --probe ${probe} --model ${model} --lang ${lang} --pred_dir $pred_dir "${@:7}" &>> $filename
    tail -n 1 $filename
done
#!/bin/bash
set -e

if [ -z "$2" ]
then
    echo "Usage: $0 <target_script> <num_gpus>"
    exit 1
fi

echo "Running $1 with $2 GPUs"

sbatch -A phd -p dgxa100 -w dgxa100-ncit-wn01 --gres gpu:$2 --gpus $2 --time=1440 --ntasks 1 --cpus-per-task=8 --mem-per-cpu=8G run_apptainer.sh $1
#!/bin/bash
set -e

if [ -z "$2" ]
then
    echo "Usage: $0 <target_script> <num_gpus>"
    exit 1
fi

echo "Running $1 with $2 GPUs"
# sbatch -A nlp_highprio -p hd -w xl675dg10-wn175 --gres gpu:$2 --gpus $2 --time=50000 --qos=longterm --ntasks 1 --cpus-per-task=8 --mem-per-cpu=8G run_apptainer.sh $1
# sbatch -A nlp_highprio -p dgxh100 -w dgxh100-precis-wn02 --gres gpu:$2 --gpus $2 --time=50000 --qos=longterm --ntasks 1 --cpus-per-task=8 --mem-per-cpu=8G run_apptainer.sh $1
# sbatch -A nlp_highprio -p dgxa100 -w dgxa100-ncit-wn02 --gres gpu:$2 --gpus $2 --time=50000 --qos=longterm --ntasks 1 --cpus-per-task=8 --mem-per-cpu=8G run_apptainer.sh $1

sbatch -A nlp_highprio -p hd -w xl675dg10-wn175 --gres gpu:$2 --gpus $2 --time=50000 --qos=longterm --ntasks 1 --cpus-per-task=8 --mem-per-cpu=8G run_apptainer.sh $1
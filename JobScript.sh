#!/bin/bash
#PBS -P volta_pilot
#PBS -j oe
#PBS -N pytorch
#PBS -q DSA4266
#PBS -l select=1:ncpus=5:mem=50gb:ngpus=1
#PBS -l walltime=24:00:00

cd $PBS_O_WORKDIR;
np=$(cat ${PBS_NODEFILE} | wc -l);


image="/app1/common/singularity-img/3.0.0/pytorch_1.9_cuda_11.3.0-ubuntu20.04-py38-ngc_21.04.simg"
singularity exec $image bash  << EOF > stdout.$PBS_JOBID 2> stderr.$PBS_JOBID

cd /hpctmp/e0310593/yolov3

NAME="augmentedDatawithSmoothing_b5e100b001"
mkdir -p ./runs/train/$NAME
python3 ./train.py --img 640 --batch 5 --epochs 100 --data ./augmentedDatawithSmoothing.yaml --weights ./yolov3.pt --name $NAME --exist-ok --hyp ./hyp_box001.yaml > ./runs/train/$NAME/stdout.txt 2> ./runs/train/$NAME/stderr.txt

NAME="augmentedDatawithSmoothing_b32e100b001"
mkdir -p ./runs/train/$NAME
python3 ./train.py --img 640 --batch 32 --epochs 100 --data ./augmentedDatawithSmoothing.yaml --weights ./yolov3.pt --name $NAME --exist-ok --hyp ./hyp_box001.yaml > ./runs/train/$NAME/stdout.txt 2> ./runs/train/$NAME/stderr.txt

NAME="Vessel_Kayak_Count_Topup_b5e100b001"
mkdir -p ./runs/train/$NAME
python3 ./train.py --img 640 --batch 5 --epochs 100 --data ./Vessel_Kayak_Count_Topup.yaml --weights ./yolov3.pt --name $NAME --exist-ok --hyp ./hyp_box001.yaml > ./runs/train/$NAME/stdout.txt 2> ./runs/train/$NAME/stderr.txt

EOF
# BL's OCT-A Classifier
Benjamin Lee

## Intro 
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

## Installing the repo
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

## Enter Docker Environment
sudo docker build -t bleemasters . -f BL_dockerfile

sudo docker run --gpus all -v "/home/julius/Desktop/Ben Lee - Masters Thesis/":/benny -w /benny -it --rm bleemasters

sudo docker run --gpus all -v "/mnt/g/My Drive/Uni/Thesis/octa-mil-classifier":/benny -w /benny -it --rm bleemasters

## Command to Run
python src/BL_main.py -n BLMILTest -m train

## Commands in MUW HPC
ssh -l q59bleel s0-l00.hpc.meduniwien.ac.at

srun enroot import 'docker://nvcr.io#muwsc/zmpbmt/bleemasters:latest'

srun -p gpu -q 3g.20gb --gres=gpu:3g.20gb:1 --container-image=$HOME/octa-mil-classifier/ls muwsc+zmpbmt+bleemasters+latest.sqsh --pty bash

# To "send" your ssh key
eval $(ssh-agent -s)
ssh-add ~/.ssh/id_ed25519
ssh -A -l q59bleel s0-l00.hpc.meduniwien.ac.at

# start interactive development
srun -p gpu -q 3g.20gb --gres=gpu:3g.20gb:1 --container-image=$HOME/muwsc+zmpbmt+bleemasters+latest.sqsh --container-mount-home  --container-workdir /root/octa-mil-classifier/ --pty bash

# sbatch things
sbatch slurm_run.sh

# How to check what jobs are submitted to the cluster:
squeue -l

# How to check if ur jobs are running:
squeue -l | grep q59bleel

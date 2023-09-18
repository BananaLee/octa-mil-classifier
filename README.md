# BL's OCT-A Classifier
Benjamin Lee

## Intro 
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

## Installing the repo
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

## Enter Docker Environment
sudo docker build -t bleemasters213 . -f BL_dockerfile
sudo docker run --gpus all -v "/home/julius/Desktop/Ben Lee - Masters Thesis/":/benny -w /benny -it --rm bleemasters213


## Command to Run
python src/BL_main.py -n BLMILTest -m train

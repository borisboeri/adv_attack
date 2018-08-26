
# Requirements

- tf 1.4
- PIL 
- cv2

# HOWTO

    git clone https://github.com/assiaben/var_noise.git

Train-test

    ./scripts/run.sh <xp name> <dataset name> <epoch to train> <epoch interval> <net model>

- xp name: xp id, I usually go like xp0, xp1 ... and document the xp in log/README.md
- dataset name: cifar, mnist (currently available)
- epochs to train: number of epoch for the training (e.g. 10)
- epoch interval: interval of epochs between validation (e.g. 2)
- net model: network archi (vgg, alexnet, mnist tf currently available)

# Test the code
    
Train a the tf mnist example of mnist for 2 epochs and validate every 1 epoch. 

    ./scripts/run.sh test mnist 2 1 tf

Check that the code logs correctly and explore the logged data. The most
relevant information are in the tabs:
- scalar: for loss, accuracy
- images: to check that the input data is correct
- histograms: to check that the gradients do not behave weirdly

    tensorboard --logdir log/test

The code saves the log file in log/test/train and log/test/eval respectively. 
The net weights are saved in log/test/train

# Load vgg and alexnet weights

- Download the weights (ask me the path)
- Change the path to the weights in init\_weights.py
- 

Run the train

    ./scripts/run.sh test mnist 2 1 vgg
    ./scripts/run.sh test mnist 2 1 alexnet

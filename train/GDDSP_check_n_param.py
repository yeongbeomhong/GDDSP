### This is the main script of GDDSP that utilizes all sub-scripts for training ###

import torch
import torch.nn as nn
import omegaconf
import sys, os, glob
import numpy as np
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import time
from dataset_custom import Dataset_Dist_Slice_Overlap_Realtime
from GDDSP_total_net import GDDSP_wet
import random
from loss.mss_loss import MSSLoss
from optimizer.radam import RAdam

### Training Steps ###
# S0. Load the configurations from config.yaml. 
# Select the GPU as the number which user entered in terminal.(ex> --gpu "2,3") Check whether the device being used is GPU or GPU.
# Enable wandb to trace the log of training loss.
# S1. Declare Dataset and set the data_directory, shuffling, Batch_size, sample_rate, slicing(window size and hop size).
# S2. Load the AutoEncoder (난 지금 Encoder만 불러올거임. Oscillator를 위한 Amp, phase를 뱉는 디코더와 FX chain을 위한 패러미터를 뱉는 디코더 아직안짬)
# S3. Choose loss function, SGD Optimizer, and evaluation metric. #Default = Multi-Scaled Spectral Loss(MSS) and Adam Optimizer
# S4. Load the Learning Schedule and the Validation Period from config file.
# S5. Train.  (Feed the settings of data,net,loss,schedule into trainer)
# S6. Save the tensor of trained Parameters at Checkpoint_folder.
# S7. Save the change in the training loss at log_folder.
### Let's Start ###

# S0. Load the configurations from config.yaml. Select the GPU as the number user enters in terminal. Enable Wandb.
def read_yaml_with_dot_parse(yaml_file):
    # Load and parse the YAML file
    config = omegaconf.OmegaConf.load(yaml_file)
    return config

config = read_yaml_with_dot_parse("./GDDSP_config_wet.yaml")
random.seed(config.seed)
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default=None, help= "Select the GPU with terminal argument (e.g. --gpu '3' or --gpu '3,4,5' )")
args = parser.parse_args()

if args.gpu is not "cpu" and args.gpu is not None:
    os.environ.update({"CUDA_VISIBLE_DEVICES": str(args.gpu)})
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
    print(f"GPU {str(args.gpu)} will be used during training.")
    print('Count of using GPUs:', torch.cuda.device_count())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Choice of CPU/GPU  :{device}") #Check whether the device being used is GPU or GPU.
else:
    print("No specific GPU was chosen.")
    print('Count of using GPUs:', torch.cuda.device_count())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device :{device}")
    print(f"Choice of CPU/GPU :{device}")  #Check whether the device being used is GPU or GPU.

# S2. Load the Total Strudcture of GDDSP (= encoder- decoder - osillator and noise generator - FX chain)
net = GDDSP_wet(config)
net.to(device)

trainable_parameters_list = filter(lambda p: p.requires_grad, net.parameters())
num_trainable_parameters = sum([torch.numel(p) for p in trainable_parameters_list])
print("\n The num of trainable parameters in net : ", num_trainable_parameters)

def count_param(layer_name):
    tpl = filter(lambda p: p.requires_grad, layer_name.parameters())
    ntp = sum([torch.numel(p) for p in trainable_parameters_list])
    return ntp

import pdb
pdb.set_trace()
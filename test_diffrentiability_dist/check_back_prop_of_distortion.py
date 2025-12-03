

# Instead of feeding clean audio reconstructed from H.O and N.G to Distrotion NN,
# Just Test the Differentiability and Back-propagtion of Distortion NN, by feeding clean GuitarSet- Distorted GuitarSet Pair.

import torch
import torch.nn as nn
import torchaudio
import omegaconf
import sys, os, glob
import numpy as np
import argparse
import pdb
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim

from dataset import Dataset_Dist_Slice_Overlap_Realtime
from components.fx_chain import W_H_Distortion
import random
from loss.mss_loss import MSSLoss
from optimizer.radam import RAdam

def read_yaml_with_dot_parse(yaml_file):
    # Load and parse the YAML file
    config = omegaconf.OmegaConf.load(yaml_file)
    return config
config = read_yaml_with_dot_parse("./check_back_prop_of_distortion.yaml")

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default=None, help= "Select the GPU with terminal argument (e.g. --gpu '3' or --gpu '3,4,5' )")
args = parser.parse_args()

import wandb
wandb.init(project="GDDSP_Train", name= config.experiment_name)

if args.gpu is not None:
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


train_dataset = Dataset_Dist_Slice_Overlap_Realtime("/data4/guitarset/audio_hex-pickup_debleeded/train/", sr = config.sample_rate, slice_length = config.slice_length, slice_hop = config.slice_hop, input_dist_amount_random =config.input_dist_amount_random)
batch_size = config.batch_size
train_dataloader = DataLoader(
    train_dataset,
    batch_size= batch_size,
    shuffle=True,
    num_workers= 4,
    pin_memory=True,
)

print("The train_dataset, valid_dataset and dataloader has been built.")
print(f"Total Counts of training audio excerpts with fixed duration : {len(train_dataset)}")
print(f"Total Time_durtaion of training audio excerpts : {config.slice_length * len(train_dataset)}")



# EQ_FR의 (batch, frame)별로 65개의 freq bin 값과, power_tanh 함수의 각 term의 실수배 계수는 모두 디코더에게서 전달받아야함.
# 왜냐하면 Input Audio의 디스토션 성질에 따라, 위 값들은 변해야하는 변수이기 때문
# 그러나 아직은 디코더에서 위 변수들을 생성하는 GRU + DENSE 맵핑 레이어를 만들지 않았으므로,
# 일단 디코더로부터 전달받았다고 치고, 그 값이 여러 Clean-Dist Pair를 경험하면서 "역전파를 통해" "미분그래프가 끊기지 않고" 훈련되는지만 체크해보자.


# S1. Make Trainable Instance of EQ_FR(w/ shape of (16, 65)) and ctanh w/shape of (16,20)
Instance_dist_pre_eq = torch.zeros((16,65), device=device, requires_grad=True)
Instance_dist_ctanh = torch.zeros((16,20), device=device, requires_grad=True)

# S2. Construct W_H_Distortion which contains its own Trainable EQ_FR params and Trainable PowerTanh params.
net = W_H_Distortion(config=config, device = "cuda")

# S3. Choose loss function, SGD Optimizer, and validation metric. #Default = Multi-Scaled Spectral Loss(MSS) and Adam Optimizer
loss = MSSLoss([2048, 1024, 512, 256], signal_key = "audio_dist").to(device)
# MSS Loss between audio_dist_pred and audio_dist_GT 


# Setting Optimizer
if config.optimizer == "adam":
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=config.lr)
elif config.optimizer == "radam":
    optimizer = RAdam([Instance_dist_pre_eq, Instance_dist_ctanh], lr=config.lr)
else:
    raise NotImplementedError

# S4. Load the Learning Schedule and the Validation Period from config file.
# Setting Scheduler
if config.lr_scheduler == "cosine":
    # restart every T_0 * validation_interval steps
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, eta_min=config.lr_min
    ) # The input argument "optimizer" to define the scheduler must matches with your optimizer's name
     
elif config.lr_scheduler == "plateau":
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=config.lr_decay
    )
elif config.lr_scheduler == "multi":
    # learning rate decreases by every 50 epochs
    # 1~50 epoch :     lr = config.lr
    # 51~100 epoch :   lr =  config.lr * gamma
    # 101~150 epoch :   lr =  config.lr * gamma * gamma 
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        [(x + 1) * config.lr_decay_period for x in range(50)], #[10 ,20, ... 500]
        gamma=config.lr_decay,
    )
    # Write scheduler.step() in Epoch-Loop.
    # Then the scheduler will count the finished epochs
    # When the counts of finished epochs match 10, 20 ,.. 500, The scheduler reduces lr of optimizer with decay ratio.
    print(f"The lr_scheduler will reduce the learning rate every {config.lr_decay_period} epoch.")
elif config.lr_scheduler == None:
    scheduler = None
else:
    raise ValueError(f"unknown lr_scheduler : {config.lr_scheduler}")
print(f"The learning Schedule during training : {config.lr_scheduler}")
# --------------------------------------- #


max_epoch = 500
### Main Training Loop ###
for i in tqdm(range(max_epoch), desc="epochs"):
    epoch = i + 1
    print(f"Distortion check backprop Training of {epoch}th epoch started")
    batches_in_one_epoch = int(len(train_dataset) / config.batch_size)
    avg_loss_value = 0
    batches_in_one_epoch = int(len(train_dataset) / config.batch_size)
    print(f"There are {batches_in_one_epoch} batches per epoch, and {config.batch_size} audio excerpts per batch.")
    
    for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="Batches") : # Iterations in one epoch
        # Feed batch_data to GPU
        
        batch["audio_dry_GT"] = batch["audio_dry_GT"].to(device)
        batch["audio_wet_GT"] = batch["audio_wet_GT"].to(device)
        audio_dist_pred = net(batch["audio_dry_GT"], {"dist_pre_eq": Instance_dist_pre_eq, "dist_ctanh" : Instance_dist_ctanh})
        
        print("the shape of Predicted Dist Audio(=Clean + W_H Effect): ", audio_dist_pred.shape)
        
        loss_value = loss(audio_dist_pred, batch["audio_wet_GT"])
        
        optimizer.zero_grad()  # Clear the gradients
        loss_value.backward()  # Compute gradients
        optimizer.step()  # Update model parameters
        avg_loss_value += loss_value.detach().item()
        
    scheduler.step()
    avg_loss_value = avg_loss_value / batches_in_one_epoch
    wandb.log({"training_loss_VS_epoch": avg_loss_value})
    
    # save the parameters periodically
    if epoch % config.check_point_save_period == 0:
        if not os.path.exists(config.check_point_save_path + config.experiment_name):
            os.makedirs(config.check_point_save_path + config.experiment_name)
        ckp_path = config.check_point_save_path + config.experiment_name + f"/batchsize{config.batch_size}_epoch{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': [Instance_dist_pre_eq, Instance_dist_ctanh],
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss_value,
            }, ckp_path)
        print(f"The trained parameters are saved at {ckp_path}")
    # save the sample audio reconstructed by GDDSP periodically with .wav format, and compare it with original audio of same data_idx.
    if epoch % config.sample_audio_recon_save_period == 0:
        if not os.path.exists(config.sample_audio_recon_save_path + config.experiment_name):
            os.makedirs(config.sample_audio_recon_save_path + config.experiment_name)
        audio_path = config.sample_audio_recon_save_path + config.experiment_name + f"/batchsize{config.batch_size}_epoch{epoch}"
        
        sample_original_audio = batch["audio_wet_GT"][0].detach().cpu().numpy()
        sample_recon_audio = audio_dist_pred[0].detach().cpu().numpy()
        # index [0] picks The last batch's first audio excerpt
        # The audio in last batch may differs by epoch. (as the train_dataloader use "shuffle")
        import soundfile as sf
        sr = config.sample_rate
        sf.write(audio_path + "_dist_GT_torch_overdrive.wav", sample_original_audio, sr)
        sf.write(audio_path + "_dist_pred_W_H_model.wav", sample_recon_audio, sr)
        print(f"The sample audio re-constructed by GDDSP is saved at {audio_path}")
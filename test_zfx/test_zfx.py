### This is the main script of GDDSP that utilizes all sub-scripts for training ###

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim

import omegaconf
import sys, os, time, random
import json
import numpy as np
import argparse
from tqdm import tqdm
from simple_estimator_structure import MLP

sys.path.append("../train")
from GDDSP_total_net import GDDSP_wet
from dataset_custom import Dataset_Dist_Slice_Overlap_Realtime
from optimizer.radam import RAdam

### Training Steps ###
# S0. Load the configurations from config.yaml. Select GPU, Enable Wandb #
# Select the GPU as the number which user entered in terminal.(ex> --gpu "2,3") Check whether the device being used is GPU or GPU.
# Enable wandb to trace training loss and validation metric.
# S1. Declare Dataset and set the data_directory, shuffling, Batch_size, sample_rate, slicing(window size and hop size).
# S2. Load the Structure of GDDSP(=Neural Net that analyzes, disentangles, and reconstructs audio)
# S3. Load the Checkpoint of GDDSP, and conduct "to(device)"
# S4. Load the Sturcture of Distortion Estimator 
# Estimator : Simple MLP that infers gain knob and colour knob of input audio, only using zfx as input information
# S5. Declare Training Loss and Validation Metric of Estimator
# S6. Setting Optimizer and lr-scheduler.
# S7. Train
#   Get gain_GT and colour_GT from nn.Dataloader directly.
#   Get zfx(intermediate latent vector of GDDSP, representing Distortion) and feed it into Estimator
#   Get gain_pred and colour_pred from Estimator.forward
#   Calculate nn.MSE between [gain_GT,colour_GT] and gain_pred, colour_pred
#   optimizer.zero_grad + loss.backward + grad.clipping + optimizer.step
#   It the epoch % valid_period=0
# S8. Save the trained Parameters of Estimator at Checkpoint_folder.

# S9. Validation Phase : Calulate Valid Loss and Valid Accuracy
# S10. Save the predicted knob value example in valid phase at config.valid_pred_knob_example_path
# (optional step) S11. Save the t-SNE picture clustering gain=5, 10, 20, 30 with valid input data

### Let's Start ###
# S0. Load the configurations from config.yaml. Select GPU, Enable Wandb #
def read_yaml_with_dot_parse(yaml_file):
    # Load and parse the YAML file
    config = omegaconf.OmegaConf.load(yaml_file)
    return config

config = read_yaml_with_dot_parse("./test_zfx_config.yaml")
random.seed(config.seed)
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default=None, help= "Select the GPU with terminal argument (e.g. --gpu '3' or --gpu '3,4,5' )")
args = parser.parse_args()

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

import wandb
experiment_full_name =  config.experiment_name + \
    f"Batch{config.batch_size}"
wandb.init(project="GDDSP_Train", name= experiment_full_name)

# S1. Declare Dataset and set the data_directory, shuffling, Batch_size, sample_rate, slicing(window size and hop size).
train_dataset = Dataset_Dist_Slice_Overlap_Realtime(
    config = config,
    device = device,
    audio_path = config.train_path
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    pin_memory=False,
)
valid_dataset = Dataset_Dist_Slice_Overlap_Realtime(
    config = config,
    device = device,
    audio_path = config.valid_path
)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size = config.batch_size,
    #batch_size=len(valid_dataset), # <- If you Group all audio excerpts in valid dataset into One Batch, It will exceed CUDA Memory allocation-limit.
    shuffle=True,
    num_workers=config.num_workers,
    pin_memory=False,
)

print(f"Total Counts of training audio excerpts with fixed duration : {len(train_dataset)}")
print(f"Total Counts of valid audio excerpts with fixed duration : {len(valid_dataset)}")

# S2. Load the Structure of GDDSP(=Neural Net that analyzes, disentangles, and reconstructs audio)
net = GDDSP_wet(config).to(device)
# S3. Load the Checkpoint of GDDSP, freeze parameter of GDDSP, to prevent updating params from backpropagation with knob estimation
if config.check_point_load_path == None :
    raise ValueError("You must load the Pre_Trained Model to Evaluate usefulness of ZFX")
else: # load parameters of checkpoint
    print(f"The path of checkpoint is {config.check_point_load_path}")
    pretrained_state_dict = torch.load(config.check_point_load_path, map_location=device)
    net.load_state_dict(pretrained_state_dict["model_state_dict"])
    GDDSP_epoch = pretrained_state_dict["epoch"] # succeeding the finished epochs of pre-trained ckp.
    print(f"You picked Pre-Trained GDDSP trained with {GDDSP_epoch} epochs")
for param in net.parameters():
    param.requires_grad = False
print("Freezing GDDSP finished.")
# S4. Load the Sturcture of Distortion Estimator 
estimator = MLP(n_input = config.z_units, n_units = config.estimator.mlp_units, n_layer = config.estimator.mlp_layers, relu=nn.ReLU).to(device)
# n_units = 2 means two output value [gain, colour]


# S5. Declare Training Loss and Validation Metric of Estimator #
loss_function = nn.MSELoss().to(device)
# for training, Calculate MSELoss
# for validation, Calcultae MSELoss and Accuracy. 
# Accuracy = (count_pred with Gain_error<1 && Colour_error<1)   /  (count_total_pred)

# S6. Setting Optimizer and lr-scheduler, and grad Limit #
if config.optimizer == "adam":
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, estimator.parameters()), lr=config.lr)
elif config.optimizer == "radam":
    optimizer = RAdam(filter(lambda x: x.requires_grad, estimator.parameters()), lr=config.lr)
else:
    raise NotImplementedError
max_norm = 5.

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
        [(x + 1) * config.lr_decay_period for x in range(50)], # epoch turning point to decrease lr : [10 ,20, ... 500]
        gamma=config.lr_decay, #lr decreasing ratio
    )
    # Write scheduler.step() in Epoch-Loop.
    # Then the scheduler will count the finished epochs
    # When the counts of finished epochs match 10, 20 ,.. 500, The scheduler reduces lr of optimizer with decay ratio.
    print(f"The lr_scheduler will reduce the learning rate every {config.lr_decay_period} epoch.")
elif config.lr_scheduler == "no":
    scheduler = None
else:
    raise ValueError(f"unknown lr_scheduler selected : {config.lr_scheduler}")

print(f"The learning Scheduler during training : {config.lr_scheduler}")
# --------------------------------------- #

trainable_parameters_list = filter(lambda p: p.requires_grad, estimator.parameters())
num_trainable_parameters = sum([torch.numel(p) for p in trainable_parameters_list])
print("\n The num of trainable parameters in Dist_Knob_Estimator : ", num_trainable_parameters)
L1Loss_function = nn.L1Loss().to(device)

# S7. Main Training Loop (zero_grad, forward propagation with network, backpropagation, save parameters at ckp) #
for epoch in tqdm(range(config.max_epoch), desc="epochs"):
    epoch += 1
    avg_loss_value = 0
    batches_in_one_epoch = len(train_dataloader)
    print("Lets Train Estimator using ZFX of GDDSP as Input, and Estimates Dist-Knobs reversely as output.")
    print(f"There are {batches_in_one_epoch} batches per epoch, and {config.batch_size} audio excerpts per batch.")
    print(f"Start the training of {epoch}th epoch.")
    for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="Batches") : # Iterations in one epoch
        start_iter = time.time()
        print(f"Now processing {batch_idx}th/{len(train_dataloader)} batch in {epoch}th epoch.")
        with torch.no_grad():
            batch["audio_wet_GT"] = batch["audio_wet_GT"].to(torch.float32).to(device)
            zfx_average_along_frames = net.produce_zfx(batch)
        GT_knob =  torch.cat((batch["gain_GT"].unsqueeze(1).to(torch.float32), batch["colour_GT"].unsqueeze(1).to(torch.float32)), dim=1).to(device)
        pred_knob = estimator(zfx_average_along_frames)
        loss_value = loss_function(pred_knob, GT_knob)

        optimizer.zero_grad()
        loss_value.backward()
        nn.utils.clip_grad_norm_(estimator.parameters(), max_norm)
        optimizer.step()  # Update estimator parameters, to get Better Mappling between in:zfx and out:gain&colour
        avg_loss_value += loss_value.detach().item() # avg_loss = average loss of all audio data in one epoch
        print("(Knob Estimator)Time for One Iteration to train : ", time.time()-start_iter)
        
    scheduler.step()
    avg_loss_value = avg_loss_value / len(train_dataloader)
    wandb.log({"(Knob Estimator) train loss VS epoch": avg_loss_value,
               "epoch" : epoch})
    
    ## S8. Save the trained Parameters of Estimator at Checkpoint_folder
    if epoch % config.check_point_save_period == 0:
        if not os.path.exists(config.check_point_estimator_path + experiment_full_name):
            os.makedirs(config.check_point_estimator_path + experiment_full_name)
        ckp_path = config.check_point_estimator_path + experiment_full_name + f"/estimator_params_epoch{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': estimator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_loss_value
            }, ckp_path)
        print(f"(Knob Estimator)The trained parameters are saved at {ckp_path}")
    
    
    # S9. Validation Phase : Calulate Valid Loss and Valid Accuracy
    if epoch % config.valid_period  == 0:
    
        avg_valid_loss_value = 0
        count_pred_near_GT = 0 # Knob Estimation with gain L1 loss < 1 && colour L1 loss <1
        optimizer.zero_grad()  # Clear the gradients
        print(f"validation at { epoch}th epoch started.")
        with torch.no_grad():
            for batch_idx, valid_batch in tqdm(enumerate(valid_dataloader), desc="Batches"):
                print(f"Now processing {batch_idx}th/{len(valid_dataloader)} valid batch in {epoch}th epoch.")

                valid_batch["audio_wet_GT"] = valid_batch["audio_wet_GT"].to(torch.float32).to(device)
                zfx_average_along_frames = net.produce_zfx(valid_batch)
                GT_knob = torch.cat((valid_batch["gain_GT"].unsqueeze(1).to(torch.float32), valid_batch["colour_GT"].unsqueeze(1).to(torch.float32)), dim=1).to(device)
                pred_knob = estimator(zfx_average_along_frames)
                
                valid_loss_value = loss_function(pred_knob, GT_knob)
                avg_valid_loss_value += valid_loss_value.detach().item()
                
                batchwise_L1_loss_gain = L1Loss_function(pred_knob[..., 0],  GT_knob[...,0])
                batchwise_L1_loss_colour = L1Loss_function(pred_knob[...,1], GT_knob[...,1])
                boolean_tensor_gain_approxi_accurate = torch.where(batchwise_L1_loss_gain  < 1, 0, 1)
                boolean_tensor_colour_approxi_accurate = torch.where(batchwise_L1_loss_colour  < 1, 0, 1)
                boolean_tensor_both_accurate = boolean_tensor_gain_approxi_accurate * boolean_tensor_colour_approxi_accurate
                count = torch.count_nonzero(boolean_tensor_both_accurate)
                count_pred_near_GT = count_pred_near_GT + count.detach().item()
                
        avg_valid_loss_value = avg_valid_loss_value / len(valid_dataloader)
        #valid loss is already averaged within iteration(batch size), so divide it only with num of batches
        accuracy = count_pred_near_GT / len(valid_dataset) 
        #counts are not yet avaraged within iteration, so divide it with whole number of valid data
        #accuracy max = 1 
        
        wandb.log({"(Knob Estimator) valid loss VS epoch": avg_valid_loss_value,
                    "(Knob Estimator) accuracy VS epoch": accuracy,
                    "epoch" : epoch})
        print(f"(Knob Estimator) validation at {epoch}th epoch finished.")

        # S10. Save the predicted knob value example in valid phase at config.valid_pred_knob_example_path
        # Save the Distortion_Knob_values of first audio of last iteration
        if not os.path.exists(config.valid_knob_example_path + experiment_full_name):
            os.makedirs(config.valid_knob_example_path+ experiment_full_name)
        example_knob = {'gain_GT': GT_knob[0][0].item(), 'colour_GT': GT_knob[0][1].item(), \
            'gain_pred' : pred_knob[0][0].item(), 'colour_pred' : pred_knob[0][1].item(), 'epoch':epoch}
        filename = f"/knob_example_epoch{epoch}.txt"
        with open(config.valid_knob_example_path+ experiment_full_name + filename, 'w') as file:
            file.write(json.dumps(example_knob))

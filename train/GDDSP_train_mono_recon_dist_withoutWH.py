### Train Mono GDDSP whose input audio and recon audio is string-wise monophonic tensor.
### During Training, Feed mono audio and calculate dry_loss(mono_dry_GT, mono_dry_pred) and wet_loss(mono_wet_GT, mono_dry_pred)
### During Valid nad Test, Set "random" argument in dataloader "False",
### 

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
from dataset_custom import Dataset_Dist_Slice_Overlap_Realtime_Stringwise
from GDDSP_total_net import GDDSP_wet_mono
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

config = read_yaml_with_dot_parse("./GDDSP_config_mono_recon_dist_withoutWH.yaml")
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

D=0 # Dataset Option
DL=0 # Main Comparison Option : Use W_H "Distortion Layer" or not
AR=0 # Use Random Amount Distortion or Fixed Amount Distortion
RE=0 # Use Reverb or Not
if config.input_dist_type == "torch_overdrive":
    D=1
elif config.input_dist_type == "spotify_pedalboard":
    D=2
elif config.input_dist_type == "guitar_real_plugin":
    D=3

if config.n_harmonics == 101 :
    DL = 2 #NN (Neither DL, Nor More Harmonics) recon_dist_without_fx_chain.py always dont use W_H Distortion Layer
elif config.n_harmonics > 101 :
    DL = 3 #MH (More Harmoncis)
else:
    pass

if config.input_dist_amount_random == True:
    AR=1
else: 
    AR=2
if config.use_room_acoustic == True:
    RE=1
else:
    RE=2
    
if config.input_dist_amount_random == True:
    experiment_full_name =  config.experiment_name + \
        f"D{D}DL{DL}AR{AR}RE{RE}_nHarm{config.n_harmonics}_Monophonic_ReconDist_withoutWH"
else:
    experiment_full_name =  config.experiment_name + \
        f"D{D}DL{DL}AR{AR}RE{RE}_nHarm{config.n_harmonics}_Monophonic_ReconDist_withoutWH"

wandb.init(project="GDDSP_Train", name= experiment_full_name)

# S1. Load Dataset and set the configurations of data_directory, shuffling, Batch_size, sample_rate, slicing(window size and hop size).
train_dataset = Dataset_Dist_Slice_Overlap_Realtime_Stringwise(
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

valid_dataset = Dataset_Dist_Slice_Overlap_Realtime_Stringwise(
    config = config,
    device = device,
    audio_path = config.valid_path
)

valid_dataloader = DataLoader(
    valid_dataset,
    batch_size = config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=False,
)

print("The train_dataset, valid_dataset and dataloader has been built.")
print(f"Total Counts of training audio excerpts with fixed duration : {len(train_dataset)}")
print(f"Total Time_durtaion of training audio excerpts : {config.slice_length * len(train_dataset)}")

# S2. Load the Total Strudcture of GDDSP (= encoder- decoder - osillator and noise generator - FX chain)
net = GDDSP_wet_mono(config)


if config.train_check_point_load_path == None :
    print("New Monophonic withoutWH string-wise Model will be trained from scratch_state.(Randomly Initialized Parameters)")
    start_epoch = 0
else: # load parameters of checkpoint
    print("You chose to train the model from checkpoint.(Continuing the previous training)")
    print(f"The path of checkpoint is {config.train_check_point_load_path}")
    pretrained_state_dict = torch.load(config.train_check_point_load_path)
    net.load_state_dict(pretrained_state_dict["model_state_dict"])
    start_epoch = pretrained_state_dict["epoch"] # succeeding the finished epochs of pre-trained ckp.
    print("The parameters has been loaded.")

#Parellel train code
if torch.cuda.device_count() > 1: # Parallel training to efficeintly use Multiple GPUs' memories
    print(f"Use {torch.cuda.device_count()} GPUs for Multi-GPU-Parallel-Training.")
    net = nn.DataParallel(net)
else :
    pass
net.to(device)


# S3. Choose loss function, SGD Optimizer, and validation metric. #Default = Multi-Scaled Spectral Loss(MSS) and Adam Optimizer
# Define training loss, and select FX types to be applied on wet input and wet recon audio.
if config.use_room_acoustic == True:
    signal_key = "audio_room" #Signal Key cannot incldue "dist" in this script, as the W_H dist layer is not used during Reconstrucion.
else:
    signal_key = "audio_dry"
print("The signal_key to be compared with input dist audio is ", signal_key)
loss = MSSLoss([2048, 1024, 512, 256], signal_key = signal_key).to(device)


# Define evaluation(validation) metrics
if config.metric == "mss":
    def metric(output, gt):
        with torch.no_grad(): #no grad_backpropagtion during validation
            return loss(output, gt)
        
elif config.metric == "f0":
    # TODO Implement
    raise NotImplementedError
else:
    raise NotImplementedError


# Setting Optimizer and Gradient Limit Clipping
if config.optimizer == "adam":
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=config.lr)
elif config.optimizer == "radam":
    optimizer = RAdam(filter(lambda x: x.requires_grad, net.parameters()), lr=config.lr)
else:
    raise NotImplementedError
max_norm = 5.

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
    raise ValueError(f"unknown lr_scheduler : {config.lr_scheduler}")

print(f"The learning Schedule during training : {config.lr_scheduler}")

# --------------------------------------- #
trainable_parameters_list = filter(lambda p: p.requires_grad, net.parameters())
num_trainable_parameters = sum([torch.numel(p) for p in trainable_parameters_list])
print("\n The num of trainable parameters in MonoDDSP+ No W_H Dist : ", num_trainable_parameters)

# Main Training Loop (zero_grad, forward propagation with network, backpropagation, save parameters at ckp) #
if config.batch_size % 6 != 0 :
    raise ValueError("Mono DDSP must process audio data with Batch Unit of integer multiple of Six")

time_start_train = time.time()
for epoch in tqdm(range(config.max_epoch), desc="epochs"):
    epoch += 1
    avg_loss_stringwise = 0
    avg_loss_hexaphonic = 0 
    
    batches_in_one_epoch = len(train_dataloader)
    print(f"There are {batches_in_one_epoch} batches per epoch, and {config.batch_size} audio excerpts per batch.")
    print(f"Start the training of MonoDDSDP + No W_H Dist on {start_epoch + epoch}th epoch.")

    for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="Batches") : # Iterations in one epoch
         
        # Clear the gradients
        # Feed Input audio(wet) to GPU (batch-wise)
        # GDDSP use only audio_wet_GT as input, and then reconstruct de-effected audio(=dry audio) by using Oscillators and Noise Generator.d (Dont use WH dist layer)
        start_iter =time.time()
        print(f"(Mono DDSP Train)Processing {batch_idx}th/{len(train_dataloader)} batch in {start_epoch + epoch}th epoch.")
        print("Counts of 4_sec_sliced audio in one batch : ", batch["audio_wet_GT"].shape[0])
        batch["audio_wet_GT"] = batch["audio_wet_GT"].to(device) #This is input data, and will be used to calculate loss between wet_GT and wet_recon
        audio_recon = net(batch)
        loss_stringwise = loss(audio_recon[signal_key], batch["audio_wet_GT"]) # Penalize reconstructed audio after oscill or reverb to be close to wet(dist) audio
        # Do not need backpropagation of loss_dry + loss_wet, as without_WH model receive backpropagation only from output of H.Osillator+N.Generator
        num_audio_this_batch = batch["audio_wet_GT"].shape[0]
        wet_GT_group_six_string_into_hexa_audio = torch.sum(torch.reshape(batch["audio_wet_GT"], (num_audio_this_batch//6, 6, -1)), dim = 1)
        wet_recon_group_six_string_into_hexa_audio = torch.sum(torch.reshape(audio_recon[signal_key], (num_audio_this_batch//6, 6, -1)), dim=1) # [36,88200] -> [6,6,88200] -> after sum, [6, 88200]
        loss_hexaphonic = loss(wet_recon_group_six_string_into_hexa_audio, wet_GT_group_six_string_into_hexa_audio).detach().item()
        
        optimizer.zero_grad()
        loss_stringwise.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm)
        optimizer.step()  # Update model parameters
        avg_loss_stringwise += loss_stringwise.detach().item()
        avg_loss_hexaphonic += loss_hexaphonic
        print("Time Consuming for One Iteration is : ", time.time()-start_iter)
    scheduler.step()
    avg_loss_stringwise = avg_loss_stringwise/ len(train_dataloader) # Averaging on audios in one batch was already done by MSS L1 Loss function
    avg_loss_hexaphonic = avg_loss_hexaphonic/ len(train_dataloader) # So to get loss per one audio, Just add process of Averaging loss on count of batches
    wandb.log({"(mono withoutWH train) dist_loss_stringwise VS epoch": avg_loss_stringwise ,
               "(mono withoutWH train) dist_loss_hexaphonic VS epoch": avg_loss_hexaphonic,
               "epoch" : start_epoch+epoch,
               "elapsed_time" : time.time() - time_start_train,
               "num_trainable_params" : num_trainable_parameters})
    
    
    ## save the parameters periodically
    if epoch % config.check_point_save_period == 0 and epoch > 10:
        if not os.path.exists(config.check_point_save_path + experiment_full_name):
            os.makedirs(config.check_point_save_path + experiment_full_name)
        ckp_path = config.check_point_save_path + experiment_full_name + f"/joint_loss_epoch{start_epoch + epoch}.pt"
        torch.save({
            'epoch': start_epoch + epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, ckp_path)
        print(f"The trained parameters are saved at {ckp_path}")
    
    ## save the sample audio reconstructed by GDDSP periodically with .wav format, and compare it with original audio of same data_idx.
    if epoch % config.sample_audio_recon_save_period == 0 and epoch > 10 :
        import soundfile as sf
        sr = config.sample_rate
        if not os.path.exists(config.sample_audio_recon_save_path + experiment_full_name):
            os.makedirs(config.sample_audio_recon_save_path + experiment_full_name)
        audio_path = config.sample_audio_recon_save_path + experiment_full_name + f"/useWH_epoch{start_epoch + epoch}"
        sample_original_audio_dry = batch["audio_dry_GT"][0].detach().cpu().numpy()
        sample_original_audio_dist = batch["audio_wet_GT"][0].detach().cpu().numpy()
        sample_original_audio_dist_hexa = sample_original_audio_dist = torch.sum(batch["audio_wet_GT"][0:6], dim=0).detach().cpu().numpy()
        sf.write(audio_path + "_original_dry.wav", sample_original_audio_dry, sr)
        sf.write(audio_path + "_original_dist.wav", sample_original_audio_dist, sr)
        sf.write(audio_path + "_original_dist_hexa.wav", sample_original_audio_dist_hexa, sr)
        # batch["~"][0] = The first audio excerpt of final minibatch of present epoch.
        # selected sample audio excerpt differs by epoch. (as the train_dataloader use "shuffle")
        if config.use_room_acoustic == True:
            sample_recon_audio_dist_before_RE = audio_recon["audio_dry"][0].detach().cpu().numpy()
            sample_recon_audio_dist_after_RE = audio_recon["audio_room"][0].detach().cpu().numpy()
            sample_recon_audio_dist_hexa_sum_after_RE = torch.sum(audio_recon["audio_room"][0:6], dim=0).detach().cpu().numpy()
            sf.write(audio_path + "_recon_dist_before_RE.wav", sample_recon_audio_dist_before_RE, sr)
            sf.write(audio_path + "_recon_dist_after_RE.wav", sample_recon_audio_dist_after_RE, sr)
            sf.write(audio_path + "_recon_dist_hexa_sum_after_RE.wav", sample_recon_audio_dist_hexa_sum_after_RE, sr)
        else: #Reverb layer wasnt constructed
            sample_recon_audio_dist_without_WH_without_RE = audio_recon["audio_dry"][0].detach().cpu().numpy()
            sample_recon_audio_dist_hexa_sum_without_WH_without_RE= torch.sum(audio_recon["audio_dry"][0:6], dim=0).detach().cpu().numpy()
            sf.write(audio_path + "_recon_dist_without_WH_withoutRE.wav", sample_recon_audio_dist_without_WH_without_RE, sr)
            sf.write(audio_path + "_recon_dist_hexa_sum_without_WH_withoutRE.wav", sample_recon_audio_dist_hexa_sum_without_WH_without_RE, sr)
            
        print(f"The sample audio reconstructed by GDDSP is saved at {audio_path}")
    
    ## Calculate valid loss Periodically, and Trace Valid loss with wandb
    ## For Fair Comparison with Polyphonic GDDSP,(To Set Task Same,) Cacluate MSS Loss of 6 string GT Sum Vs. 6 string Recon Sum, Not Mono Vs Mono.

    if epoch % config.valid_period  == 0 :
        avg_valid_loss_stringwise = 0
        avg_valid_loss_hexaphonic = 0
        optimizer.zero_grad()  # Clear the gradients
        print(f"Mono GDDSP validation at {start_epoch + epoch}th epoch started.")
        with torch.no_grad():
            for batch_idx, valid_batch in tqdm(enumerate(valid_dataloader), desc="Batches"):
                print(f"{batch_idx+1}th Mono GDDSP valid batch is under processing. There are {len(valid_dataloader)} batches in valid dataset")
                print(f"There are {len(valid_dataset) // len(valid_dataloader)} audio excerpts in one minibatch of valid dataset")
                valid_batch["audio_wet_GT"] = valid_batch["audio_wet_GT"].to(device)
                audio_recon = net(valid_batch)
                
                num_audio_this_batch = valid_batch["audio_wet_GT"].shape[0]
                valid_loss_stringwise = loss(audio_recon[signal_key], valid_batch["audio_wet_GT"])
                wet_GT_group_six_string_into_hexa_audio = torch.sum(torch.reshape(valid_batch["audio_wet_GT"], (num_audio_this_batch //6, 6, -1)), dim = 1)
                wet_recon_group_six_string_into_hexa_audio = torch.sum(torch.reshape(audio_recon[signal_key], (num_audio_this_batch //6, 6, -1)), dim= 1) # [36,88200] -> [6,6,88200] -> after sum, collect sum of dim1 per six elements into dim0 [6, 88200]
                valid_loss_hexaphonic = loss(wet_recon_group_six_string_into_hexa_audio, wet_GT_group_six_string_into_hexa_audio)
                # Calculate dist audio loss by comparting overall sum of 6 strings(GT) vs 6 strings(Recon). 
                # Neighboring six audio in minibatch represents six strings from same audio source slice.
                
                avg_valid_loss_stringwise += valid_loss_stringwise.detach().item() 
                avg_valid_loss_hexaphonic += valid_loss_hexaphonic.detach().item() 
                
        avg_valid_loss_stringwise = avg_valid_loss_stringwise/ len(valid_dataloader)
        avg_valid_loss_hexaphonic = avg_valid_loss_hexaphonic / len(valid_dataloader)
        wandb.log({"(mono withoutWH valid) dist_loss_stringwise VS epoch": avg_valid_loss_stringwise,
                   "(mono withoutWH valid) dist_loss_hexaphonic VS epoch": avg_valid_loss_hexaphonic,
                   "epoch" : start_epoch+epoch})
        print(f"validation at {start_epoch + epoch}th epoch finished.")

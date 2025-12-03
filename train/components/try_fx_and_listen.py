import sys
sys.path.append("..")
import torch
import torchaudio
from dataset import Dataset_Clean_Slice_Overlap_Realtime, Dataset_Dist_Slice_Overlap_Realtime

save_dir = "./try_fx_and_listen/"
dataset = Dataset_Dist_Slice_Overlap_Realtime(audio_path = "/data4/guitarset/audio_hex-pickup_debleeded/train/", sr = 22050, slice_length = 4, slice_hop = 2)

for slice_idx in range(100):
    
    input_dry_audio = dataset[slice_idx]["audio_dry_GT"] # tensor size : dur*sr
    input_dist_audio = dataset[slice_idx]["audio_wet_GT"] # Chect the max sample point value of wet_audio. Is it re-scaled in 0~1 ?
    # input_dist_audio's gain and colour = random between 10~100
    print("max audio pressure of dry : ",  torch.max(input_dry_audio))
    print("max audio pressure of dist: ",  torch.max(input_dist_audio))






'''
print(("Is Gradient_Graph preserved in input? :", input_dry_audio_tensor.requires_grad))
print(("Is Gradient_Graph preserved in distorted audio? :", drived_audio_tensor.requires_grad))
print(("Is Gradient_Graph preserved in equalized audio? :", eq_audio_tensor.requires_grad))

import scipy.io.wavfile as wavfile
wavfile.write(save_dir + "original_audio.wav", 22050, input_dry_audio_tensor.detach().numpy())
wavfile.write(save_dir + "drived_audio.wav", 22050, drived_audio_tensor.detach().numpy())
'''


# To retain(preserve) gradient tracing (chain rule graph) throughout Overdrive Effect, See the link below
# https://arxiv.org/abs/2207.08759
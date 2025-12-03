"""
2020_01_15 - 2020_01_29
Harmonic Oscillator model for DDSP decoder.
TODO : 
    upsample + interpolation 
"""

import numpy as np
import torch
import torch.nn as nn


class HarmonicOscillator(nn.Module):
    def __init__(self, sr=22050, frame_length=64, attenuate_gain=0.02):
        super(HarmonicOscillator, self).__init__()
        self.sr = sr
        self.frame_length = frame_length
        self.attenuate_gain = attenuate_gain
        self.framerate_to_audiorate = nn.Upsample(
            scale_factor=self.frame_length, mode="linear", align_corners=False
        ) 
        # Convert "Frame-wise [Amplitude, Phase] of harmonics" -> "SamplePoint-wise [Amplitude, Phase] of harmonics"  
        # frame_length = sample points num per frame.

    def forward(self, z, string_idx=1):
        """
        Compute Addictive Synthesis of Sinusoids
        Argument: 
            z['six_f0'] : six fundamental frequency envelopes for each string
                - dimension (batch_num, frame_rate_time_samples)
            z['c'] : harmonic distribution of partials for each sample 
                - dimension (batch_num, partial_num, frame_rate_time_samples)
            z['a'] : loudness of entire sound for each sample
                - dimension (batch_num, frame_rate_time_samples)
        Returns:
            addictive_output : synthesized sinusoids for each sample 
                - dimension (batch_num, audio_rate_time_samples)
        """
        if string_idx == None : #Monophonc DDSP Mode
            fundamentals = z["one_f0"].squeeze(-1) # phase advancement(=pitch f0)
            framerate_c_bank = z["c"] # relative amlitude of harmonics 
            framerate_loudness = z["a"] # amplitude of fundamental wav at each frame.
        else : #Polyphonic DDSP Mode. Receive "list" of f0,a,c from Decoder
            string_idx = string_idx-1  # string idx given in GDDSP_total net is 1~6, So convert it into index 0~5 suitable for list [a_1st, ... a_6th]
            fundamentals = z["six_f0"][..., string_idx] # This is Phase_advancement per frame of Fundamental wav. w/ shape [frames, 1]
            framerate_c_bank = z["c"][string_idx] # This is relative-amplitude-seq of harmonics except Fundamental wav w/ shape [frames, config.n_harmonics]
            framerate_loudness = z["a"][string_idx] # "a" = Global Amplitude = Amplitude of Fundamental wav
            # Why z["c"], z["a"] pick string-wise tensor indexing at first dim, while z["six_f0"] pick string-wise tensor by indexing at last dim?
            # It's because z["c"], z["a"] is list composed of [(B, Frame)_1st, ... (B, Frame)_6th] =   6 X (Batch X Frame Tensor)
            # While z["six_f0"] is Tensor with shape Batch X Frame X 6
        num_osc = framerate_c_bank.shape[1] # Num of Harmonics except Fundamental Wav. shape[0] =batch , shape[1] =Num of Harmonics
        # Build a frequency envelopes of each partials from z['f0'] data
        partial_mult = (
            torch.linspace(1, num_osc, num_osc, dtype=torch.float32).unsqueeze(-1).to(fundamentals.device)
        )
        framerate_f0_bank = (
            fundamentals.unsqueeze(-1).expand(-1, -1, num_osc).transpose(1, 2) * partial_mult
        )

        # Antialias z['c']
        mask_filter = (framerate_f0_bank < self.sr / 2).float() # Nyquist Freq = sr/2. Ignore Harmonics representing freq over Nyquist Freq.
        # relative amplitude of each harmonic(=z["c"]) was produced using z_vector. z_vector was produced using MFCC
        # So, there cannot be "meaningful information" over Nyquist Freq in z["c"], 
        # as the MFCC, mel specrrogram, FFT cannot catch input audio components over Nyquist Freq.
        antialiased_framerate_c_bank = framerate_c_bank * mask_filter
        # Upsample frequency envelopes and build phase bank
        audiorate_f0_bank = self.framerate_to_audiorate(framerate_f0_bank)
        audiorate_phase_bank = torch.cumsum(audiorate_f0_bank / self.sr, 2)

        # Upsample amplitude envelopes
        audiorate_a_bank = self.framerate_to_audiorate(antialiased_framerate_c_bank)

        # Build harmonic sinusoid bank and sum to build harmonic sound
        sinusoid_bank = (
            torch.sin(2 * np.pi * audiorate_phase_bank) * audiorate_a_bank * self.attenuate_gain
        ) # attenuate_gain helps the Earliest Epoch's reconstructed audio scale to be close to Input audio's volume scale

        audiorate_loudness = self.framerate_to_audiorate(framerate_loudness.unsqueeze(0)).squeeze(0)
        addictive_output = torch.sum(sinusoid_bank, 1) * audiorate_loudness

        return addictive_output

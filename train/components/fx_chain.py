'''
This Script defines each class representing differentiable audio effect(distortion, delay, reverb) as well as the chain-class which integrates these effects in series.
Each Effect Class applies Power_tanh_distortion or FIR effect on Dry_Audio, 
so that GDDSP can reconstruct Input audio precisely by appylyinh effect on Dry_reconstructed_audio(=output of Harmonic Oscillators and Noise Generator)
Workflow of GDDSP:
input audio(4secs) -> fed to encoder -> encoder extracts f0, loud, timbre
f0, loud, timbre ->  fed to decoder -> decoder infers parameters for Harmonic Oscillator, Noise Generator, Distortion, Reverb(including Delay)
adjustable parameters : {"six_f0" : ~, "a": ~, "c": ~, "H" : ~, "dist" : ~, "reverb" : ~}
six_f0, a,c -> fed to Harmonic Oscillators -> Harmonic Oscialltor produces Sinusoids(*Fundamental wav + Harmonics of Input Audio)
H -> fed to Noise Generaotor -> Noise Generator produces filtered noise which represents aperiodic components of Input Audio

dist -> fed to W_H_Distortion -> W_H_Distortion applies Nonlineaer Distortion on 
'''


import torch
import torch.nn as nn
# All Fx modules must be differentiable.
# Because the backpropagation of Spectral Loss(Input Wet Audio, Recon Wtet Audio) must not be disconnected.

# To Train the Layers of Decoder which infers Control Knobs of Oscillator(amplitude and phase of sinusoids) 
# and to simultaneously train the Layers of Decoder which infers Control Konbs of FX Chain,
# The Backpropagation of Spectral Loss should go through Wet Reconstructed Audio -> Fx Chain -> Decoder Blocks(for FX knobs) and Dry Audio -> Decoder Blocks(for Harmonic Osillator knobs)
# So the Gradeint_Graph of all intermediate activation tensors in FX chain must be preserved.
# The Built-in Distortion/Delay library from sox or torchaudio disconnect the Gradient_Graph, so here I constructed novel differentiable FX classes.

### Codes of Distortion ###
''' Implementation of Diffenrentiable Distortion Blocks
# This is My(Yeongbeom) Own Implementation of Wiener-Hammerstein Distortion Model introduced in paper : https://www.aes.org/e-lib/browse.cfm?elib=21955
# This Distortion Model is chosen by the reasons below.
    # 1. it is a light-weighted model
    # 2. it effectively mimics the analog distortion pedals of real-world.
    # 3. It is differentiable so that it does not disconnect back-propagation of Spectral Loss through distorted recon_audio and clean recon_audio.
    # by 3, This Distortion Module can be used for self-supervised training of Neural Synthesizer, with loss between recon_audio and input_audio.
    # 4. This Model can expresses both linear transformation and non-linear transformation of audio waveform.
'''
class W_H_Distortion(nn.Module): # This is Differentiable Distortion Module (= convolution_EQ + power_tanh )
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        if self.config.distortion.use_pre_eq:
            self.pre_EQ = convolution_EQ(config = self.config, dict_key = "dist_pre_eq") #pre_eq layer
        self.power_tanh = power_tanh(config = self.config ) # power_tanh layer
        
        # Use Paramters below, when you do not use Decoder inferring knobs for Distortion
        # If all target audio data are guaranteed to have same dist type and magnitude,
        # We dont need Decoder which predicts different dist by different input.
        # Just Declare One Global Distortion knobs-set, and train this one set for all data.
        if self.config.input_dist_amount_random == False:
            self.EQ_FR_shared_on_all_data  = nn.Parameter(torch.ones(self.config.distortion.n_eq_band, requires_grad=True))
            self.ctanh_shared_on_all_data = nn.Parameter(torch.ones(self.config.distortion.n_tanh, requires_grad=True)) #Initial value is 1, but can be changed during train.
            self.pre_amp_shared_on_all_data = nn.Parameter(torch.tensor([3.0], dtype=torch.float32), requires_grad=True) #Initial value is 3, but can be changed during train
    def forward(self, audio, latent_dict_fed_from_Decoder):
        # x : Input Dry Recon audio
        # y : pre_EQ(x)
        # z : power_tanh(y) = c1 * tanh(y + offset) + c2 * tanh(y^2 + offset) + ... + cn * tanh(y^n + offset)
        # If All input data share one fixed distortion, dont use knobs inferred by decoder.
        if latent_dict_fed_from_Decoder == None:
            EQ_FR = self.EQ_FR_shared_on_all_data.to(audio.device)
            ctanh = self.ctanh_shared_on_all_data.to(audio.device)
            pre_amp_before_tanh = self.pre_amp_shared_on_all_data.to(audio.device)
            # W-H Distortion make the one Distortion Tensor to Fixed_Input_Distortion Setting shared in total Dataset
        else:
            EQ_FR = latent_dict_fed_from_Decoder["dist_pre_eq"]
            ctanh = latent_dict_fed_from_Decoder["dist_ctanh"]
            pre_amp_before_tanh = latent_dict_fed_from_Decoder["pre_amp_before_tanh"]
            # W-H Distortion can realize various Distortion Amount differ by audio, with Knobs predicted by Decoder.
        x = audio
        if self.config.distortion.use_pre_eq :
            y = self.pre_EQ(x, EQ_FR)
            z = self.power_tanh(y, ctanh, pre_amp_before_tanh = pre_amp_before_tanh)
        else:
            z = self.power_tanh(x, ctanh, pre_amp_before_tanh = pre_amp_before_tanh)
        return z

# class convolution_EQ and class power_tanh = Sub-layers of W_H_Distortion #
class convolution_EQ(nn.Module):
    '''
    This is EQ class included in W_H Distortion Network
    EQ applies Room Response Effect(Various Attenuation by freq) on Dry Signal by Convolution in Frequency-Time Space.
    '''
    def __init__(self, config=None, dict_key = "dist_pre_eq", attenuate_gain = 1): 
        # dict_key = "dist_pre_eq" or "dist_post_eq"
        # Use "pre" to implement EQ layer before Non-linear tanh Distortion
        # Use "post" to implement EQ layer following Non-linear tanh Distortion
        super().__init__()
        self.config = config
        self.dict_key = dict_key
        
        if config == None:
            self.frame_length = 88
            self.n_fft = 441
            self.n_band = 65
        else:
            self.frame_length = int(self.config.sample_rate * self.config.frame_resolution)  #set frame_length(=windowing hop) as same as frame_length of z_enc and Loud_enc, to unify num of temporal frames.
            self.n_fft = self.config.distortion.n_fft
            self.n_band = self.config.distortion.n_eq_band

        # this n_fft is set as same as n_fft of loudness encoder, to unify the temporal size between latent vectors.
        # n_fft = num of freq bins in one frame = full temporal length of one framae
        # frame_length = effective temporal length of one frame without overlapping = hop length of windowing
        self.attenuate_gain = attenuate_gain
        self.smoothing_window = nn.Parameter(torch.hann_window(self.n_fft, dtype = torch.float32), requires_grad = False)

    def forward(self, audio, EQ_FR):
        
        '''
        Compute linear-phase LTI-FVR filter banks in batch from network output,
        and create time-varying filtered noise by overlap-add method.
        
        Input Arguments:
        
        1. Dry_audio w/ shape: (batch, timesteps)
        Dry_audio signal is fed from audio_recon dictionary made by autoencoder.py
        2. Frequency-Domain Transfer Function(FR) of EQ in Freq-time Domain
        This FR is fed from the Decoder
        This FR represents EQ's Timbral Effect, and will be convolved with Dry_audio produced by Harmonic Oscillator, in freq-time Domain.
           
        Output: 
        equlized windowed audio w/ shape (Batch, Sample points for total duration)
        
        Notation:
        Variable named with Capital Letters : values in Fourier spectrum space. shape : (time frames, coefficient of freq bins)
        variable named with small letters : values in Waveform space. shape : (time frames, sound pressure)
        '''        
        if isinstance(audio, dict):
            audio = audio["audio_dry"]
        pad_length = self.n_fft - self.frame_length
        paded_input_audio = nn.functional.pad(audio, (pad_length//2, pad_length//2))
        sliced_audio = paded_input_audio.unfold(1, self.n_fft, self.frame_length)
        # Sample Points per Window including Overlapping = n_fft
        # Points Advancement(=hop = stride) of Window = frame_length
        # Num of frames = [(audio_sample_point_length - n_fft) // frame_length] +1
        # To Match Num of Frames of sliced_audio with Num of Frames of EQ_FR(=audio_sample_point_length//frame_length), we need padding.
        # (1, self.n_fft, self.frame_length) indicates (slicing dimension, size, step) of slicing
        # dimension : 1 -> Slicing targets the time_duration axis. (0th dimension = batch_idx dimension)
        
        sliced_windowed_audio = sliced_audio * self.smoothing_window.to(sliced_audio.device)
        sliced_windowed_audio = sliced_windowed_audio 
        batch_num, frame_num = sliced_windowed_audio.shape[0],  sliced_windowed_audio.shape[1]
        
        if isinstance(EQ_FR, dict): 
            #EQ_FR is output of decoder
            EQ_FR = EQ_FR[self.dict_key] 
            #dict_key = "dist_pre_eq"
        if EQ_FR.dim() == 3 :
            print("pre-EQ Tranfer function included in Distortion has 3 dimensions(Batch, Frames, Freq bins)")
            print("Time-Varying pre-EQ bins in each frame will be convolved with corresponding frame of dry audio repectively")
        elif EQ_FR.dim() ==2 : # EQ_FR produced per audio in batch
            print("pre-EQ Tranfer function included in Distortion has 2 dimensions(Batch, Freq bins)")
            print(f"There are {self.config.distortion.n_eq_band} pre-eq Freq bins per audio, not per frame")
            EQ_FR = EQ_FR.unsqueeze(1).repeat(1, frame_num, 1) # Make Frame-axis manually, and insert same Freq-bin-set into all frames(but still differ by audio)
        elif EQ_FR.dim() ==1 : # One set of EQ_FR produced in W_H_Distortion's nn.parameter 
            print("pre-EQ Tranfer function included in Distortion has 1 dimensions(Freq bins)")
            print(f"One Common Traianble Distortion Instance will be applied on all dry recon audio in dataset")
            EQ_FR = EQ_FR.unsqueeze(0).unsqueeze(0).repeat(batch_num, frame_num, 1)
            # audio.shape[0] = Num of audio in batch 
            # audio.shape[1] = Num of Frames in audio. Differ by Seconds-length of audio.
        else:
            raise ValueError("Tensor of freq bins for pre-EQ in Distortion has wrong dimension.")

        # EQ_FR is produced by GRU and FC layers in Decoder. The Structure of these layers are introduced in Original DDSP Paper
        # eq_ir : waveform corresponding to EQ_FR = inversed_fft(EQ_FR)
        filter_num_bin = EQ_FR.shape[-1]
        ZERO_PHASE_FR_BANK = EQ_FR
        ZERO_PHASE_FR_BANK = ZERO_PHASE_FR_BANK.view(-1, filter_num_bin )
        # Flatten the batch_axis and frame_axis.
        # Originally The value of filter_coeff is recorded in form [[audio1_frame1, audio1_frame2, ... audio1_frameN], [audio2_frame1, audio2_frame2, ... audio2_frameN]]
        # for frame-wise convolution, boundary line between neighboring audios is not necessary.
        # So convert the form of FR_BANK into [audio1_frame1, audio1_frame2, ... audio1_frameN, audio2_frame1, audio2_frame2, ... audio2_frameN]
        # Get the impulse response representing EQ's Timbral Effect in Waveform space. (sound pressure - time space)
        zero_phase_ir_bank = torch.fft.irfft(ZERO_PHASE_FR_BANK, n= filter_num_bin * 2 - 1 )
        # zero-padding will be conducted on waveform ir, instead of FR in freq bin-time space.
        # Goal of zero-padding is synchronizing the temporal length of each frame of dry_audio and each frame of eq_ir.
    
        # Convert the zero-phase IR into causal-phase IR following the method of DDSP & Hann-window it.
        linear_phase_ir_bank = zero_phase_ir_bank.roll(filter_num_bin  - 1, 1) 
        # roll function converts eq_ir into symmetric form(=zero phase form), which is suiatable for hann_windowing
        filter_window = torch.hann_window(filter_num_bin  * 2 - 1, dtype = torch.float32).to(sliced_audio.device)
        # Create Hann Window whose temporal length mathces EQ_Impulse Response.
        # There are "unexpected freq components" in sliced signal. 
        # These wrong components are created during fft of Filter_coeffieicent, as the sliced signal does not include all periodicities of original infinite signal. 
        # Hann Window reduces these unexptected components(=Edge Effect)
        windowed_linear_phase_ir_bank = linear_phase_ir_bank * filter_window.view(1, -1) #Hann-Windowing eq_ir
        zero_paded_windowed_linear_phase_ir_bank = nn.functional.pad(windowed_linear_phase_ir_bank, (0, self.n_fft - 1))
        
        # Get the linear-phase Frequency-Domain Transfer Function(=Freq-Domain EQ Effect) by applying fft on linear-phase IR.
        ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK = torch.fft.rfft(zero_paded_windowed_linear_phase_ir_bank)
        
        # Convert the shape of input dry_audio(=reconstructed by oscillators and Noise Generator) to match the shape of IR.
        sliced_windowed_audio  = sliced_windowed_audio.view(-1, self.n_fft)
        # Flatten the axis of batch and frame_idx to simplify the convolution.
        zero_paded_audio = nn.functional.pad(sliced_windowed_audio , (0, filter_num_bin * 2 - 2))
        ZERO_PADED_AUDIO = torch.fft.rfft(zero_paded_audio)
        
        ''' Regardless of Original length of EQ_FR and eq_ir, after Zero padding, the shape_difference between eq_ir and dry_audio will be resolved.'''
        ''' So convolution in last axis will not evoke the shape_error'''
        # Shape Change Process #
        # sliced_windowed_audio.view(-1, n_fft) : shape ( Batch * num_frames, n_fft )
        # ZERO_PHASE_FR_BANK.view(-1, filter_coeff_length) : shape (Batch * num_frames, filter_num_bin)
        # zero_phase_ir_bank : shape(Batch * num_frames, 2*filter_num_bin -1 )
        # After zero padding, 
        # audio + zero_pad(0, 2*filter_coeff_length-2)  : shape(Batch * num_frames,  n_fft + 2*filter_num_bin-2)  <- Let this X
        # ir_bank + zero_pad(0, frame_length -1 ) : shape(Batch * num_frames,  2*filter_num_bin+ n_fft -2) <- Let this Y
        # You can check that the final axis length of X and Y always become same.
        
        # Make the Empty Tensor which will receive the calculation result of Convolution. 
        EQUALIZED_AUDIO_real = torch.zeros_like(ZERO_PADED_AUDIO).to(sliced_audio.device)
        EQUALIZED_AUDIO_imag = torch.zeros_like(ZERO_PADED_AUDIO).to(sliced_audio.device)
        
        # Convolution between freq bins of AUDIO and PHASE_FILTER_BANK(=EQ_FR)
        # a+bi : ZERO_PADED_AUDIO
        # c+di : LINEAR_PHASE_FILTER_BANK
        # i^th freq bin represent center frequency value(Hz) of  i * sr/n_fft 
        # default interval between neighboring freq bins = sr/n_fft = 22050/441 = 50Hz
        # (a+bi)*(c+di) = (ac-bd) + (ad+bc)i = Result of Convolution on dry_audio and EQ_FR
        # ac-bd
        EQUALIZED_AUDIO_real = ZERO_PADED_AUDIO[:, :].real * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :].real \
            - ZERO_PADED_AUDIO[:, :].imag * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :].imag
        # ab+bc
        EQUALIZED_AUDIO_imag = ZERO_PADED_AUDIO[:, :].real * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :].imag \
            + ZERO_PADED_AUDIO[:, :].imag * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :].real
        # Collect (ac-bd) and (ad+bc)i in one tensor.
        EQUALIZED_AUDIO = torch.complex(EQUALIZED_AUDIO_real, EQUALIZED_AUDIO_imag).to(sliced_audio.device)
        equalized_audio = torch.fft.irfft(EQUALIZED_AUDIO).view(batch_num, frame_num, -1) * self.attenuate_gain 
        # restore the shape of (batch, frame, smample_point) from flatten shape (batch*frame, sample_point)    
        #EQUALIZED_AUDIO : The output of Convolution(dry_audio_freq_spectrum , EQ_filter_freq_spectrum)
        #equalized_audio : The Convolution(dry_audio_wav, EQ_filter_wav), acquired by inversed_rfft on FILTERED_AUDIO
        
        # Reverse Process of Windowing
        # Recover the shape of audible audio (Batch, sr*duration) from windowed shape
        # (Batch, Frames, sample points per Overlapping-Frame) -> (Batch, Frames * sample points per Frame)
        overlap_add_filter = torch.eye(equalized_audio.shape[-1], requires_grad = False).unsqueeze(1).to(sliced_audio.device)
        output_signal = nn.functional.conv_transpose1d(equalized_audio.transpose(1, 2), overlap_add_filter, stride = self.frame_length, padding = 0).squeeze(1)
        length_original_audio_before_pre_eq = int(self.config.sample_rate * self.config.slice_length)
        output_signal = output_signal[..., : length_original_audio_before_pre_eq] 
        return output_signal

# class convolution_EQ and class power_tanh = Sub-layers of W_H_Distortion #
class power_tanh(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config == None:
            self.frame_length = 88
            self.n_fft = 441
        else:
            self.frame_length = int(self.config.sample_rate * self.config.frame_resolution)  #set frame_length(=windowing hop) as same as frame_length of z_enc and Loud_enc, to unify num of temporal frames.
            self.n_fft = self.config.distortion.n_fft

    def forward(self, y, c_tanh, pre_amp_before_tanh = None):
        # x : Input Dry Recon audio before pre-EQ
        # y : pre_EQ(x) = equalized waveform w/ shape (batch, frame_idx, sample points per frame)
        # z : power_tanh(y) = c0*ay + c1*tanh(ay) + c2*tanh((ay)^2) + ... + cn-1 * tanh((ay)^(n-1)),  a=pre-amp multiplication ratio
        # c_tanh : tensor indicates each coefficient(mixing ratio) of tanh(x), tanh(x^2), tanh(x^3) ... tanh(x^n_tanh)
        n_tanh = self.config.distortion.n_tanh
        
        ### Pre-Amplifying before Applying Tanh Formula ###
        if not pre_amp_before_tanh == None: #pre-EQ before this tanh process only allow attenuation. So Add Pre-Amplifying process
            print("Now Pre-Amplifying pre-equalized audio before applying tanh-distortion")
            sample_points_per_audio = y.shape[1]
            if pre_amp_before_tanh.shape[0] == y.shape[0] : # In case that pre-amp has batch axis
                pre_amp_before_tanh = pre_amp_before_tanh.unsqueeze(1).repeat(1, sample_points_per_audio)
                y = y * pre_amp_before_tanh #Before Clipped(Compressed) by Tanh, Give Pre-Ampliying Volume up to mimic Real-world distortion process.
            else: 
                pre_amp_before_tanh = pre_amp_before_tanh.unsqueeze(1).repeat(y.shape[0], sample_points_per_audio)
                y = y * pre_amp_before_tanh
        ### Pre-Amp Finished ###
        
        # If c_tanh does not have batch_axis or frame_axis, duplicate the c_tanh value to batch_axis and frame axis,
        # to Enable Element-wise multiplication Between 3D audio(=y) and 3D c_tanh
        if c_tanh.dim() == 3 :
            print("Time-Varying c_tanh in each frame will be convolved with corresponding frame of pre_equlized_audio")
            y = y[..., : c_tanh.shape[1] *self.frame_length ] # Cut y's time axis to be suitable for producing non-overlapping frames.
            y = y.reshape(c_tanh.shape[0], c_tanh.shape[1], -1) # Slicing y with non-overlapping frames, to apply frame-wise adjustable power_tanh effect on y.
        elif c_tanh.dim() == 2 :
            # c_tanh differs by audio(differ alonh batch-axis), not by frame.
            print(f"There are {self.config.distortion.n_tanh} Global tanh terms per audio, not per frame")
            sample_points_per_audio = y.shape[1]
            c_tanh = c_tanh.unsqueeze(1).repeat(1, sample_points_per_audio, 1)
        elif c_tanh.dim() == 1 :
            # One Common Global c_tanh will be applied on all audio and all frames.
            print(f"One Trainable Tanh Series will be applied on All Dry Recon audio in Dataset.(Distortion Shared on audios)")
            sample_points_per_audio = y.shape[1]
            c_tanh = c_tanh.unsqueeze(0).unsqueeze(0).repeat(y.shape[0], sample_points_per_audio, 1) 
        else:
            raise ValueError("ctanh in Distortion has wrong dimension.")
        
        # z = torch.zeros_like(y).to(y.device)
        if self.config.distortion.formula == "power":
            tanh_terms = torch.zeros(y.shape[0], y.shape[1], n_tanh).to(y.device)
            for i in range(1, n_tanh): #Iterating for c_tanh_1 ~ c_tanh_n-1
                tanh_terms[: ,: ,i] = c_tanh[: ,: ,i] * torch.tanh(y[:,:]**i) # i+1 in torch.tanh is chosen to start power series from y^1, not y^0
            z = y * c_tanh[:, :, 0] + torch.sum(tanh_terms, dim=2, keepdim=False)  #z = c0y + c1tanh(y^1) + ... cn-1(y^(n-1))
        elif self.config.distortion.formula == "sum" :
            tanh_terms = torch.zeros(y.shape[0], y.shape[1], n_tanh).to(y.device)
            for i in range(1, n_tanh):
                tanh_terms[: ,: ,i] = c_tanh[: ,: ,i] * torch.tanh(y[:,:]*i)
            z = y * c_tanh[:, :, 0] + torch.sum(tanh_terms, dim=2, keepdim=False)
            # z = c0y + c1tanh(y*1) + ... cn-1(y * (n-1))
        elif self.config.distortion.formula == "sum_w_offset" : 
            tanh_terms = torch.zeros(y.shape[0], y.shape[1], n_tanh - 1).to(y.device)
            DC_due_to_offset_at_zero_sound_pressure = torch.zeros(y.shape[0], y.shape[1], n_tanh -1).to(y.device)
            offset = c_tanh[:, :, n_tanh-1]
            for i in range(1, n_tanh - 1):
                tanh_terms[: ,: ,i] = c_tanh[: ,: ,i] * torch.tanh(y[:,:]*i + offset*i)
                DC_due_to_offset_at_zero_sound_pressure[:, :, i] = c_tanh[:, :, i] * torch.tanh(offset*i)
            z = y * c_tanh[:, :, 0] + torch.sum(tanh_terms, dim=2, keepdim=False) - torch.sum(DC_due_to_offset_at_zero_sound_pressure, dim=2, keepdim=False)
            # offset = cn-1
            # z = c0y + c1tanh(y*1 + offset*1) + ...+ cn-2(y*n-2  + offset*n-2) - bias_due_to_offset
        else:
            raise ValueError("Invalid distortion.formula option was given by configuration.yaml")
        return z

### Codes of Reverb ###
''' Code of Reverb was imported from torch_implemented DDSP (github :https://github.com/sweetcocoa/ddsp-pytorch )'''
import numpy as np
import torch
import torch.nn as nn
# One Fixed Global Reverb, FIR and Decay Knowledge will be shared on all data fed into model.
class TrainableFIRReverb(nn.Module): 
    # FIR = Finite Impulse Response
    # This Reverb Class does not conduct One-Shot Estimation.
    # Instead, This Reverb Update Internal nn.Parameters(wet ratio and decay rate) with multiple audio data with same input Reverb Setting.
    def __init__(self, config):

        super(TrainableFIRReverb, self).__init__()
        # default reverb length is set to 3sec.
        self.config = config
        self.reverb_length = self.config.sample_rate * self.config.room_acoustic.reverb_sec #default 22050 * 1sec

        ### Knobs of Reverb : FIR shape , drywet ratio , and decay of FIR Envelope ###
        # but equal-loudness crossfade between identity impulse and fir reverb impulse is not implemented yet.
        
        # impulse response of reverb.
        self.fir = nn.Parameter(
            torch.rand(1, self.reverb_length, dtype=torch.float32) * 2 - 1,
            requires_grad=True)
        # Initialized drywet to around 26%.
        self.drywet = nn.Parameter(
            torch.tensor([-1.0], dtype=torch.float32), requires_grad=True)
        # Initialized decay to 5, to make t60 = 1sec.
        self.decay = nn.Parameter(
            torch.tensor([3.0], dtype=torch.float32), requires_grad=True)
        ### These Parameters are not inferred by Decoder, but Trained with Backpropagation directed along Wet Recon Loss -> Reverb ###
        # Reverb FIR can be updated during Training, but cannot differ by Input Audio. #
        # Because Reverb FIR are not extracted from Input audio, and just penalized by all Recon audio in dataset

    def forward(self, z):
        
        """
        Compute FIR Reverb
        Input:
            : batch of time-domain signals = z
            * Do not receive output latent of Decoder. Reverb Does not need Knobs differ by Input Audio.
        Output:
            output_signal : batch of reverberated signals
        Expression:
            Instance with name of Capital Letters : Tensor on Spectrogram Space ( Freq bin coefficient - Time space)
            Instance with name of small letters : Tensor on waveform space ( Sound Pressure - Time space )
        """
        # Send batch of input signals in time domain to frequency domain.
        # Appropriate zero padding is required for linear convolution.
        input_signal = z # z = audio_recon["audio_dry"] or audio_reocn["audio_dist"]
        # z w/ shape [ Batch, SamplePoints ]
        zero_pad_input_signal = nn.functional.pad(input_signal, (0, self.fir.shape[-1] - 1))
        INPUT_SIGNAL = torch.fft.rfft(zero_pad_input_signal)

        # Build decaying impulse response and send it to frequency domain.
        # Appropriate zero padding is required for linear convolution.
        # Dry-wet mixing is done by mixing impulse response, rather than mixing at the final stage.
        decay_envelope = torch.exp(
            -(torch.exp(self.decay) + 2)
            * torch.linspace(0, 1, self.reverb_length, dtype=torch.float32).to(input_signal.device)
        )
        # fir * exp(decay) = Decreainsg IR by time = decay_fir
        decay_fir = self.fir * decay_envelope

        ir_identity = torch.zeros(1, decay_fir.shape[-1]).to(input_signal.device)
        ir_identity[:, 0] = 1
        # ir_identity  = [1, 0, 0, 0, 0, ..... 0 ] w/ shape [1, FIR_n_sample_points]
        # role of ir_identity => Returns Dry Signal After <Dry convolve IR>

        final_fir = (
            torch.sigmoid(self.drywet) * decay_fir + (1 - torch.sigmoid(self.drywet)) * ir_identity
        ) # Applys dry-wet ratio.
        # torch.sigmoid(self.drywet) = wet ratio
        # decay_fir* dry_signal = wet signal
        # 1 - torch.sigmoid(self.drywet) = 1- wet ratio = dry ratio
        # dry_signal * ir_identity = dry signal
        
        # final_fir , decay_fir, ir_identiy each has shape [1, sr*reverb_sec]. 
        # They dont need Batch Num(>1) as Same fir will be applied on all audio in Batch.
        zero_pad_final_fir = nn.functional.pad(final_fir, (0, input_signal.shape[-1] - 1))
        FIR = torch.fft.rfft(zero_pad_final_fir)
        
        ### Convolution on Spectrogram Space is Faster then Convolution on Waveform Space ###
        # (a+bi) = INPUT_SIGNAL on freq space (complex signal, regards both amp and phase)
        # (c+di) = FIR of Reverb on freq space
        # INPUT_SIGNAL convolve FIR = (a+bi)(c+di) = (ac-bd)  + (ad+bc)i
        OUTPUT_SIGNAL_real = INPUT_SIGNAL[:, :].real * FIR[:, :].real \
            - INPUT_SIGNAL[:, :].imag * FIR[:, :].imag # ac-bd
        OUTPUT_SIGNAL_imag = INPUT_SIGNAL[:, :].real * FIR[:, :].imag \
            + INPUT_SIGNAL[:, :].imag * FIR[:, :].real # (ab+bc)i
        OUTPUT_SIGNAL = torch.complex(OUTPUT_SIGNAL_real, OUTPUT_SIGNAL_imag).to(input_signal.device)
        ### Convolution finished ###
        
        output_signal = torch.fft.irfft(OUTPUT_SIGNAL) # Get Reverberated Signal on Waveformspace
        #output_signal length = sr*input_signal_sec + sr*reverb_sec -1 -1 =110248
        
        if output_signal.shape[-1] > input_signal.shape[-1] : 
            # Cut Tail of (Last Sample Point of input * Reverb FIR)
            output_signal_trimmed =  output_signal[..., :input_signal.shape[-1]]
        
        return output_signal_trimmed


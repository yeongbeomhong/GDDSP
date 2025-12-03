"""
2020_01_20 - 2020_01_29
Simple trainable filtered noise model for DDSP decoder.
"""

import torch
import torch.nn as nn


class FilteredNoise(nn.Module):
    def __init__(self, frame_length = 64, attenuate_gain = 1e-2):
        super(FilteredNoise, self).__init__()
        
        self.frame_length = frame_length
        self.attenuate_gain = attenuate_gain
        
    def forward(self, z):
        """
        Compute linear-phase LTI-FVR (time-variant in terms of frame by frame) filter banks in batch from network output,
        and create time-varying filtered noise by overlap-add method.
        
        Argument:
            z['H'] : filter coefficient bank for each batch, which will be used for constructing linear-phase filter.
                - dimension : (batch_num, frame_num, filter_coeff_length)
        Variable named with Capital Letters : values in Fourier spectrum space. shape : (time frames, coefficient of freq bins)
        variable named with samll letters : values in Waveform space. shape : (time frames, sound pressure)
        
        """
        
        batch_num, frame_num, filter_coeff_length = z['H'].shape
        INPUT_FILTER_COEFFICIENT = z['H']
        filter_window = nn.Parameter(torch.hann_window(filter_coeff_length * 2 - 1, dtype = torch.float32), requires_grad = False).to(INPUT_FILTER_COEFFICIENT.device)
        
        # Desired linear-phase filter can be obtained by time-shifting a zero-phase form (especially to a causal form to be real-time),
        # which has zero imaginery part in the frequency response. 
        # Therefore, first we create a zero-phase filter in frequency domain.
        # Then, IDFT & make it causal form. length IDFT-ed signal size can be both even or odd, 
        # but we choose odd number such that a single sample can represent the center of impulse response.
        ZERO_PHASE_FR_BANK = INPUT_FILTER_COEFFICIENT        
        ZERO_PHASE_FR_BANK = ZERO_PHASE_FR_BANK.view(-1, filter_coeff_length)
        # Flatten the batch_axis and frame_axis.
        # Originally The value of filter_coeff is recored in form [[audio1_frame1, audio1_frame2, ... audio1_frameN], [audio2_frame1, audio2_frame2, ... audio2_frameN]]
        # for frame-wise convolution, boundary line between two neighboring audio in batch axis in not necessary.
        # So convert the form of FR_BANK into [audio1_frame1, audio1_frame2, ... audio1_frameN, audio2_frame1, audio2_frame2, ... audio2_frameN]
        
        zero_phase_ir_bank = torch.fft.irfft(ZERO_PHASE_FR_BANK, n= filter_coeff_length * 2 - 1 )
        # argument "n" of torch.fft.irfft = argument "signal_sizes" of torch.irfft
        #irfft = inversed_rfft = inversed_fft of real-valued(실수값) input.
        
        # Make linear phase causal impulse response & Hann-window it.
        # Then zero pad + DFT for linear convolution.
        # In this Process, the ir(impulse response) of Filter becomes same as "zero paded noise" or "zero paded audio signal.""
        linear_phase_ir_bank = zero_phase_ir_bank.roll(filter_coeff_length - 1, 1)
        windowed_linear_phase_ir_bank = linear_phase_ir_bank * filter_window.view(1, -1)
        zero_paded_windowed_linear_phase_ir_bank = nn.functional.pad(windowed_linear_phase_ir_bank, (0, self.frame_length - 1))
        ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK = torch.fft.rfft(zero_paded_windowed_linear_phase_ir_bank)
        
        # Generate white noise & zero pad & DFT for linear convolution.
        noise = torch.rand(batch_num, frame_num, self.frame_length, dtype = torch.float32).view(-1, self.frame_length).to(INPUT_FILTER_COEFFICIENT.device) * 2 - 1
        # noise : white noise which is signal source to reconstruct a-periodic components of input audio.
        # frame_length : hop length per one frame. (= Step_length of windowing)
        # frame_num * frame_length = (approximately) original timesteps of one input guitar recording = num of original sample points of one input guitar recording
        # ex> Let  88200 = 22500 sr * 4 sec = original sample points
        # then, the original audio will be padded with left length 2*frame_length, and right length 2*frame_length
        # -> padded audio waveform length = 88200 + 2*88 + 2*88 = 88552
        
        zero_paded_noise = nn.functional.pad(noise, (0, filter_coeff_length * 2 - 2))
        ZERO_PADED_NOISE = torch.fft.rfft(zero_paded_noise)
        
        # How the num of sample_points per frame of "noise" and "ir_filter" match each other?
        # noise.view(-1, frame_length) : shape ( Batch * num_frames, frame_length )
        # ZERO_PHASE_FR_BANK.view(-1, filter_coeff_length) : shape (Batch * num_frames, filter_coeff_length )
        # zero_phase_ir_bank : shape(Batch * num_frames, 2*filter_coeff_length -1 )
        # After zero padding, 
        # noise + zero_pad(0, 2*filter_coeff_length-2)  : shape(Batch * num_frames,  frame_length + 2*filter_coeff_length-2)  <- Let this X
        # ir_bank + zero_pad(0, frame_length -1 ) : shape(Batch * num_frames,  2*filter_coeff_length + frame_length -2) <- Let this Y
        # You can check that the final axis length of X and Y are same.

        FILTERED_NOISE_real = torch.zeros_like(ZERO_PADED_NOISE).to(INPUT_FILTER_COEFFICIENT.device)
        FILTERED_NOISE_imag = torch.zeros_like(ZERO_PADED_NOISE).to(INPUT_FILTER_COEFFICIENT.device)
        
        # Convolve & IDFT to make filtered noise for each frame, noise band, and batch.
        # a+bi = ZERE_PADED_WHITE_NOISE
        # c+di = LINEAR_PHASE_FILTER_BANK
        # (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
        # ac-bd
        FILTERED_NOISE_real = ZERO_PADED_NOISE[:, :].real * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :].real \
            - ZERO_PADED_NOISE[:, :].imag * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :].imag
            
        # ab+bc
        FILTERED_NOISE_imag = ZERO_PADED_NOISE[:, :].real * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :].imag \
            + ZERO_PADED_NOISE[:, :].imag * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :].real
        
        FILTERED_NOISE = torch.complex(FILTERED_NOISE_real, FILTERED_NOISE_imag).to(INPUT_FILTER_COEFFICIENT.device)
        filtered_noise = torch.fft.irfft(FILTERED_NOISE).view(batch_num, frame_num, -1) * self.attenuate_gain
        #shape of filted_noise = shape of zero_paded_noise
        #FILTERD_NOISE : The output of Convoluition(white_noise_spectrum , noise_filter_spectrum)
        #filtered_noise : The Convolution(white_noise_wav, noise_filter_wav), acquired by inversed_rfft on FILTERD_NOISE
        
        # Overlap-add to build time-varying filtered noise.
        # This Procedure collects windowed value of filtered noise into one full-long Time-axis.
        # Recovers the shape noise (Batch, sr*duration) from shape (Batch, frame_num, frame_length), to match the shape of periodic sinusoids produced by Oscillators.
        # Recovering is not simply same as concatenating frames. As there are overlapping temporal region in neighboriing frames.
        # So follow the steps below.
        overlap_add_filter = torch.eye(filtered_noise.shape[-1], requires_grad = False).unsqueeze(1).to(INPUT_FILTER_COEFFICIENT.device)
        output_signal = nn.functional.conv_transpose1d(filtered_noise.transpose(1, 2), 
                                                       overlap_add_filter, 
                                                       stride = self.frame_length, 
                                                       padding = 0).squeeze(1)
        
        
        return output_signal

#!/usr/bin/env python
# encoding: utf-8
#
# Copyright 2022 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import enum
import json
import os
import pathlib
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tensorflow import Tensor, signal, keras, saved_model
import numpy as np
import torch
import librosa
import pretty_midi

from ..basic_pitch.constants import (
    AUDIO_SAMPLE_RATE,
    AUDIO_N_SAMPLES,
    ANNOTATIONS_FPS,
    FFT_HOP,
)
from ..basic_pitch import ICASSP_2022_MODEL_PATH, note_creation as infer
from ..basic_pitch.commandline_printing import (
    generating_file_message,
    no_tf_warnings,
    file_saved_confirmation,
    failed_to_save,
)
def window_audio_file(audio_original, hop_size):
    """
    Pad appropriately an audio file, and return as
    windowed signal, with window length = AUDIO_N_SAMPLES
    Returns:
        audio_windowed: tensor with shape (n_windows, AUDIO_N_SAMPLES, 1)
            audio windowed into fixed length chunks
        window_times: list of {'start':.., 'end':...} objects (times in seconds)

    """
    from tensorflow import expand_dims, reshape
    audio_original = audio_original
    if audio_original.ndim == 2: #input audio feeded with batch axis. [B, 88200] Need to flatten Batch, Frame Axis
        audio_windowed = expand_dims(signal.frame(audio_original, AUDIO_N_SAMPLES, hop_size, pad_end=True, pad_value=0),axis=-1,)
        audio_windowed = reshape(audio_windowed, [audio_windowed.shape[0]*audio_windowed.shape[1], audio_windowed.shape[2], audio_windowed.shape[3]])
    else:# audio_original.ndim == 1 [88200] input audio feeded without batch axis
        audio_windowed = expand_dims(
            signal.frame(audio_original, AUDIO_N_SAMPLES, hop_size, pad_end=True, pad_value=0),
            axis=-1,) # No need to flatten Batch, Frame Axis
    '''
    window_times = [
        {
            "start": t_start,
            "end": t_start + (AUDIO_N_SAMPLES / AUDIO_SAMPLE_RATE),
        }
        for t_start in np.arange(audio_windowed.shape[0]) * hop_size / AUDIO_SAMPLE_RATE
    ]
    return audio_windowed, window_times
    '''
    return audio_windowed
    
# I Edited the method below
# I do not need the Process of Loading .wav from path, and dont need to convert wav into tensor 
# I will just feed the preprocessed batch(=sliced Tensor. default dur = 4sec) into basic-pitch inference.
def get_audio_input(audio_input: Tensor, overlap_len: int, hop_size: int) -> Tuple[Tensor, List[Dict[str, int]], int]:
    """
    Read wave file (as mono), pad appropriately, and return as
    windowed signal, with window length = AUDIO_N_SAMPLES

    Returns:
        audio_windowed: tensor with shape (n_windows, AUDIO_N_SAMPLES, 1)
            audio windowed into fixed length chunks
        window_times: list of {'start':.., 'end':...} objects (times in seconds)
        audio_original_length: int
            length of original audio file, in frames, BEFORE padding.

    """
    assert overlap_len % 2 == 0, "overlap_length must be even, got {}".format(overlap_len)
    if isinstance(audio_input, torch.Tensor):
        audio_input = audio_input.numpy()
    # The basic-pitch model's inference classes handle numpy_type_Data. so convert the type of audio into numpy
    if audio_input.ndim == 2: #audio is feeded with batch axis
        audio_original = audio_input
    else :
        audio_input = np.expand_dims(audio_input, axis=0) # Manually add batch axis with batch_size =1
        audio_original = audio_input
    original_length = audio_original.shape[-1] # n_sample points of one audio in batch
    input_batch_size = audio_original.shape[0]
    audio_original = np.concatenate((np.zeros((audio_original.shape[0], int(overlap_len / 2)), dtype=np.float32), audio_original), axis=1 )
    audio_windowed = window_audio_file(audio_original, hop_size)
    return audio_windowed, original_length, input_batch_size


def trim_output(output: Tensor, audio_original_length: int, n_overlapping_frames: int, input_batch_size :int) -> np.array:
    # Audio-wise Trimming (YB's code replacing unwrap_output)
    """Unwrap batched model predictions to a single matrix.
    Args:
        output: array (n_batches, n_times_short, n_freqs)
        audio_original_length: length of original audio signal (in samples)
        n_overlapping_frames: number of overlapping frames in the output
        audio_original_length = sample points of One input audio in minibatch
    Returns:
        array (n_batches, n_framees_trimmed, n_freqs)
    """
    raw_output = output.numpy()
    if len(raw_output.shape) != 3:
        raise ValueError("raw_output of Basic_Pitch Lightweight pitch Encoder has wrong shape")
    n_olap = int(0.5 * n_overlapping_frames)
    if n_olap > 0: #In the case That Neighboring frames share n_overlapping_Frames points.
        raw_output = raw_output[:, n_olap:-n_olap, :]  # remove half of the overlapping frames from beginning and end
    raw_output_batched = raw_output.reshape(input_batch_size, -1, 264) # -1 = frames num inferred per audio
    n_frames_per_one_audio = int(np.floor(audio_original_length * (ANNOTATIONS_FPS / AUDIO_SAMPLE_RATE)))
    output_batched_trimmed = raw_output_batched[:,  :n_frames_per_one_audio, :] # Trim "Pitch inferred at tail of input audio(few audio points + padded elements)" per audio in batch
    '''
    output_shape = raw_output.shape
    unwrapped_output = raw_output.reshape(output_shape[0] * output_shape[1], output_shape[2])
    return unwrapped_output[:n_output_frames_original, :]
    # 근데 unwrapped_output shape가 지금 (33024, 264bins)임. 즉, (64*344, 264)  가 아니고 (64*344*3/2, 264)
    # 우리는 피치 인코더에서 (64batch, 344frames, 264) 를 얻기 원하고, 64 * 344 frames = 22016 이니까 33024 중에서 22016앞부분만 취하고
    # 뒤의 11008 프레임은 버리면 될까??? 근데 그러면 안 된다.
    # 그러면 각각의 오디오의 3번째 프레임찌꺼기를 버리는게 아니라, 64개오디오 중 후반부 22개오디오에서 나온 프레임 11008개를 모두 버리게 됨.
    # -> [batch*Frames, 264] 2D form에서 [:n_frames,  :] 를 하지 말고,  [batch, frame, 264] 3D form에서 [:,  :n_frames, :]를 하자.
    '''
    return output_batched_trimmed
def run_inference(
    audio_input: Tensor,
    model: keras.Model,
    debug_file: Optional[pathlib.Path] = None,
) -> Dict[str, np.array]:
    """ Main Function of inference. Run the model on the input audio Tensor.
    Args:
        audio_input: The audio to run inference on.
        model: A loaded keras model to run inference with.
        debug_file: An optional path to output debug data to. Useful for testing/verification.
    Returns:
       A dictionary with the notes, onsets and contours from model inference.
    """
    #n_overlapping_frames = 30
    n_overlapping_frames = 0 # There are literally No overlapping Between Frames, because frames of GDDSP DAtaset is from different audio sources
    overlap_len = n_overlapping_frames * FFT_HOP #0 if n_overlapping_frames =0
    hop_size = AUDIO_N_SAMPLES - overlap_len
    audio_windowed, audio_original_length, input_batch_size = get_audio_input(audio_input, overlap_len, hop_size)
    output = model(audio_windowed) # output = raw_output = {onset:~, contour:~, note:~}
    trimmed_contour = trim_output(output["contour"], audio_original_length, n_overlapping_frames, input_batch_size)
    # GDDSP do not use onset and note. JUST Unwrap Overlapping Frame-wise contour -> into 1D Time-Series contour
    # Unwrap Input :  Out: [Frames, Timesteps per frame, 264] ->  [Frames * timesteps per frame, 264]
    # The reason of Unwrapping : to Trim Additional Points exceed the Original Input Audio's Total Sample Points(Batch*sr*duraiton)
    '''
    if debug_file:
        with open(debug_file, "w") as f:
            json.dump(
                {
                    "audio_windowed": audio_windowed.numpy().tolist(),
                    "audio_original_length": audio_original_length,
                    "hop_size_samples": hop_size,
                    "overlap_length_samples": overlap_len,
                    "unwrapped_output": {k: v.tolist() for k, v in unwrapped_output.items()},
                },
                f,
            )
    '''
    # return unwrapped_output
    return trimmed_contour

class OutputExtensions(enum.Enum):
    MIDI = "mid"
    MODEL_OUTPUT_NPZ = "npz"
    MIDI_SONIFICATION = "wav"
    NOTE_EVENTS = "csv"

def save_note_events(
    note_events: List[Tuple[float, float, int, float, Optional[List[int]]]],
    save_path: Union[pathlib.Path, str],
) -> None:
    """Save note events to file

    Args:
        note_events: A list of note event tuples to save. Tuples have the format
            ("start_time_s", "end_time_s", "pitch_midi", "velocity", "list of pitch bend values")
        save_path: The location we're saving it
    """

    with open(save_path, "w") as fhandle:
        writer = csv.writer(fhandle, delimiter=",")
        writer.writerow(["start_time_s", "end_time_s", "pitch_midi", "velocity", "pitch_bend"])
        for start_time, end_time, note_number, amplitude, pitch_bend in note_events:
            row = [start_time, end_time, note_number, int(np.round(127 * amplitude))]
            if pitch_bend:
                row.extend(pitch_bend)
            writer.writerow(row)


# The "predict" method will be mainly used as f0_encoder of GDDSP

def predict(
    audio_input: Tensor,
    model_or_model_path: Union[keras.Model, pathlib.Path, str] = ICASSP_2022_MODEL_PATH,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 127.70,
    minimum_frequency: Optional[float] = None,
    maximum_frequency: Optional[float] = None,
    multiple_pitch_bends: bool = False,
    melodia_trick: bool = True,
    debug_file: Optional[pathlib.Path] = None,
    midi_tempo: float = 120,
) -> Tuple[Dict[str, np.array], pretty_midi.PrettyMIDI, List[Tuple[float, float, int, float, Optional[List[int]]]],]:
    """Run a single prediction.

    Args:
        audio_path: File path for the audio to run inference on.
        model_or_model_path: Path to load the Keras saved model from. Can be local or on GCS.
        onset_threshold: Minimum energy required for an onset to be considered present.
        frame_threshold: Minimum energy requirement for a frame to be considered present.
        minimum_note_length: The minimum allowed note length in milliseconds.
        minimum_freq: Minimum allowed output frequency, in Hz. If None, all frequencies are used.
        maximum_freq: Maximum allowed output frequency, in Hz. If None, all frequencies are used.
        multiple_pitch_bends: If True, allow overlapping notes in midi file to have pitch bends.
        melodia_trick: Use the melodia post-processing step.
        debug_file: An optional path to output debug data to. Useful for testing/verification.
    Returns:
        The model output, midi data and note events from a single prediction
    """

    with no_tf_warnings():
        # It's convenient to be able to pass in a keras saved model so if
        # someone wants to place this function in a loop,
        # the model doesn't have to be reloaded every function call
        import tensorflow as tf
        '''
        # In you want to specify Basic Pitch tensorlfow's device explicitly
        gpus = tf.config.list_physical_devices('GPU')# Set the GPU visible before run Pitch Encoding function of Basic-Pitch Model.
        if not gpus == []:
            tf.config.set_visible_devices(gpus[0], 'GPU')  # Set tensorflow to use the first GPU
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True) # Allow GPU memory growth (optional)
            except: # If set_memory_growth is already True, Setting it True agian will raise tf error. So Just let's Pass this process
                pass
            with tf.device('/GPU:0'):
                if isinstance(model_or_model_path, (pathlib.Path, str)):
                    model = saved_model.load(str(model_or_model_path))
                else:
                    model = model_or_model_path
        else:
            if isinstance(model_or_model_path, (pathlib.Path, str)):
                model = saved_model.load(str(model_or_model_path))
            else:
                model = model_or_model_path
        '''
        if isinstance(model_or_model_path, (pathlib.Path, str)):
            model = saved_model.load(str(model_or_model_path))
        else:
            model = model_or_model_path         
        if isinstance(audio_input, torch.Tensor):
            audio_input = audio_input.numpy()
        # model_output = run_inference(audio_input, model, debug_file)
        trimmed_contour = run_inference(audio_input, model, debug_file)
        #min_note_len = int(np.round(minimum_note_length / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
    '''
    # Originally, Output of Basic-Pitch Inference is [raw_output, midi_data, note_events]
    # And raw_output is made up of [onset, contour, note] (note = quantized contour. semitone-interval.  contour = 1/6tone-interval)
    # But GDDSP will use just contour in raw_output
    # So ignore raw_output[0]=onset, raw_output[2]=note, ignore producing midi_data, note_events
        midi_data, note_events = infer.model_output_to_notes(
            model_output,
            onset_thresh=onset_threshold,
            frame_thresh=frame_threshold,
            min_note_len=min_note_len,  # convert to frames
            min_freq=minimum_frequency,
            max_freq=maximum_frequency,
            multiple_pitch_bends=multiple_pitch_bends,
            melodia_trick=melodia_trick,
            midi_tempo=midi_tempo,
        )
    if debug_file:
        with open(debug_file) as f:
            debug_data = json.load(f)
        with open(debug_file, "w") as f:
            json.dump(
                {
                    **debug_data,
                    "min_note_length": min_note_len,
                    "onset_thresh": onset_threshold,
                    "frame_thresh": frame_threshold,
                    "estimated_notes": [
                        (
                            float(start_time),
                            float(end_time),
                            int(pitch),
                            float(amplitude),
                            [int(b) for b in pitch_bends] if pitch_bends else None,
                        )
                        for start_time, end_time, pitch, amplitude, pitch_bends in note_events
                    ],
                },
                f,
            )
     '''
    #return model_output, midi_data, note_events
    return trimmed_contour

def predict_and_save(
    audio_input_batch: Tensor,
    output_directory: Union[pathlib.Path, str],
    save_midi: bool,
    sonify_midi: bool,
    save_model_outputs: bool,
    save_notes: bool,
    model_path: Union[pathlib.Path, str] = ICASSP_2022_MODEL_PATH,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 58,
    minimum_frequency: Optional[float] = None,
    maximum_frequency: Optional[float] = None,
    multiple_pitch_bends: bool = False,
    melodia_trick: bool = True,
    debug_file: Optional[pathlib.Path] = None,
    sonification_samplerate: int = 44100,
    midi_tempo: float = 120,
) -> None:
    """Make a prediction and save the results to file.

    Args:
        audio_path_list: List of file paths for the audio to run inference on.
        output_directory: Directory to output MIDI and all other outputs derived from the model to.
        save_midi: True to save midi.
        sonify_midi: Whether or not to render audio from the MIDI and output it to a file.
        save_model_outputs: True to save contours, onsets and notes from the model prediction.
        save_notes: True to save note events.
        model_path: Path to load the Keras saved model from. Can be local or on GCS.
        onset_threshold: Minimum energy required for an onset to be considered present.
        frame_threshold: Minimum energy requirement for a frame to be considered present.
        minimum_note_length: The minimum allowed note length in frames.
        minimum_freq: Minimum allowed output frequency, in Hz. If None, all frequencies are used.
        maximum_freq: Maximum allowed output frequency, in Hz. If None, all frequencies are used.
        multiple_pitch_bends: If True, allow overlapping notes in midi file to have pitch bends.
        melodia_trick: Use the melodia post-processing step.
        debug_file: An optional path to output debug data to. Useful for testing/verification.
        sonification_samplerate: Sample rate for rendering audio from MIDI.
    """
    model = saved_model.load(str(model_path))

    for i in range(audio_input_batch.shape[0]):
        audio_input = audio_input_batch[i]
        
        print("")
        try:
            model_output, midi_data, note_events = predict(
                audio_input,
                model,
                onset_threshold,
                frame_threshold,
                minimum_note_length,
                minimum_frequency,
                maximum_frequency,
                multiple_pitch_bends,
                melodia_trick,
                debug_file,
                midi_tempo,
            )

            if save_model_outputs:
                model_output_path = build_output_path(audio_input, output_directory, OutputExtensions.MODEL_OUTPUT_NPZ)
                try:
                    np.savez(model_output_path, basic_pitch_model_output=model_output)
                    file_saved_confirmation(OutputExtensions.MODEL_OUTPUT_NPZ.name, model_output_path)
                except Exception as e:
                    failed_to_save(OutputExtensions.MODEL_OUTPUT_NPZ.name, model_output_path)
                    raise e

            if save_midi:
                try:
                    midi_path = build_output_path(audio_input, output_directory, OutputExtensions.MIDI)
                except IOError as e:
                    raise e
                try:
                    midi_data.write(str(midi_path))
                    file_saved_confirmation(OutputExtensions.MIDI.name, midi_path)
                except Exception as e:
                    failed_to_save(OutputExtensions.MIDI.name, midi_path)
                    raise e

            if sonify_midi:
                midi_sonify_path = build_output_path(audio_input, output_directory, OutputExtensions.MIDI_SONIFICATION)
                try:
                    infer.sonify_midi(midi_data, midi_sonify_path, sr=sonification_samplerate)
                    file_saved_confirmation(OutputExtensions.MIDI_SONIFICATION.name, midi_sonify_path)
                except Exception as e:
                    failed_to_save(OutputExtensions.MIDI_SONIFICATION.name, midi_sonify_path)
                    raise e

            if save_notes:
                note_events_path = build_output_path(audio_input, output_directory, OutputExtensions.NOTE_EVENTS)
                try:
                    save_note_events(note_events, note_events_path)
                    file_saved_confirmation(OutputExtensions.NOTE_EVENTS.name, note_events_path)
                except Exception as e:
                    failed_to_save(OutputExtensions.NOTE_EVENTS.name, note_events_path)
                    raise e
        except Exception as e:
            raise e

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
import numpy as np
import argparse
import h5py
import librosa
from scipy import signal
import matplotlib.pyplot as plt
import time
import math
import pandas as pd
import random

from utilities import (create_folder, read_audio, calculate_scalar_of_tensor, 
    pad_truncate_sequence, get_subdir, read_metadata)
import config


class LogMelExtractor(object):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        '''Log mel feature extractor. 
        
        Args:
          sample_rate: int
          window_size: int
          hop_size: int
          mel_bins: int
          fmin: int, minimum frequency of mel filter banks
          fmax: int, maximum frequency of mel filter banks
        '''
        
        self.window_size = window_size
        self.hop_size = hop_size
        self.window_func = np.hanning(window_size)
        
        self.melW = librosa.filters.mel(
            sr=sample_rate, 
            n_fft=window_size, 
            n_mels=mel_bins, 
            fmin=fmin, 
            fmax=fmax).T
        '''(n_fft // 2 + 1, mel_bins)'''

    def transform(self, audio):
        '''Extract feature of a singlechannel audio file. 
        
        Args:
          audio: (samples,)
          
        Returns:
          feature: (frames_num, freq_bins)
        '''
    
        window_size = self.window_size
        hop_size = self.hop_size
        window_func = self.window_func
        
        # Compute short-time Fourier transform
        stft_matrix = librosa.core.stft(
            y=audio, 
            n_fft=window_size, 
            hop_length=hop_size, 
            window=window_func, 
            center=True, 
            dtype=np.complex64, 
            pad_mode='reflect').T
        '''(N, n_fft // 2 + 1)'''
    
        # Mel spectrogram
        mel_spectrogram = np.dot(np.abs(stft_matrix) ** 2, self.melW)
        
        # Log mel spectrogram
        logmel_spectrogram = librosa.core.power_to_db(
            mel_spectrogram, ref=1.0, amin=1e-10, 
            top_db=None)
        
        logmel_spectrogram = logmel_spectrogram.astype(np.float32)
        
        return logmel_spectrogram


def calculate_feature_for_all_audio_files(args):
    '''Calculate feature of audio files and write out features to a hdf5 file. 
    
    Args:
      dataset_dir: string
      subtask: 'a' | 'b' | 'c'
      data_type: 'development' | 'evaluation'
      workspace: string
      mini_data: bool, set True for debugging on a small part of data
    '''

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    subtask = args.subtask
    data_type = args.data_type
    workspace = args.workspace
    mini_data = args.mini_data
    
    sample_rate = config.sample_rate
    window_size = config.window_size
    hop_size = config.hop_size
    mel_bins = config.mel_bins
    fmin = config.fmin
    fmax = config.fmax
    frames_per_second = config.frames_per_second
    frames_num = config.frames_num
    total_samples = config.total_samples
    classes_num = config.classes_num
    lb_to_idx = config.lb_to_idx
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
        
    sub_dir = get_subdir(subtask, data_type)
    metadata_path = os.path.join(dataset_dir, sub_dir, 'meta.csv')
    audios_dir = os.path.join(dataset_dir, sub_dir, 'audio')
    
    feature_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(sub_dir))
    create_folder(os.path.dirname(feature_path))
        
    # Feature extractor
    feature_extractor = LogMelExtractor(
        sample_rate=sample_rate, 
        window_size=window_size, 
        hop_size=hop_size, 
        mel_bins=mel_bins, 
        fmin=fmin, 
        fmax=fmax)

    # Read metadata
    meta_dict = read_metadata(metadata_path)

    # Extract features and targets 
    if mini_data:
        mini_num = 10
        total_num = len(meta_dict['audio_name'])
        random_state = np.random.RandomState(1234)
        indexes = random_state.choice(total_num, size=mini_num, replace=False)
        meta_dict['audio_name'] = meta_dict['audio_name'][indexes]
        meta_dict['scene_label'] = meta_dict['scene_label'][indexes]
        meta_dict['identifier'] = meta_dict['identifier'][indexes]
        meta_dict['source_label'] = meta_dict['source_label'][indexes]
        
    print('Extracting features of all audio files ...')
    extract_time = time.time()
    
    # Hdf5 file for storing features and targets
    hf = h5py.File(feature_path, 'w')

    hf.create_dataset(
        name='audio_name', 
        data=[audio_name.encode() for audio_name in meta_dict['audio_name']], 
        dtype='S80')

    if 'scene_label' in meta_dict.keys():
        hf.create_dataset(
            name='scene_label', 
            data=[scene_label.encode() for scene_label in meta_dict['scene_label']], 
            dtype='S24')
            
    if 'identifier' in meta_dict.keys():
        hf.create_dataset(
            name='identifier', 
            data=[identifier.encode() for identifier in meta_dict['identifier']], 
            dtype='S24')
            
    if 'source_label' in meta_dict.keys():
        hf.create_dataset(
            name='source_label', 
            data=[source_label.encode() for source_label in meta_dict['source_label']], 
            dtype='S8')

    hf.create_dataset(
        name='feature', 
        shape=(0, frames_num, mel_bins), 
        maxshape=(None, frames_num, mel_bins), 
        dtype=np.float32)

    for (n, audio_name) in enumerate(meta_dict['audio_name']):
        audio_path = os.path.join(audios_dir, audio_name)
        print(n, audio_path)
        
        # Read audio
        (audio, _) = read_audio(
            audio_path=audio_path, 
            target_fs=sample_rate)
        
        # Pad or truncate audio recording
        audio = pad_truncate_sequence(audio, total_samples)
        
        # Extract feature
        feature = feature_extractor.transform(audio)
        
        # Remove the extra frames caused by padding zero
        feature = feature[0 : frames_num]
        
        hf['feature'].resize((n + 1, frames_num, mel_bins))
        hf['feature'][n] = feature
            
    hf.close()
        
    print('Write hdf5 file to {} using {:.3f} s'.format(
        feature_path, time.time() - extract_time))
    
    
def calculate_scalar(args):
    '''Calculate and write out scalar of features. 
    
    Args:
      data_type: 'train'
      workspace: string
      mini_data: bool, set True for debugging on a small part of data
    '''

    # Arguments & parameters
    subtask = args.subtask
    data_type = args.data_type
    workspace = args.workspace
    mini_data = args.mini_data
    
    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
    
    sub_dir = get_subdir(subtask, data_type)
    
    feature_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(sub_dir))
        
    scalar_path = os.path.join(workspace, 'scalars', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(sub_dir))
    create_folder(os.path.dirname(scalar_path))
        
    # Load data
    load_time = time.time()
    
    with h5py.File(feature_path, 'r') as hf:
        features = hf['feature'][:]
    
    # Calculate scalar
    features = np.concatenate(features, axis=0)
    (mean, std) = calculate_scalar_of_tensor(features)
    
    with h5py.File(scalar_path, 'w') as hf:
        hf.create_dataset('mean', data=mean, dtype=np.float32)
        hf.create_dataset('std', data=std, dtype=np.float32)
    
    print('All features: {}'.format(features.shape))
    print('mean: {}'.format(mean))
    print('std: {}'.format(std))
    print('Write out scalar to {}'.format(scalar_path))
            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    parser_logmel = subparsers.add_parser('calculate_feature_for_all_audio_files')
    parser_logmel.add_argument('--dataset_dir', type=str, required=True)
    parser_logmel.add_argument('--subtask', type=str, choices=['a', 'b', 'c'], required=True)
    parser_logmel.add_argument('--data_type', type=str, choices=['development', 'evaluation'], required=True)
    parser_logmel.add_argument('--workspace', type=str, required=True)
    parser_logmel.add_argument('--mini_data', action='store_true', default=False)

    parser_scalar = subparsers.add_parser('calculate_scalar')
    parser_scalar.add_argument('--subtask', type=str, choices=['a', 'b', 'c'], required=True)
    parser_scalar.add_argument('--data_type', type=str, choices=['development', 'evaluation'], required=True)
    parser_scalar.add_argument('--workspace', type=str, required=True)
    parser_scalar.add_argument('--mini_data', action='store_true', default=False)
    
    args = parser.parse_args()
    
    if args.mode == 'calculate_feature_for_all_audio_files':
        calculate_feature_for_all_audio_files(args)
        
    elif args.mode == 'calculate_scalar':
        calculate_scalar(args)
        
    else:
        raise Exception('Incorrect arguments!')
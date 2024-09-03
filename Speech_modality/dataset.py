# -*- coding: utf-8 -*- #
"""
    FileName     [ dataset.py ]
    Synopsis     [ the emotion classifier dataset ]
    Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""

import json
from pathlib import Path
from os.path import join as path_join

import torchaudio
from torch.utils.data import Dataset
import torch
from torchaudio.transforms import Resample

SAMPLE_RATE = 16000


class IEMOCAPDataset(Dataset):
    def __init__(self, data_dir, meta_path, pre_load=True):
        self.data_dir = data_dir
        self.pre_load = pre_load
        with open(meta_path, 'r') as f:
            self.data = json.load(f)
        self.class_dict = self.data['labels']
        self.idx2emotion = {value: key for key, value in self.class_dict.items()}
        self.class_num = len(self.class_dict)
        self.meta_data = self.data['meta_data']
        _, origin_sr = torchaudio.load(
            path_join(self.data_dir, self.meta_data[0]['path']))
        self.resampler = Resample(origin_sr, SAMPLE_RATE)
        if self.pre_load:
            self.wavs = self._load_all()
    
    def stereo_to_mono_convertor(self, signal):
        # If there is more than 1 channel in your audio
        if signal.shape[0] > 1:
            # Do a mean of all channels and keep it in one channel
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _load_wav(self, path):
        wav, _ = torchaudio.load(path_join(self.data_dir, path))
        wav = self.stereo_to_mono_convertor(wav)
        wav = self.resampler(wav).squeeze(0)
        return wav

    def _load_all(self):
        wavforms = []
        for info in self.meta_data:
            wav = self._load_wav(info['path'])
            wavforms.append(wav)
        return wavforms

    def __getitem__(self, idx):
        label = self.meta_data[idx]['label']
        label = self.class_dict[label]
        if self.pre_load:
            wav = self.wavs[idx]
        else:
            wav = self._load_wav(self.meta_data[idx]['path'])
        return wav.numpy(), label, Path(self.meta_data[idx]['path']).stem

    def __len__(self):
        return len(self.meta_data)

class SynthesisDataset(Dataset):
    def __init__(self, data_dir, meta_path, pre_load=False):
        self.data_dir = data_dir
        self.pre_load = pre_load
        with open(meta_path, 'r') as f:
            self.data = json.load(f)
        self.class_dict = self.data['labels']
        self.idx2emotion = {value: key for key, value in self.class_dict.items()}
        self.class_num = len(self.class_dict)
        self.meta_data = self.data['meta_data']
        if 'MSP-IMPROV' in self.meta_data[0]['path']:
            _, origin_sr = torchaudio.load(
            path_join('/mnt/Internal/ASR/MSP_IMPROV', self.meta_data[0]['path']))
        else:
            _, origin_sr = torchaudio.load(
                path_join(self.data_dir, self.meta_data[0]['path']))
        self.resampler = Resample(origin_sr, SAMPLE_RATE)
        if self.pre_load:
            self.wavs = self._load_all()

    def stereo_to_mono_convertor(self, signal):
        # If there is more than 1 channel in your audio
        if signal.shape[0] > 1:
            # Do a mean of all channels and keep it in one channel
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _load_wav(self, path):
        # print(path)
        if 'MSP-IMPROV' not in path:
            wav, _ = torchaudio.load(path_join(self.data_dir, path))
        else:
            wav, _ = torchaudio.load(path_join('/mnt/Internal/ASR/MSP_IMPROV', path))
        wav = self.stereo_to_mono_convertor(wav)
        wav = self.resampler(wav).squeeze(0)
        return wav

    def _load_all(self):
        wavforms = []
        for info in self.meta_data:
            wav = self._load_wav(info['path'])
            wavforms.append(wav)
        return wavforms

    def __getitem__(self, idx):
        label = self.meta_data[idx]['label']
        label = self.class_dict[label]
        if self.pre_load:
            wav = self.wavs[idx]
        else:
            wav = self._load_wav(self.meta_data[idx]['path'])
        return wav.numpy(), label, Path(self.meta_data[idx]['path']).stem

    def __len__(self):
        return len(self.meta_data)

def collate_fn(samples):
    return zip(*samples)
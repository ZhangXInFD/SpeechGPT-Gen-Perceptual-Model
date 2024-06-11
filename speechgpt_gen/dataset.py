from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
from functools import wraps
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from beartype import beartype

TOKEN_PAD_VALUE = 1024
WAV_PAD_VALUE = 0

def collate_one_or_multiple_tensors(fn):
    @wraps(fn)
    def inner(data):
        is_one_data = not isinstance(data[0], tuple)
        if is_one_data:
            data = tuple(map(lambda x:torch.stack(x), data))
            return data
        outputs = []
        for datum in zip(*data):
            if isinstance(datum[0], torch.Tensor):
                if datum[0].dtype == torch.bool:
                    output = pad_sequence(datum, batch_first=True, padding_value=False)
                else:
                    output = fn(datum)
            else:
                output = list(datum)
            outputs.append(output)

        return tuple(outputs)
    return inner

@collate_one_or_multiple_tensors
def tokens_collate_fn(data):
    return pad_sequence(data, batch_first=True, padding_value=TOKEN_PAD_VALUE)

@collate_one_or_multiple_tensors
def wav_collate_fn(data):
    return pad_sequence(data, batch_first=True, padding_value=WAV_PAD_VALUE)

def get_dataloader(ds, is_raw_wav=False, **kwargs):
    collate_fn = wav_collate_fn if is_raw_wav else tokens_collate_fn
    return DataLoader(ds, collate_fn=collate_fn, **kwargs)



class SoundStormDataset(Dataset):
    
    @beartype
    def __init__(self, 
                 file_list: list,
                 is_raw_wav: bool=False,
                 is_tokens: bool=False,
                 sample_rate: int= 16000,
                 max_sequence: int=512,
                 hierarchical: bool=False,
                 audio_root = None,
                 downsample_rate: int=320,
                 device = 'cpu'):
        self.file_list = file_list
        self.is_raw_wav = is_raw_wav
        self.is_tokens = is_tokens
        self.sample_rate = sample_rate
        self.hierarchical = hierarchical
        self.audio_root = audio_root
        self.downsample_rate = downsample_rate
        self.max_sequence = max_sequence
        self.device = device
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        file = self.file_list[index].strip()
        if self.is_tokens:
            tokens = torch.from_numpy(np.load(file))
            if tokens.size(0) > self.max_sequence:
                start = torch.randint(0, tokens.size(0) - self.max_sequence, (1,))
                tokens = tokens[start: (start + self.max_squence)]
            semantic_tokens = tokens[:, 0]
            acoustic_tokens = tokens[:, 1:]
            return semantic_tokens[:self.max_sequence], acoustic_tokens[:self.max_sequence]
        # while True:
        #     try:
        #         wav, sr = torchaudio.load(file)
        #         if wav.sum() != 0:
        #             break
        #         raise ValueError('Error audio file')
        #     except:
        #         with open('./error_file.txt', 'a+') as f:
        #             f.write(file + '\n')
        #         index -= 1
        #         file = self.file_list[index].strip()
        if self.hierarchical:
            file_name, units = file.split('\t')
            units = torch.from_numpy(np.array(units.split(' ')).astype(int))
            spk, chapter = file_name.split('_')[:2]
            wav_file = f'{self.audio_root}/{spk}/{chapter}/{file_name}.flac'
            wav, sr = torchaudio.load(wav_file)
            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            wav = wav.mean(axis=0)
            wav = wav.unsqueeze(0)
        else:
            wav, sr = torchaudio.load(file)
            if wav.size(0) > 1:
                wav = wav.mean(axis=0)
                wav = wav.unsqueeze(0)
            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if self.is_raw_wav:
            if wav.size(-1) > self.max_sequence:
                start = torch.randint(0, wav.size(-1) - self.max_sequence, (1,))
                wav = wav[:, start: (start + self.max_sequence)]
                if self.hierarchical:
                    units = units[start // self.downsample_rate:(start + self.max_sequence) // self.downsample_rate]
            if self.hierarchical:
                return units, wav[:, :self.max_sequence].squeeze(0), min(wav.size(-1), self.max_sequence)            
            return wav.squeeze()[:self.max_sequence], min(wav.size(-1), self.max_sequence)
        
class HierDataset(Dataset):
    
    def __init__(self,
                 data_list, 
                 audio_root: str,
                 sample_rate: int= 16000,
                 max_sequence: int=512,
                 downsample_rate: int=320,
                 ):
        
        self.data_list = data_list
        self.audio_root = audio_root
        self.sample_rate = sample_rate
        self.max_sequence = max_sequence
        self.downsample_rate = downsample_rate
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data = self.data_list[index]
        # wav, sr = torchaudio.load(data['audio_path'])
        # audio_path, units, text = data.strip().split('\t')
        audio_path, units = data.strip().split('\t')[:2]
        audio_path = f'{self.audio_root}/{"/".join(audio_path.split("_")[:2])}/{audio_path}.flac'
        wav, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        # units = torch.from_numpy(data['units'])
        units = torch.from_numpy(np.array(units.split()).astype(int))
        if units.size(-1) > self.max_sequence:
            start = torch.randint(0, units.size(-1) - self.max_sequence, (1,))
            wav = wav[:, (start * self.downsample_rate): (start + self.max_sequence) * self.downsample_rate]
            units = units[start: (start + self.max_sequence)]
        mask = torch.ones_like(units, dtype=torch.bool, device=units.device)
            
        return wav.squeeze(), units, mask
        
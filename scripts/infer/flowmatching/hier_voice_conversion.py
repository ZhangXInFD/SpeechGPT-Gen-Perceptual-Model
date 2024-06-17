from speechgpt_gen_preceptual import HierarchicalConditionalFlowMatcher
import torch
import torchaudio
from einops import rearrange
import os
import random
from tqdm import tqdm
import yaml
import joblib
import fairseq
import torch.nn.functional as F
import numpy as np
from torchaudio.functional import resample
import shutil
import dac
import argparse

class FeatureReader(object):
    def __init__(self, ckpt_path, layer, max_chunk=1600000, fp16=False, sampling_rate=16000, device='cpu'):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.device = device
        self.model = model[0].eval().to(self.device)
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        self.fp16 = fp16
        if fp16:
            self.model.half()
        
        self.layer_shift = 0
        self.target_sample_hz = sampling_rate
        
        # logger.info(f"TASK CONFIG:\n{self.task.cfg}")
    
    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self

    def read_audio(self, path):
        wav, sr = torchaudio.load(path)
        if sr != self.target_sample_hz:
            wav = resample(wav, sr, self.target_sample_hz)
        return wav

    @torch.no_grad()
    def get_feats(self, waveform):
        x = waveform
        with torch.no_grad():
            if self.fp16:
                x = x.half().to(self.device)
            else:
                x = x.float().to(self.device)
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                        source=x_chunk,
                        padding_mask=None,
                        mask=False,
                        output_layer=self.layer + self.layer_shift,
                )
        
                feat.append(feat_chunk)
        if len(feat) == 0:
            return torch.zeros(0, 0)
        return torch.cat(feat, 1).squeeze(0)




class ApplyKmeans(object):
    def __init__(self, km_path, device='cpu'):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        self.device = device
        if self.device != 'cpu':
            self.C = self.C.to(device)
            self.Cnorm = self.Cnorm.to(device)
            
    def to(self, device):
        self.device = device
        self.C = self.C.to(device)
        self.Cnorm = self.Cnorm.to(device)
        return self

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            self.C = self.C.to(x)
            self.Cnorm = self.Cnorm.to(x)
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


class Speech2Unit(torch.nn.Module):
    def __init__(
        self, 
        ckpt_dir,
        layer=9, 
        max_chunk=1600000, 
        fp16=False, 
        sampling_rate=16000,
        device='cpu'):

        """
        Args:
            ckpt_dir(str): path to hubert model dir(e.g. hubert_base_ls960.pt)
            layer(int): feat from which layer of hubert models defauly by 9
            max_chunk(int): default by 1600000
            fp16(bool): default by False
            sampling_rate(int): sampling_rate default by 16000
        """
        super().__init__()

        ckpt_path = os.path.join(ckpt_dir, "hubert_base_ls960.pt")
        km_path = os.path.join(ckpt_dir, "hubert_base_ls960_L9_km500.bin")

        self.feature_reader = FeatureReader(ckpt_path, layer, max_chunk, fp16, sampling_rate, device=device)
        self.apply_kmeans = ApplyKmeans(km_path, device=device)
        self.device= device
        
    def to(self, device):
        self.device = device
        self.feature_reader.to(device)
        self.apply_kmeans.to(device)
        return self
    
    @staticmethod
    def merge_duplicates(cluster_ids):
        dup_cluster_list = []
        duration_list = []
        count = 1
        for i in range(0, len(cluster_ids)):
            if i + 1 < len(cluster_ids) and cluster_ids[i] == cluster_ids[i+1]:
                count += 1
            else:
                dup_cluster_list.append(cluster_ids[i])
                duration_list.append(count)
                count = 1
        return dup_cluster_list, duration_list
    

    def __call__(self, path, merged=False):
        waveform = self.feature_reader.read_audio(path).to(device)
        
        feat = self.feature_reader.get_feats(waveform)
        cluster_ids = self.apply_kmeans(feat)

        # merged_units = "<sosp>" + "".join([f"<{str(x)}>" for x in dup_cluster_list]) + "<eosp>"
        # unmerged_units = "<sosp>" + "".join([f"<{str(x)}>" for x in cluster_ids]) + "<eosp>"

        if not merged:
            return cluster_ids
        else:
            dup_cluster_list, duration_list = self.merge_duplicates(cluster_ids)
            return dup_cluster_list
        
class DACWrapper(torch.nn.Module):
    
    def __init__(self, ckpt_path) -> None:
        super().__init__()
        self.model = dac.DAC.load(ckpt_path)
        self.downsample_rate = self.model.hop_length
        self.sample_rate = self.model.sample_rate
        
    @torch.inference_mode
    def encode(self, wav, return_code=False):
        wav = self.model.preprocess(wav, self.sample_rate)
        z, codes, latents, _, _ = self.model.encode(wav)
        if return_code:
            return codes
        else:
            return rearrange(z, 'b d t -> b t d')
        
    @torch.inference_mode
    def decode(self, z):
        return self.model.decode(z)


class VoiceConversion:
    
    def __init__(self, 
                 tokenizer, 
                 model: HierarchicalConditionalMatcher, 
                 semantic_tokenizer: Speech2Unit,
                 device='cpu',
                 explicit=False):
        self.tokenizer = tokenizer.to(device)
        self.tokenizer.eval()
        self.model = model.to(device)
        self.model.eval()
        self.semantic_tokenizer = semantic_tokenizer.to(device)
        self.semantic_tokenizer.eval()
        self.device = device
        self.explicit = explicit
        
    @torch.no_grad()    
    def semantic_encode(self, wav_file):
        tokens = self.semantic_tokenizer(wav_file)
        tokens = torch.from_numpy(tokens).to(self.device)
        return tokens
    
    @torch.no_grad()    
    def encode(self, wav_file, target_file=None, target_length=None):
        wav, sr = torchaudio.load(wav_file)
        if sr != self.tokenizer.sample_rate:
            wav = torchaudio.functional.resample(wav, sr , self.tokenizer.sample_rate)
        if target_file is not None:
            if target_length is not None:
                torchaudio.save(target_file, wav[:, :target_length], self.tokenizer.sample_rate)
            else:
                torchaudio.save(target_file, wav, self.tokenizer.sample_rate)
        wav = wav.to(self.device)
        rep = self.tokenizer.encode(wav.unsqueeze(0))
        return rep
    
    @torch.no_grad()
    def decode(self, file, rep):
        wav = self.tokenizer.decode(rearrange(rep, 'b t d -> b d t'))
        torchaudio.save(file, wav.squeeze(0).cpu().detach(), self.tokenizer.sample_rate)
        
    @torch.no_grad()    
    def generate(self, prompt_file, src_file, tgt_dir, max_prompt_token_length=150, steps=[8]):
        if not os.path.exists(tgt_dir):
            os.makedirs(tgt_dir)
        src_rep = self.encode(src_file)
        self.decode(f'{tgt_dir}/raw.wav', src_rep)
        semantic_tokens = self.semantic_encode(src_file).unsqueeze(0)
        prompt_rep = self.encode(prompt_file, target_file=f'{tgt_dir}/prompt_o.wav', target_length=max_prompt_token_length * self.tokenizer.downsample_rate)[:, :max_prompt_token_length]
        prompt_semantic_tokens = self.semantic_encode(prompt_file)[:max_prompt_token_length].unsqueeze(0)
        prompt_rep = prompt_rep[:, :prompt_semantic_tokens.size(-1)]
        self.decode(f'{tgt_dir}/prompt_r.wav', prompt_rep)
        # if self.explicit:
        #     prompt_rep = self.encode(prompt_file, st=1)[:, :max_prompt_token_length]
        for step in steps:
            # generated = self.model.generate(semantic_emb=src_rep,
            #                                 steps = step)
            # self.decode(f'{tgt_dir}/unconditonal_{step}.wav', generated + src_rep if self.explicit else generated)
            generated = self.model.generate(semantic_tokens=semantic_tokens,
                                            context=prompt_rep,
                                            context_semantic_tokens=prompt_semantic_tokens,
                                            steps = step)
            self.decode(f'{tgt_dir}/generate_{step}.wav', generated + src_rep if self.explicit else generated)
        
def search_file(file_list, file_prefix):
    for filename in file_list:
        if filename.startswith(file_prefix):
            return filename
    return False        
    

if __name__ == '__main__':
    ckpt_dir = '/remote-home/xzhang/SpeechGPT-Gen-Flow-Matcher/Log/hier/hubert_dac'
    dev_set = 'vctk'
    with open(f'{ckpt_dir}/config.yml') as f:
        cfg = yaml.safe_load(f)
    
     
    semantic_tokenizer = Speech2Unit('/remote-home/xzhang/audiolm/hubert_kmeans/checkpoints')
    st_ckpt = cfg['trainer_args'].get('speechtokenizer_ckpt') 
    tokenizer = DACWrapper(st_ckpt)
    
    
    # cnf_model = HierarchicalConditionalMatcher(cfg=cfg['model_args'])
    # cnf_model.load(f'{ckpt_dir}/HierarchicalConditionalFlowMatcher_best_dev.pt')
    cnf_model = HierarchicalConditionalFlowMatcher.from_pretrained(ckpt_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vc = VoiceConversion(tokenizer=tokenizer,
                        model=cnf_model,
                        semantic_tokenizer=semantic_tokenizer,
                        device=device,
                        explicit=cfg['model_args'].get("explicit", False))
    if dev_set == 'librispeech':
        root_dir = '/remote-home/share/data/SpeechPretrain/LibriSpeech/LibriSpeech/dev-clean'
        prompt_dir = '/remote-home/share/data/SpeechPretrain/LibriSpeech/LibriSpeech/dev-clean'
    elif dev_set == 'vctk':
        root_dir = '/remote-home/share/data/SpeechPretrain/VCTK/wav48_silence_trimmed'
        prompt_dir = '/remote-home/share/data/SpeechPretrain/VCTK/wav48_silence_trimmed'
    if dev_set == 'cross_lingual':
        root_dir = '/remote-home/share/data/SpeechPretrain/LibriSpeech/LibriSpeech/dev-clean'
        prompt_dir = '/remote-home/share/data/SpeechPretrain/AIShell-2/data/wav'
    text_dir = '/remote-home/share/data/SpeechPretrain/VCTK/txt'
    prompt_speakers_all =  [folder for folder in os.listdir(prompt_dir if 'vctk' not in root_dir.lower() else text_dir) if '.txt' not in folder]
    speakers = [folder for folder in os.listdir(root_dir if 'vctk' not in root_dir.lower() else text_dir) if '.txt' not in folder]
    if 'librispeech' not in root_dir.lower():
        file_dict = {speaker: os.listdir(f'{root_dir}/{speaker}') for speaker in speakers}
    else:
        file_dict = {speaker:[f'{chapter}/{file}' for chapter in os.listdir(f'{root_dir}/{speaker}') for file in os.listdir(f'{root_dir}/{speaker}/{chapter}') if '.txt' not in file] for speaker in speakers}
    prompt_file_dict = {speaker: os.listdir(f'{prompt_dir}/{speaker}') for speaker in prompt_speakers_all} if 'librispeech' != dev_set else file_dict
    tgt_root = f'./voice_conversion_with_gt/{dev_set}/hier/hubert_dac_final'
    k = 20
    random.seed(0)
    prompt_speakers = random.sample(prompt_speakers_all, k)
    src_speakers = random.sample(speakers, k)
    for prompt_speaker, src_speaker in tqdm(zip(prompt_speakers, src_speakers)):
        for i in range(2):
            while src_speaker == prompt_speaker:
                src_speaker = random.choice(speakers)
            if 'VCTK' in root_dir:
                src_files = [x for x in file_dict[src_speaker] if int(x.split('_')[1]) < 14]
                prompt_files = [x for x in prompt_file_dict[prompt_speaker] if int(x.split('_')[1]) > 14]
            else:
                src_files = file_dict[src_speaker]
                prompt_files = prompt_file_dict[prompt_speaker]
            src_file = random.choice(src_files)
            prompt_file = random.choice(prompt_files)
            tgt_dir = f'{tgt_root}/{prompt_speaker}_{src_speaker}_{i}'
            if not os.path.exists(tgt_dir):
                os.makedirs(tgt_dir)
            if 'vctk' in root_dir.lower():
                src_text_file = '_'.join(src_file.split('.')[0].split('_')[:2]) +'.txt'
                shutil.copy(f'{text_dir}/{src_speaker}/{src_text_file}', f'{tgt_dir}/text.txt')
                tgt_gt = '_'.join(src_file.replace(src_speaker, prompt_speaker).split('_')[:2])
                tgt_gt = search_file(file_list=prompt_file_dict[prompt_speaker], file_prefix=tgt_gt)
                if not tgt_gt:
                    shutil.rmtree(tgt_dir)
                    continue
                else:
                    tgt_gt_rep = vc.encode(f'{prompt_dir}/{prompt_speaker}/{tgt_gt}')
                    vc.decode(f'{tgt_dir}/gt.wav', tgt_gt_rep)  
                                  
            elif 'librispeech' in root_dir.lower():
                spk, chapter, idx = src_file.split('/')[-1].split('.')[0].split('-')
                with open(f'{tgt_dir}/text.txt', 'w+') as out_f:
                    with open(f'{root_dir}/{src_speaker}/{chapter}/{src_speaker}-{chapter}.trans.txt', 'r') as in_f:
                        out_f.write(in_f.readlines()[int(idx)])
            vc.generate(prompt_file=f'{prompt_dir}/{prompt_speaker}/{prompt_file}',
                        src_file=f'{root_dir}/{src_speaker}/{src_file}',
                        tgt_dir=tgt_dir,
                        steps=[4, 8, 16, 32, 64, 128, 256, 512, 1024]
            )
from speechgpt_gen_preceptual import ConditionalFlowMatcher
import torch
from speechtokenizer import SpeechTokenizer
import torchaudio
from einops import rearrange
import os
import random
from tqdm import tqdm
import shutil
import yaml

class VoiceConversion:
    
    def __init__(self, 
                 tokenizer: SpeechTokenizer, 
                 model: ConditionalFlowMatcher, 
                 device='cpu',
                 explicit=False):
        self.tokenizer = tokenizer.to(device)
        self.tokenizer.eval()
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.explicit = explicit
    
    @torch.no_grad()    
    def encode(self, wav_file, st=0, end=8, target_file=None, target_length=None):
        wav, sr = torchaudio.load(wav_file)
        if sr != self.tokenizer.sample_rate:
            wav = torchaudio.functional.resample(wav, sr , self.tokenizer.sample_rate)
        if target_file is not None:
            if target_length is not None:
                torchaudio.save(target_file, wav[:, :target_length], self.tokenizer.sample_rate)
            else:
                torchaudio.save(target_file, wav, self.tokenizer.sample_rate)
        tokens = self.tokenizer.encode(wav.unsqueeze(0).to(self.device))
        rep = self.tokenizer.quantizer.decode(tokens[st:end], st=st)
        return rearrange(rep, 'b d t -> b t d')
    
    @torch.no_grad()
    def decode(self, file, rep):
        wav = tokenizer.decoder(rearrange(rep, 'b t d -> b d t'))
        torchaudio.save(file, wav.squeeze(0).cpu().detach(), self.tokenizer.sample_rate)
        
    @torch.no_grad()    
    def generate(self, prompt_file, src_file, tgt_dir, max_prompt_token_length=150, steps=[8]):
        if not os.path.exists(tgt_dir):
            os.makedirs(tgt_dir)
        src_rep = self.encode(src_file)
        self.decode(f'{tgt_dir}/raw.wav', src_rep)
        src_rep = self.encode(src_file, st=0, end=1)
        prompt_rep = self.encode(prompt_file, target_file=f'{tgt_dir}/prompt_o.wav', target_length=max_prompt_token_length * self.tokenizer.downsample_rate)[:, :max_prompt_token_length]
        prompt_semantic_rep = self.encode(prompt_file, st=0, end=1)[:, :max_prompt_token_length]
        self.decode(f'{tgt_dir}/prompt_r.wav', prompt_rep)
        # if self.explicit:
        #     prompt_rep = self.encode(prompt_file, st=1)[:, :max_prompt_token_length]
        for step in steps:
            # generated = self.model.generate(semantic_emb=src_rep,
            #                                 steps = step)
            # self.decode(f'{tgt_dir}/unconditonal_{step}.wav', generated + src_rep if self.explicit else generated)
            generated = self.model.generate(semantic_emb=src_rep,
                                            context=prompt_rep,
                                            context_semantic_emb=prompt_semantic_rep,
                                            steps = step)
            self.decode(f'{tgt_dir}/generate_{step}.wav', generated + src_rep if self.explicit else generated)
        
def search_file(file_list, file_prefix):
    for filename in file_list:
        if filename.startswith(file_prefix):
            return filename
    return False        
    

if __name__ == '__main__':
    ckpt_dir = '/remote-home/share/personal/xzhang/SpeechGPT-gen/spt_snake'
    dev_set = 'vctk'
    with open(f'{ckpt_dir}/config.yml') as f:
        cfg = yaml.safe_load(f)
    
    st_cfg = cfg['trainer_args']['speechtokenizer_cfg']
    st_ckpt = cfg['trainer_args']['speechtokenizer_ckpt']
     
    tokenizer = SpeechTokenizer.load_from_checkpoint(st_cfg, st_ckpt)
    
    
    # cnf_model = ConditionalFlowMatcher(cfg=cfg['model_args'])
    # cnf_model.load(f'{ckpt_dir}/ConditionalFlowMatcher_best_dev.pt')
    cnf_model = ConditionalFlowMatcher.from_pretrained(ckpt_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vc = VoiceConversion(tokenizer=tokenizer,
                        model=cnf_model,
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
    tgt_root = f'./eval/voice_conversion/{dev_set}/prior/spt_snake'
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
                    # print(src_file, prompt_file, tgt_gt)
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

from speechgpt_gen import ConditionalFlowMatcher
import torch
from speechtokenizer import SpeechTokenizer
import torchaudio
from einops import rearrange
import os
import random
from tqdm import tqdm
import shutil
import yaml
import json

class semantic2wav:
    
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
        wav = self.tokenizer.decoder(rearrange(rep, 'b t d -> b d t'))
        torchaudio.save(file, wav.squeeze(0).cpu().detach(), self.tokenizer.sample_rate)
        
    @torch.no_grad()    
    def generate(self, prompt_file, src_token, tgt_dir, max_prompt_token_length=150, steps=[8]):
        if not os.path.exists(tgt_dir):
            os.makedirs(tgt_dir)
        src_rep = self.tokenizer.quantizer.decode(src_token.unsqueeze(0).unsqueeze(1).to(self.model.device))
        src_rep = rearrange(src_rep, 'b d t -> b t d')
        self.decode(f'{tgt_dir}/rvq1_recon.wav', src_rep)
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
    # ckpt_dir = '/remote-home/xzhang/Speech/USLM2/Log/uconformer/spt_base'
    ckpt_dir = '/remote-home/xzhang/Speech/SpeechGPT-Gen-Flow-Matcher/Log/uconformer_concat_cond/spt_snake'
    dev_set = 'librispeech'
    with open(f'{ckpt_dir}/config.yml') as f:
        cfg = yaml.safe_load(f)
    
    st_cfg = cfg['trainer_args']['speechtokenizer_cfg']
    st_ckpt = cfg['trainer_args']['speechtokenizer_ckpt']
     
    tokenizer = SpeechTokenizer.load_from_checkpoint(st_cfg, st_ckpt)
    
    
    cnf_model = ConditionalFlowMatcher(cfg=cfg['model_args'])
    cnf_model.load(f'{ckpt_dir}/ConditionalFlowMatcher_best_dev.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vc = semantic2wav(tokenizer=tokenizer,
                        model=cnf_model,
                        device=device,
                        explicit=cfg['model_args'].get("explicit", False))
    if dev_set == 'librispeech':
        prompt_dir = '/remote-home/share/data/SpeechPretrain/LibriSpeech/LibriSpeech/dev-clean'
    elif dev_set == 'vctk':
        prompt_dir = '/remote-home/share/data/SpeechPretrain/VCTK/wav48_silence_trimmed'
    if dev_set == 'cross_lingual':
        prompt_dir = '/remote-home/share/data/SpeechPretrain/AIShell-2/data/wav'
    prompt_speakers_all =  [folder for folder in os.listdir(prompt_dir) if '.txt' not in folder]
    prompt_file_dict = {speaker: os.listdir(f'{prompt_dir}/{speaker}') for speaker in prompt_speakers_all} \
        if 'librispeech' != dev_set else {speaker:[f'{chapter}/{file}' for chapter in os.listdir(f'{prompt_dir}/{speaker}') for file in os.listdir(f'{prompt_dir}/{speaker}/{chapter}') if '.txt' not in file] for speaker in prompt_speakers_all}
    tgt_root = f'./tts/{dev_set}/uconformer_concat_cond/spt_snake'
    k = 3
    random.seed(0)
    s_file = '/remote-home/xzhang/Speech/SpeechGPT-Gen-Flow-Matcher/librispeech_tts.jsonl'
    with open(s_file) as f:
        data_list = [json.loads(line) for line in f.readlines()]
    for j, sample in tqdm(enumerate(random.sample(data_list, 30))):
        for i in range(k):
            speaker = random.choice(prompt_speakers_all)
            prompt_file = random.choice(prompt_file_dict[speaker])
            tgt_dir = f'{tgt_root}/{j}_{speaker}'
            if not os.path.exists(tgt_dir):
                os.makedirs(tgt_dir)
                                  
            text = sample['input']
            src_token = torch.tensor([int(x) for x in sample['response_r'].strip('<SOSP>').strip('<EOSP>').strip('<').strip('>').split('><')])
            with open(f'{tgt_dir}/text.txt', 'w+') as out_f:
                out_f.write(text)
            vc.generate(prompt_file=f'{prompt_dir}/{speaker}/{prompt_file}',
                        src_token=src_token,
                        tgt_dir=tgt_dir,
                        steps=[1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
            )

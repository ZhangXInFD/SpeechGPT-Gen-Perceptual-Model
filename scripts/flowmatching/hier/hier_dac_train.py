import dac
from speechgpt_gen import ConditionalFlowMatcherTrainer, HierarchicalConditionalMatcher, HierDataset
import yaml
import argparse
import torch.nn as nn
import torch
from einops import rearrange



class DACWrapper(nn.Module):
    
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
        


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, help='Config file path')
    parser.add_argument('--continue_train', action='store_true', help='Continue trainning from checkpoints')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cnf_model = HierarchicalConditionalMatcher(cfg=cfg.get('model_args'))
    
    # Initial tokenizer
    st_ckpt = cfg['trainer_args'].get('speechtokenizer_ckpt') 
    tokenizer = DACWrapper(st_ckpt)
    tokenizer.eval()
    
    # Dataset file
    train_file_list = cfg['trainer_args'].get('train_file_list')
    valid_file_list = cfg['trainer_args'].get('valid_file_list')
    with open(train_file_list, 'r') as f:
        train_file_list = f.readlines()
    trainset = HierDataset(data_list=train_file_list,
                           audio_root=f"{cfg['trainer_args'].get('train_audio_root')}",
                           sample_rate=tokenizer.sample_rate,
                           max_sequence=cfg['trainer_args'].get('max_sequence'),
                           downsample_rate=tokenizer.downsample_rate)
    with open(valid_file_list, 'r') as f:
        valid_file_list = f.readlines()
    validset = HierDataset(data_list=valid_file_list,
                           audio_root=cfg['trainer_args'].get('valid_audio_root'),
                           sample_rate=tokenizer.sample_rate,
                           max_sequence=cfg['trainer_args'].get('max_sequence'),
                           downsample_rate=tokenizer.downsample_rate)


    
    
    trainer = ConditionalFlowMatcherTrainer(model=cnf_model,
                                cfg=cfg,
                                trainset=trainset,
                                devset=validset,
                                tokenizer=tokenizer,
                                )
    if args.continue_train:
        trainer.continue_train()
    else:
        trainer.train()
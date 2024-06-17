from speechgpt_gen import SoundStorm, SoundStormTrainer
from speechtokenizer import SpeechTokenizer
import yaml
import argparse
import torch


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, help='Config file path')
    parser.add_argument('--continue_train', action='store_true', help='Continue trainning from checkpoints')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    soundstorm = SoundStorm(cfg=cfg.get('model_args'))
    
    
    
    # Dataset file
    train_file_list = cfg['trainer_args'].get('train_file_list')
    valid_file_list = cfg['trainer_args'].get('valid_file_list')
    with open(train_file_list, 'r') as f:
        train_file_list = f.readlines()
    with open(valid_file_list, 'r') as f:
        valid_file_list = f.readlines()


    # Initial tokenizer
    st_cfg = cfg['trainer_args'].get('speechtokenizer_cfg')
    st_ckpt = cfg['trainer_args'].get('speechtokenizer_ckpt') 
    tokenizer = SpeechTokenizer.load_from_checkpoint(st_cfg, st_ckpt)
    tokenizer.eval()
    
    # Initial parameters with codebooks of SpeechTokenizer
    sp_params = torch.load(st_ckpt, map_location='cpu')
    soundstorm.semantic_token_emb.weight = torch.nn.Parameter(sp_params['quantizer.vq.layers.0._codebook.embed'])
    acoustic_embeds = []
    for i in range(1, 8):
        acoustic_embed = torch.cat([sp_params[f'quantizer.vq.layers.{i}._codebook.embed'], torch.zeros(1,1024)], axis=0)
        acoustic_embeds.append(acoustic_embed)
    acoustic_embeds = torch.cat(acoustic_embeds, axis=0)
    soundstorm.net.code_embeds.weight = torch.nn.Parameter(acoustic_embeds)
    
    trainer = SoundStormTrainer(model=soundstorm,
                                cfg=cfg,
                                train_file_list=train_file_list,
                                valid_file_list=valid_file_list,
                                tokenizer=tokenizer,
                                )
    if args.continue_train:
        trainer.continue_train()
    else:
        trainer.train()
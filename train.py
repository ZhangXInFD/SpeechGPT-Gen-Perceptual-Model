from speechgpt_gen import ConditionalFlowMatcher, ConditionalFlowMatcherTrainer
from speechtokenizer import SpeechTokenizer
import yaml
import argparse


if __name__ == '__main__':
    
    # cnf_config = '/remote-home/xzhang/Speech/SpeechGPT-Gen-Flow-Matcher/config/uconformer_concat_cond/spt_snake_cfg.yaml'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, help='Config file path')
    parser.add_argument('--continue_train', action='store_true', help='Continue trainning from checkpoints')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cnf_model = ConditionalFlowMatcher(cfg=cfg.get('model_args'))
    
    
    
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
    # st_cfg = '/remote-home/xzhang/Speech/SpeechTokenizer/ckpt/config.json'
    # st_ckpt = '/remote-home/xzhang/Speech/SpeechTokenizer/ckpt/SpeechTokenizer.pt'
    tokenizer = SpeechTokenizer.load_from_checkpoint(st_cfg, st_ckpt)
    tokenizer.eval()
    
    trainer = ConditionalFlowMatcherTrainer(model=cnf_model,
                                cfg=cfg,
                                train_file_list=train_file_list,
                                valid_file_list=valid_file_list,
                                tokenizer=tokenizer,
                                )
    if args.continue_train:
        trainer.continue_train()
    else:
        trainer.train()
CONFIG="config/SoundStorm/SpeechTokenizer/spt_snake_cfg.yaml"


CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch scripts/soundstorm/spt/train.py \
    -c ${CONFIG}
    # --continue_train
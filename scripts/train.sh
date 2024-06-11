CONFIG="config/uconformer/spt_snake_cfg.yaml"


CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch scripts/train.py \
    -c ${CONFIG}\
    --continue_train
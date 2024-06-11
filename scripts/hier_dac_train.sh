CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch scripts/hier_dac_train.py \
    -c config/hier/hubert_dac.yaml\
    --continue_train
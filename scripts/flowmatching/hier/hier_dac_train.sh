CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch scripts/flowmatching/hier_dac_train.py \
    -c config/FlowMatching/hier/hubert_dac.yaml\
    --continue_train
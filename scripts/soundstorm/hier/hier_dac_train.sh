CONFIG="config/SoundStorm/Hier/hubert_dac.yaml"


CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch scripts/soundstorm/hier/hier_dac_train.py \
    -c ${CONFIG}
    # --continue_train
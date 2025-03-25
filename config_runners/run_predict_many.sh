# checkpoints/finetune_clean/balmy-violet-577_exp5_224_148/checkpoint-147476
# checkpoints/finetune_clean/hardy-bush-576_exp4_custom_one_src_token/checkpoint-73738

model="checkpoints/finetune_clean/youthful-wave-590_exp5_9M_448+296/checkpoint-294952"
CVD=0

echo "Processing model: $model"
CUDA_VISIBLE_DEVICES=$CVD python predict.py --checkpoint $model \
                                         --output-folder predictions \
                                         --config-file configs/predict_massbank_greedy.yaml &

CUDA_VISIBLE_DEVICES=$CVD python predict.py --checkpoint $model \
                                        --output-folder predictions \
                                        --config-file configs/predict_massbank_beam10.yaml &

wait
CUDA_VISIBLE_DEVICES=2 python spectus/train_spectus_grpo.py --config-file configs/finetune_HF_grpo_debug.yaml \
                                                --checkpoint checkpoints/finetune_clean/youthful-wave-590_exp5_9M_448+296/checkpoint-294952 \
                                                --additional-info _grpo \
                                                --additional-tags grpo:from_pretrained \
                                                --wandb-group grpo \
                                                --device cpu
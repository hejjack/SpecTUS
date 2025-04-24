python spectus/train_spectus_grpo.py --config-file configs/finetune_grpo_exp2_cosine_cosine.yaml \
                                    --checkpoint checkpoints/finetune_clean/youthful-wave-590_exp5_9M_448+296/checkpoint-294952 \
                                    --additional-info _grpo_exp2_cosine_cosine \
                                    --additional-tags grpo:exp2:from_pretrained:cosine_cosine \
                                    --wandb-group grpo

python spectus/train_spectus_grpo.py --config-file configs/finetune_grpo_exp2_linear_sine.yaml \
                                    --checkpoint checkpoints/finetune_clean/youthful-wave-590_exp5_9M_448+296/checkpoint-294952 \
                                    --additional-info _grpo_exp2_linear_sine \
                                    --additional-tags grpo:exp2:from_pretrained:linear_sine \
                                    --wandb-group grpo

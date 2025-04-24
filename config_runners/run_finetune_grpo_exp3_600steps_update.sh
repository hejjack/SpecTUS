python spectus/train_spectus_grpo.py --config-file configs/finetune_grpo_exp3_600steps_update.yaml \
                                    --checkpoint checkpoints/finetune_clean/youthful-wave-590_exp5_9M_448+296/checkpoint-294952 \
                                    --additional-info _grpo_exp3_600steps_update \
                                    --additional-tags grpo:exp3:from_pretrained:600steps_update \
                                    --wandb-group grpo

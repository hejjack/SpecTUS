python spectus/train_spectus_grpo.py --config-file configs/finetune_grpo_exp4_gs2.yaml \
                                    --checkpoint checkpoints/finetune_clean/youthful-wave-590_exp5_9M_448+296/checkpoint-294952 \
                                    --additional-info _grpo_exp4_gs2 \
                                    --additional-tags grpo:exp4:from_pretrained:gs2 \
                                    --wandb-group grpo

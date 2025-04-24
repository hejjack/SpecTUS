python spectus/train_spectus_grpo.py --config-file configs/finetune_HF_grpo_lr0.3.yaml \
                                    --checkpoint checkpoints/finetune_clean/youthful-wave-590_exp5_9M_448+296/checkpoint-294952 \
                                    --additional-info _grpo_lr0.3 \
                                    --additional-tags grpo:from_pretrained \
                                    --wandb-group grpo

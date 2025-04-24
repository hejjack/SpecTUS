CUDA_VISIBLE_DEVICES=1 python spectus/train_spectus_grpo.py --config-file configs/finetune_grpo_exp4_gs2.yaml \
                                    --checkpoint checkpoints/finetune_clean/youthful-wave-590_exp5_9M_448+296/checkpoint-294952 \
                                    --additional-info _grpo_exp5_exact_reward_1.3 \
                                    --additional-tags grpo:exp5:from_pretrained:exact_reward_1.3 \
                                    --wandb-group grpo

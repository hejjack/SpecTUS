python spectus/my_grpo/grpo_summarization.py --config-file configs/finetune_grpo.yaml \
                     --checkpoint checkpoints/finetune_clean/youthful-wave-590_exp5_9M_448+296/checkpoint-294952 \
                     --additional-info _grpo \
                     --additional-tags grpo:from_pretrained \
                     --wandb-group debug
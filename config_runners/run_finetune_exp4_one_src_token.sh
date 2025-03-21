python spectus/train_spectus.py --config-file configs/finetune_exp2_mf10M.yaml \
                        --checkpoint checkpoints/pretrain_clean/unearthly-imp-575_exp4_custom_one_src_token/checkpoint-112000 \
                        --additional-info "_exp4_custom_one_src_token" \
                        --wandb-group finetune_clean  \
                        --additional-tags "exp4:nist:from_pretrained" \

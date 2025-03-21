python spectus/train_spectus.py --config-file configs/finetune_exp2_mf10M.yaml \
                     --checkpoint checkpoints/pretrain_clean/grateful-field-566_exp3_custom_neims/checkpoint-112000 \
                     --additional-info "_exp3_custom_neims" \
                     --wandb-group finetune_clean  \
                     --additional-tags "exp3:neims_custom:from_pretrained" \

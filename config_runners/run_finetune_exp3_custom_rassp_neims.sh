python spectus/train_spectus.py --config-file configs/finetune_exp2_mf10M.yaml \
                     --checkpoint checkpoints/pretrain_clean/sandy-star-569_exp3_custom_rassp_neims/checkpoint-112000 \
                     --additional-info "_exp3_custom_rassp_neims" \
                     --wandb-group finetune_clean  \
                     --additional-tags "exp3:custom_rassp:custom_neims:from_pretrained" \

python spectus/train_spectus.py --config-file configs/finetune_exp2_mf10M.yaml \
                     --checkpoint checkpoints/pretrain_clean/drawn-fire-571_exp3_custom_rassp_neims_nist/checkpoint-112000 \
                     --additional-info "_exp3_custom_rassp_neims_nist" \
                     --wandb-group finetune_clean  \
                     --additional-tags "exp3:custom_rassp:custom_neims:from_pretrained" \

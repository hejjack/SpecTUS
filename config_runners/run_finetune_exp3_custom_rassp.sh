python spectus/train_spectus.py --config-file configs/finetune_exp2_mf10M.yaml \
                     --checkpoint checkpoints/pretrain_clean/golden-morning-569_exp3_custom_rassp/checkpoint-112000 \
                     --additional-info "_exp3_custom_rassp" \
                     --wandb-group finetune_clean  \
                     --additional-tags "exp3:custom_rassp:from_pretrained" \

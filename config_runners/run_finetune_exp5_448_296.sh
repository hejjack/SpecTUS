python spectus/train_spectus.py --config-file configs/finetune_exp5_448_296.yaml \
                     --checkpoint checkpoints/pretrain_clean/fresh-universe-588_exp5_448/checkpoint-448000 \
                     --additional-info _exp5_9M_448+296 \
                     --additional-tags log_29_1.28:mf10M:nist:from_pretrained:exp5:296k \
                     --wandb-group finetune_clean

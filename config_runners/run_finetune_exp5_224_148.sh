python spectus/train_spectus.py --config-file configs/finetune_exp5_224_148.yaml \
                     --checkpoint checkpoints/pretrain_clean/helpful-star-588_exp5_112_112/checkpoint-224000 \
                     --additional-info _exp5_9M_224+148 \
                     --additional-tags log_29_1.28:mf10M:nist:from_pretrained:exp5:148k \
                     --wandb-group finetune_clean

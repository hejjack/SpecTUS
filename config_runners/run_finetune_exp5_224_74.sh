CUDA_VISIBLE_DEVICES=2 python spectus/train_spectus.py --config-file configs/finetune_exp2_mf10M.yaml \
                                            --checkpoint checkpoints/pretrain_clean/sandy-star-569_exp3_custom_rassp_neims/checkpoint-224000 \
                                            --additional-info "_exp5_224_74" \
                                            --wandb-group finetune_clean  \
                                            --additional-tags "exp5:rassp:neims:from_pretrained:224_74" \

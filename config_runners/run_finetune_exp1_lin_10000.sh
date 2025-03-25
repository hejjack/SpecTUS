CUDA_VISIBLE_DEVICES=2  python spectus/train_spectus.py --config-file configs/finetune_exp1_lin_10000.yaml \
                                            --additional-info _exp1_lin_10000 \
                                            --additional-tags exp1:lin_10000:from_scratch \
                                            --wandb-group finetune_clean
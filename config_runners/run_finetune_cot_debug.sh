CUDA_VISIBLE_DEVICES=1 python spectus/train_spectus.py \
                            --config-file configs/finetune_cot_debug.yaml \
                            --additional-info _cot_debug \
                            --additional-tags cot:debug \
                            --wandb-group debug
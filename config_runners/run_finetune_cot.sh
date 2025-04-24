python spectus/train_spectus.py --config-file configs/finetune_cot.yaml \
                     --checkpoint checkpoints/pretrain_clean/sweet-dawn-604_cot_112k/checkpoint-112000 \
                     --additional-info _cot_112k_74k \
                     --additional-tags cot:from_pretrained \
                     --wandb-group finetune_clean
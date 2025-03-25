# pretrain RASSP_1 NEIMS_1
python spectus/train_spectus.py --config-file configs/pretrain_exp5_448.yaml \
                     --additional-info "_exp5_448" \
                     --additional-tags "custom_rassp:custom_neims:custom_old_rassp:custom_old_neims:from_scratch" \
                     --wandb-group pretrain_clean \

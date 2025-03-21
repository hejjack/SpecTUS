# pretrain RASSP_1 NEIMS_1
python spectus/train_spectus.py --config-file configs/pretrain_exp5_224.yaml \
                     --additional-info "_exp5_112_112" \
                     --additional-tags "custom_rassp:custom_neims:custom_old_rassp:custom_old_neims:from_scratch" \
                     --wandb-group pretrain_clean \
                     --resume-id nzj1j8x2 \
                     --checkpoint checkpoints/pretrain_clean/helpful-star-588_exp5_112_112/checkpoint-112000 \

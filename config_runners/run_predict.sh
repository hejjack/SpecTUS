CUDA_VISIBLE_DEVICES=2 python spectus/predict.py --checkpoint checkpoints/finetune_clean/youthful-wave-590_exp5_9M_448+296/checkpoint-294952 \
                                          --output-folder predictions_debug \
                                          --config-file configs/predict_nist_test_beam10.yaml \

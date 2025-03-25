model=checkpoints/finetune_clean/youthful-wave-590_exp5_9M_448+296/checkpoint-294952
configs=(nist_test_greedy \
          nist_test_beam10 \
          nist_test_beam50 \
          nist_valid_greedy \
          nist_valid_beam10 \
)

for config in ${configs[@]}; do
    python predict.py --checkpoint $model \
                      --output-folder predictions \
                      --config-file configs/predict_${config}.yaml &

done

wait

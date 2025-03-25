model=checkpoints/finetune_clean/youthful-wave-590_exp5_9M_448+296/checkpoint-294952
# libnames=(cayman \
#           mona \
#           rcx_inhouse \
#           swgdrug \
# )

libnames=(mona_oo \
          rcx_inhouse_oo \
)

for libname in ${libnames[@]}; do
    python predict.py --checkpoint $model \
                      --output-folder predictions \
                      --config-file configs/predict_${libname}_beam10.yaml &

    python predict.py --checkpoint $model \
                      --output-folder predictions \
                      --config-file configs/predict_${libname}_beam50.yaml &

    python predict.py --checkpoint $model \
                      --output-folder predictions \
                      --config-file configs/predict_${libname}_greedy.yaml &
done

wait

CVD=2
model=checkpoints/finetune_clean/youthful-wave-590_exp5_9M_448+296/checkpoint-294952
libnames=(cayman \
          mona \
          rcx_inhouse \
          swgdrug \
)


for libname in ${libnames[@]}; do
    CUDA_VISIBLE_DEVICES=$CVD python spectus/predict.py --checkpoint $model \
                                              --output-folder predictions \
                                              --config-file configs/predict_${libname}_beam10.yaml &

    CUDA_VISIBLE_DEVICES=$CVD python spectus/predict.py --checkpoint $model \
                                              --output-folder predictions \
                                              --config-file configs/predict_${libname}_beam50.yaml &

    CUDA_VISIBLE_DEVICES=$CVD python spectus/predict.py --checkpoint $model \
                                              --output-folder predictions \
                                              --config-file configs/predict_${libname}_greedy.yaml &
done

wait
MODEL_NAMES=(youthful-wave-590_exp5_9M_448+296)

# Correct for loop with proper array reference
for MODEL_NAME in ${MODEL_NAMES[@]} ; do
    for prediction in `find predictions/$MODEL_NAME -type f -name "predictions.jsonl"` ; do
        echo "Processing prediction: `wc -l $prediction`"
        python evaluate_predictions.py --predictions-path $prediction \
                                        --config-file configs/evaluate.yaml &
    done
done


# Optional: Wait for all background processes to finish
wait
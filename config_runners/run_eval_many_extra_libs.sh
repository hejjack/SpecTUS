MODEL=youthful-wave-590_exp5_9M_448+296
# LIBS=(Cayman_library MONA_GCMS RCX_NO SWGDRUG)
LIBS=(MassBank)
GEN_METHODS=(greedy beam10)

# Correct for loop with proper array reference
lib_index=-1
for lib in ${LIBS[@]} ; do
    lib_index=$((lib_index+1))
    for gen_method in ${GEN_METHODS[@]} ; do
                # resolve the predictions path
        prediction=`find -name predictions.jsonl -path "./predictions/$MODEL/${lib}/*$gen_method*/*"`
        echo "Search command: $search_command"
        echo "Processing $lib $gen_method"
        echo "Processing prediction: $prediction"

        python spectus/evaluate_predictions.py --predictions-path $prediction \
                                          --config-file configs/evaluate.yaml
    done
done

# Optional: Wait for all background processes to finish
wait
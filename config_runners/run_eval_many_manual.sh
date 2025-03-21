# ./predictions/db_search_morgan_tanimoto/NIST/1731407440_test_full_50cand/predictions.jsonl
# ./predictions/db_search_morgan_tanimoto/NIST/1731490929_test_full_1cand/predictions.jsonl
# ./predictions/db_search_morgan_tanimoto/NIST/1731490894_test_full_10cand/predictions.jsonl
# ./predictions/db_search_sss/NIST/1731407440_test_full_50cand/predictions.jsonl
# ./predictions/db_search_sss/NIST/1731490929_test_full_1cand/predictions.jsonl
# ./predictions/db_search_sss/NIST/1731490894_test_full_10cand/predictions.jsonl
# ./predictions/db_search_hss/NIST/1731407440_test_full_50cand/predictions.jsonl
# ./predictions/db_search_hss/NIST/1731490929_test_full_1cand/predictions.jsonl
# ./predictions/db_search_hss/NIST/1731490894_test_full_10cand/predictions.jsonl

PREDICTIONS=(./predictions/db_search_morgan_tanimoto/NIST/1731407440_test_full_50cand/predictions.jsonl
./predictions/db_search_morgan_tanimoto/NIST/1731490929_test_full_1cand/predictions.jsonl
./predictions/db_search_morgan_tanimoto/NIST/1731490894_test_full_10cand/predictions.jsonl
./predictions/db_search_sss/NIST/1731407440_test_full_50cand/predictions.jsonl
./predictions/db_search_sss/NIST/1731490929_test_full_1cand/predictions.jsonl
./predictions/db_search_sss/NIST/1731490894_test_full_10cand/predictions.jsonl
./predictions/db_search_hss/NIST/1731407440_test_full_50cand/predictions.jsonl
./predictions/db_search_hss/NIST/1731490929_test_full_1cand/predictions.jsonl
./predictions/db_search_hss/NIST/1731490894_test_full_10cand/predictions.jsonl)


# Correct for loop with proper array reference
for prediction in "${PREDICTIONS[@]}"; do
    echo "Processing prediction: $prediction `wc -l $prediction`"
    python spectus/evaluate_predictions.py --predictions-path $prediction \
                                    --labels-path data/nist/test.jsonl \
                                    --config-file configs/evaluate.yaml &
done

# Optional: Wait for all background processes to finish
wait
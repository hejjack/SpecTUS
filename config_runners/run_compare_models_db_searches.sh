# ./predictions/db_search_morgan_tanimoto/NIST/1731407440_test_full_50cand/predictions.jsonl
# ./predictions/db_search_morgan_tanimoto/NIST/1731490929_test_full_1cand/predictions.jsonl
# ./predictions/db_search_morgan_tanimoto/NIST/1731490894_test_full_10cand/predictions.jsonl
# ./predictions/db_search_sss/NIST/1731407440_test_full_50cand/predictions.jsonl
# ./predictions/db_search_sss/NIST/1731490929_test_full_1cand/predictions.jsonl
# ./predictions/db_search_sss/NIST/1731490894_test_full_10cand/predictions.jsonl
# ./predictions/db_search_hss/NIST/1731407440_test_full_50cand/predictions.jsonl
# ./predictions/db_search_hss/NIST/1731490929_test_full_1cand/predictions.jsonl
# ./predictions/db_search_hss/NIST/1731490894_test_full_10cand/predictions.jsonl

python spectus/compare_models.py \
                --additional-info db_searches_test \
                --db-search-prediction-paths \
                "./predictions/db_search_morgan_tanimoto/NIST/1731407440_test_full_50cand \
                ./predictions/db_search_morgan_tanimoto/NIST/1731490929_test_full_1cand \
                ./predictions/db_search_morgan_tanimoto/NIST/1731490894_test_full_10cand \
                ./predictions/db_search_sss/NIST/1731407440_test_full_50cand \
                ./predictions/db_search_sss/NIST/1731490929_test_full_1cand \
                ./predictions/db_search_sss/NIST/1731490894_test_full_10cand \
                ./predictions/db_search_hss/NIST/1731407440_test_full_50cand \
                ./predictions/db_search_hss/NIST/1731490929_test_full_1cand \
                ./predictions/db_search_hss/NIST/1731490894_test_full_10cand"

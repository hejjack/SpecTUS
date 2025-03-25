
python spectus/compare_models.py \
                --additional-info all_balmy_test \
                --models-prediction-paths \
                "./predictions/balmy-violet-577_exp5_224_148/NIST/1730720158_test_full_greedy \
                ./predictions/balmy-violet-577_exp5_224_148/NIST/1730754395_test_full_beam50 \
                ./predictions/balmy-violet-577_exp5_224_148/NIST/1730720159_test_full_beam10" \
                --db-search-prediction-paths \
                "./predictions/db_search_morgan_tanimoto/NIST/1731490929_test_full_1cand \
                ./predictions/db_search_sss/NIST/1731407440_test_full_50cand \
                ./predictions/db_search_sss/NIST/1731490929_test_full_1cand \
                ./predictions/db_search_sss/NIST/1731490894_test_full_10cand \
                ./predictions/db_search_hss/NIST/1731407440_test_full_50cand \
                ./predictions/db_search_hss/NIST/1731490929_test_full_1cand \
                ./predictions/db_search_hss/NIST/1731490894_test_full_10cand"

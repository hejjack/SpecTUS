
python spectus/compare_models.py \
                --additional-info comparison_debug \
                --models-prediction-paths \
                "./predictions/youthful-wave-590_exp5_9M_448+296/NIST/1736157156_valid_full_greedy" \
                --db-search-prediction-paths \
                "./predictions/db_search_hss/NIST/1729676322_valid_full_1cand \
                ./predictions/db_search_morgan_tanimoto/NIST/1729630748_valid_full_1cand \
                ./predictions/db_search_sss/NIST/1729676364_valid_full_1cand"

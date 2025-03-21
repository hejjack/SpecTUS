#  NIST
python spectus/ctions.py --predictions-path predictions/youthful-wave-590_exp5_9M_448+296/Cayman_library/1736109908_all_full_greedy/predictions.jsonl \
                                       --config-file configs/evaluate.yaml

#  db search
# python ctions.py --predictions-path predictions/db_search_morgan_tanimoto/NIST/1729630748_valid_full_1cand/predictions.jsonl \
#                                --config-file configs/evaluate.yaml
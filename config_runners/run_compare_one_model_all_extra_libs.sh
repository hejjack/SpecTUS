MODEL=youthful-wave-590_exp5_9M_448+296
LIBS=(Cayman_library MONA_GCMS RCX_NO RCX_OO MONA_GCMS_overlaps_only SWGDRUG)
# DB_SEARCHES=(db_search_hss db_search_morgan_tanimoto db_search_sss)
# cands=(1 10 50)

# for each library and each num of candidates compare the model with all db_searches
for lib in ${LIBS[@]}; do
    model_predictions=`find -maxdepth 4 -path "./predictions/$MODEL/$lib/*"`
    db_search_predictions=`find -maxdepth 4 -regex "./predictions/db_search_\(sss\|hss\)/$lib/.*"`              # SSS, HSS
    db_search_morgan_tanimoto=`find -maxdepth 4 -path "./predictions/db_search_morgan_tanimoto/$lib/*1cand*"`   # Morgan Tanimoto
    echo "Processing $lib"
    echo "Processing model prediction: $model_predictions"
    echo "Processing db search prediction: $db_search_predictions"
    python spectus/compare_models.py \
        --additional-info "" \
        --models-prediction-paths "$model_predictions" \
        --db-search-prediction-paths "$db_search_morgan_tanimoto \
                                     $db_search_predictions" \
        --save-path ./results/$MODEL &
done

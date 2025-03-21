DB_TYPES=("hss" "sss" "mt")

for db_type in "${DB_TYPES[@]}"
do
    python spectus/predict_db_search.py --output-folder predictions \
                                   --config-file configs/predict_db_search_${db_type}_extra_libs.yaml \
                                   --num-workers 32 &
done
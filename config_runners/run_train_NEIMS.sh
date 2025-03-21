# this script is used to run the NEIMS model training according to the original paper
# please prepare your data according to the tutorial in the deep-molecular-massspec/Model_Retrain_Quickstart.md
# before running this script

# change the default parameter 'num_hidden_layers' to 7 in molecule_predictors.py
# and gate_bidirectional_predictions to True in molecule_predictors.py too.



TARGET_PATH_NAME=tmp/massspec_predictions
cd deep-molecular-massspec
CUDA_VISIBLE_DEVICES=0 python molecule_estimator.py --dataset_config_file=$TARGET_PATH_NAME/spectra_tf_records/query_replicates_val_predicted_replicates_val.json \
                                                    --train_steps=100000 \
                                                    --model_dir=$TARGET_PATH_NAME/models/output \
                                                    --hparams=make_spectra_plots=True,batch_size=100,num_hidden_layers=7,gate_bidirectional_predictions=True \
                                                    --alsologtostderr \
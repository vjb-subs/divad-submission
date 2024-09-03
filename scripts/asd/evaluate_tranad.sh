#!/bin/bash

# import utility functions
source ../utils.sh
source utils.sh

dataset_name=asd
detector_name=tranad
step_name=evaluate_online_scorer

# make_datasets args
dataset=$1
app_ids=$2
test_app_ids=$3
val_prop=$4

# build_features args
transform_chain=$5
regular_scaling_method=$6
transform_fit_normal_only=yes

# make_window_datasets args
window_size=$7
window_step=$8
downsampling_size=$9
downsampling_step=${10}
downsampling_func=${11}
asd_balancing="none"  # never balance as we need sequence information
normal_data_prop="1.0"
anomaly_augmentation="none"
ano_augment_n_per_normal="1.0"

# train_window_model args
dim_feedforward=${12}
last_activation=${13}
optimizer=${14}
adamw_weight_decay=${15}
learning_rate=${16}
batch_size=${17}
n_epochs=${18}
early_stopping_target=${19}
early_stopping_patience=${20}

# evaluate_online_scorer args
ignored_anomaly_labels=""
ignored_delayed_window=$((window_size-1))

make_datasets_id=$(get_make_datasets_id "$dataset" "$app_ids" "$test_app_ids" "$val_prop")
echo "make_datasets_id: ${make_datasets_id}"
build_features_id=$(get_build_features_id "$transform_chain" "$regular_scaling_method" "$transform_fit_normal_only")
echo "build_features_id: ${build_features_id}"
make_window_datasets_id=$(get_make_window_datasets_id "$window_size" "$window_step" "$asd_balancing" "$normal_data_prop" \
"$anomaly_augmentation" "$ano_augment_n_per_normal" "$downsampling_size" "$downsampling_step" "$downsampling_func")
echo "make_window_datasets_id: ${make_window_datasets_id}"
train_window_model_id=$(get_train_window_model_id "$detector_name" "$dim_feedforward" "$last_activation" "$optimizer" \
"$adamw_weight_decay" "$learning_rate" "$batch_size" "$n_epochs" "$early_stopping_target" "$early_stopping_patience")
echo "train_window_model_id: ${train_window_model_id}"
evaluate_online_scorer_id=$(get_evaluate_online_scorer_id "$ignored_anomaly_labels" "$ignored_delayed_window")
echo "evaluate_online_scorer_id: ${evaluate_online_scorer_id}"

for scores_avg_beta in 0 0.8 0.9 0.95 0.96667 0.975 0.98 0.98333 0.9875 0.99167 0.99375 0.995; do
  train_online_scorer_id=$(get_train_online_scorer_id "$scores_avg_beta")
  echo "train_online_scorer_id: ${train_online_scorer_id}"
  exathlon \
  dataset="$dataset_name" \
  detector="$detector_name" \
  step="$step_name" \
  dataset.make_datasets.step_id="$make_datasets_id" \
  dataset.make_datasets.data_manager.dataset="$dataset" \
  dataset.make_datasets.data_manager.app_ids="$app_ids" \
  dataset.make_datasets.data_manager.test_app_ids="$test_app_ids" \
  dataset.make_datasets.data_manager.val_prop="$val_prop" \
  dataset.build_features.step_id="$build_features_id" \
  dataset.build_features.transform_chain.value="$transform_chain" \
  dataset.build_features.transform_chain.contains__regular_scaling.regular_scaling.type_.value="$regular_scaling_method" \
  dataset.build_features.transform_fit_normal_only="$transform_fit_normal_only" \
  detector.make_window_datasets.step_id="$make_window_datasets_id" \
  detector.make_window_datasets.window_manager.window_size="$window_size" \
  detector.make_window_datasets.window_manager.window_step="$window_step" \
  detector.make_window_datasets.window_manager.downsampling_size="$downsampling_size" \
  detector.make_window_datasets.window_manager.downsampling_step="$downsampling_step" \
  detector.make_window_datasets.window_manager.downsampling_func="$downsampling_func" \
  detector.make_window_datasets.window_manager.dataset_name.value="$dataset_name" \
  detector.make_window_datasets.window_manager.dataset_name.eq__asd.asd_balancing="$asd_balancing" \
  detector.make_window_datasets.window_manager.normal_data_prop.value="$normal_data_prop" \
  detector.make_window_datasets.window_manager.anomaly_augmentation.value="$anomaly_augmentation" \
  detector.make_window_datasets.window_manager.anomaly_augmentation.ne__none.ano_augment_n_per_normal="$ano_augment_n_per_normal" \
  detector.train_window_model.step_id="$train_window_model_id" \
  detector.train_window_model.dim_feedforward="$dim_feedforward" \
  detector.train_window_model.last_activation="$last_activation" \
  detector.train_window_model.optimizer.value="$optimizer" \
  detector.train_window_model.optimizer.eq__adamw.adamw_weight_decay="$adamw_weight_decay" \
  detector.train_window_model.learning_rate="$learning_rate" \
  detector.train_window_model.batch_size="$batch_size" \
  detector.train_window_model.n_epochs="$n_epochs" \
  detector.train_window_model.early_stopping_target="$early_stopping_target" \
  detector.train_window_model.early_stopping_patience="$early_stopping_patience" \
  detector.train_online_scorer.step_id="$train_online_scorer_id" \
  detector.train_online_scorer.scores_avg_beta="$scores_avg_beta" \
  detector.evaluate_online_scorer.step_id="$evaluate_online_scorer_id" \
  detector.evaluate_online_scorer.evaluator.ignored_anomaly_labels="$ignored_anomaly_labels" \
  detector.evaluate_online_scorer.evaluator.ignored_delayed_window="$ignored_delayed_window"
done
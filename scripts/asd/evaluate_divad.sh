#!/bin/bash

# import utility functions
source ../utils.sh
source utils.sh

dataset_name=asd
detector_name=divad
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
asd_balancing=${12}
normal_data_prop="1.0"
anomaly_augmentation="none"
ano_augment_n_per_normal="1.0"

# train_window_model args
type_=${13}
pzy_dist=${14}
pzy_kl_n_samples=${15}
pzy_gm_n_components=${16}
pzy_gm_softplus_scale=${17}
pzy_vamp_n_components=${18}
qz_x_conv1d_filters=${19}
qz_x_conv1d_kernel_sizes=${20}
qz_x_conv1d_strides=${21}
qz_x_n_hidden=${22}
pzd_d_n_hidden=${23}
px_z_conv1d_filters=${24}
px_z_conv1d_kernel_sizes=${25}
px_z_conv1d_strides=${26}
px_z_n_hidden=${27}
time_freq=${28}
sample_normalize_x=${29}
sample_normalize_mag=${30}
apply_hann=${31}
n_freq_modes=${32}
phase_encoding=${33}
phase_cyclical_decoding=${34}
qz_x_freq_conv1d_filters=${35}
qz_x_freq_conv1d_kernel_sizes=${36}
qz_x_freq_conv1d_strides=${37}
px_z_freq_conv1d_filters=${38}
px_z_freq_conv1d_kernel_sizes=${39}
px_z_freq_conv1d_strides=${40}
latent_dim=${41}
rec_unit_type=${42}
activation_rec=${43}
conv1d_pooling=${44}
conv1d_batch_norm=${45}
rec_weight_decay=${46}
weight_decay=${47}
dropout=${48}
dec_output_dist=${49}
min_beta=${50}
max_beta=${51}
beta_n_epochs=${52}
loss_weighting=${53}
d_classifier_weight=${54}
optimizer=${55}
learning_rate=${56}
lr_scheduling=${57}  # either "none", "pw_constant" or "one_cycle"
lrs_pwc_red_factor=${58}
lrs_pwc_red_freq=${59}
adamw_weight_decay=${60}
softplus_scale=${61}
batch_size=${62}
n_epochs=${63}
early_stopping_target=${64}
early_stopping_patience=${65}
domain_key="file_name"

# train_window_scorer args
scoring_method=${66}
agg_post_dist=${67}
agg_post_gm_n_components=${68}
mean_nll_n_samples=${69}
scores_avg_beta=${70}

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
train_window_model_id=$(get_train_window_model_id "$detector_name" "$type_" "$pzy_dist" "$pzy_kl_n_samples" "$pzy_gm_n_components" "$pzy_gm_softplus_scale" \
"$pzy_vamp_n_components" "$qz_x_conv1d_filters" "$qz_x_conv1d_kernel_sizes" "$qz_x_conv1d_strides" "$qz_x_n_hidden" "$pzd_d_n_hidden" "$px_z_conv1d_filters" \
"$px_z_conv1d_kernel_sizes" "$px_z_conv1d_strides" "$px_z_n_hidden" "$time_freq" "$sample_normalize_x" "$sample_normalize_mag" "$apply_hann" \
"$n_freq_modes" "$phase_encoding" "$phase_cyclical_decoding" "$qz_x_freq_conv1d_filters" "$qz_x_freq_conv1d_kernel_sizes" "$qz_x_freq_conv1d_strides" \
"$px_z_freq_conv1d_filters" "$px_z_freq_conv1d_kernel_sizes" "$px_z_freq_conv1d_strides" "$latent_dim" "$rec_unit_type" "$activation_rec" \
"$conv1d_pooling" "$conv1d_batch_norm" "$rec_weight_decay" "$weight_decay" "$dropout" "$dec_output_dist" "$min_beta" "$max_beta" "$beta_n_epochs" \
"$loss_weighting" "$d_classifier_weight" "$optimizer" "$learning_rate" "$lr_scheduling" "$lrs_pwc_red_factor" "$lrs_pwc_red_freq" \
"$adamw_weight_decay" "$softplus_scale" "$batch_size" "$n_epochs" "$early_stopping_target" "$early_stopping_patience")
echo "train_window_model_id: ${train_window_model_id}"
train_window_scorer_id=$(get_train_window_scorer_id "$detector_name" "$scoring_method" "$agg_post_dist" \
"$agg_post_gm_n_components" "$mean_nll_n_samples")
echo "train_window_scorer_id: ${train_window_scorer_id}"
evaluate_online_scorer_id=$(get_evaluate_online_scorer_id "$ignored_anomaly_labels" "$ignored_delayed_window")
echo "evaluate_online_scorer_id: ${evaluate_online_scorer_id}"

if [ "$scores_avg_beta" == "-1" ] || [ "$scores_avg_beta" == "-1.0" ]; then
  for scores_avg_beta in 0 0.8 0.9 0.95 0.96667 0.975 0.98 0.98333 0.9875 0.99167 0.99375 0.995; do
    # train_online_scorer_id=$(get_train_online_scorer_id "$scores_avg_beta")
    train_online_scorer_id="b$scores_avg_beta"
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
    detector.train_window_model.type_.value="$type_" \
    detector.train_window_model.domain_key="$domain_key" \
    detector.train_window_model.pzy_dist.value="$pzy_dist" \
    detector.train_window_model.pzy_dist.ne__standard.pzy_kl_n_samples="$pzy_kl_n_samples" \
    detector.train_window_model.pzy_dist.eq__gm.pzy_gm_n_components="$pzy_gm_n_components" \
    detector.train_window_model.pzy_dist.eq__gm.pzy_gm_softplus_scale="$pzy_gm_softplus_scale" \
    detector.train_window_model.pzy_dist.eq__vamp.pzy_vamp_n_components="$pzy_vamp_n_components" \
    detector.train_window_model.qz_x_conv1d_filters="$qz_x_conv1d_filters" \
    detector.train_window_model.qz_x_conv1d_kernel_sizes="$qz_x_conv1d_kernel_sizes" \
    detector.train_window_model.qz_x_conv1d_strides="$qz_x_conv1d_strides" \
    detector.train_window_model.qz_x_n_hidden="$qz_x_n_hidden" \
    detector.train_window_model.pzd_d_n_hidden="$pzd_d_n_hidden" \
    detector.train_window_model.px_z_conv1d_filters="$px_z_conv1d_filters" \
    detector.train_window_model.px_z_conv1d_kernel_sizes="$px_z_conv1d_kernel_sizes" \
    detector.train_window_model.px_z_conv1d_strides="$px_z_conv1d_strides" \
    detector.train_window_model.px_z_n_hidden="$px_z_n_hidden" \
    detector.train_window_model.time_freq.value="$time_freq" \
    detector.train_window_model.time_freq.eq__True.sample_normalize_x="$sample_normalize_x" \
    detector.train_window_model.time_freq.eq__True.sample_normalize_mag="$sample_normalize_mag" \
    detector.train_window_model.time_freq.eq__True.apply_hann="$apply_hann" \
    detector.train_window_model.time_freq.eq__True.n_freq_modes="$n_freq_modes" \
    detector.train_window_model.time_freq.eq__True.phase_encoding="$phase_encoding" \
    detector.train_window_model.time_freq.eq__True.phase_cyclical_decoding="$phase_cyclical_decoding" \
    detector.train_window_model.time_freq.eq__True.qz_x_freq_conv1d_filters="$qz_x_freq_conv1d_filters" \
    detector.train_window_model.time_freq.eq__True.qz_x_freq_conv1d_kernel_sizes="$qz_x_freq_conv1d_kernel_sizes" \
    detector.train_window_model.time_freq.eq__True.qz_x_freq_conv1d_strides="$qz_x_freq_conv1d_strides" \
    detector.train_window_model.time_freq.eq__True.px_z_freq_conv1d_filters="$px_z_freq_conv1d_filters" \
    detector.train_window_model.time_freq.eq__True.px_z_freq_conv1d_kernel_sizes="$px_z_freq_conv1d_kernel_sizes" \
    detector.train_window_model.time_freq.eq__True.px_z_freq_conv1d_strides="$px_z_freq_conv1d_strides" \
    detector.train_window_model.latent_dim="$latent_dim" \
    detector.train_window_model.type_.eq__rec.rec_unit_type="$rec_unit_type" \
    detector.train_window_model.type_.eq__rec.activation_rec="$activation_rec" \
    detector.train_window_model.type_.eq__rec.conv1d_pooling="$conv1d_pooling" \
    detector.train_window_model.type_.eq__rec.conv1d_batch_norm="$conv1d_batch_norm" \
    detector.train_window_model.type_.eq__rec.rec_weight_decay="$rec_weight_decay" \
    detector.train_window_model.weight_decay="$weight_decay" \
    detector.train_window_model.dropout="$dropout" \
    detector.train_window_model.dec_output_dist="$dec_output_dist" \
    detector.train_window_model.min_beta="$min_beta" \
    detector.train_window_model.max_beta="$max_beta" \
    detector.train_window_model.beta_n_epochs="$beta_n_epochs" \
    detector.train_window_model.loss_weighting.value="$loss_weighting" \
    detector.train_window_model.loss_weighting.eq__fixed.d_classifier_weight="$d_classifier_weight" \
    detector.train_window_model.learning_rate="$learning_rate" \
    detector.train_window_model.lr_scheduling="$lr_scheduling" \
    detector.train_window_model.lrs_pwc_red_factor="$lrs_pwc_red_factor" \
    detector.train_window_model.lrs_pwc_red_freq="$lrs_pwc_red_freq" \
    detector.train_window_model.optimizer.value="$optimizer" \
    detector.train_window_model.optimizer.eq__adamw.adamw_weight_decay="$adamw_weight_decay" \
    detector.train_window_model.softplus_scale="$softplus_scale" \
    detector.train_window_model.batch_size="$batch_size" \
    detector.train_window_model.n_epochs="$n_epochs" \
    detector.train_window_model.early_stopping_target="$early_stopping_target" \
    detector.train_window_model.early_stopping_patience="$early_stopping_patience" \
    detector.train_window_scorer.step_id="$train_window_scorer_id" \
    detector.train_window_scorer.scoring_method.value="$scoring_method" \
    detector.train_window_scorer.scoring_method.contains__agg_post.agg_post_dist.value="$agg_post_dist" \
    detector.train_window_scorer.scoring_method.contains__agg_post.agg_post_dist.eq__gm.agg_post_gm_n_components="$agg_post_gm_n_components" \
    detector.train_window_scorer.scoring_method.contains__mean_nll.mean_nll_n_samples="$mean_nll_n_samples" \
    detector.train_online_scorer.step_id="$train_online_scorer_id" \
    detector.train_online_scorer.scores_avg_beta="$scores_avg_beta" \
    detector.evaluate_online_scorer.step_id="$evaluate_online_scorer_id" \
    detector.evaluate_online_scorer.evaluator.ignored_anomaly_labels="$ignored_anomaly_labels" \
    detector.evaluate_online_scorer.evaluator.ignored_delayed_window="$ignored_delayed_window"
  done
else
  # train_online_scorer_id=$(get_train_online_scorer_id "$scores_avg_beta")
  train_online_scorer_id="b$scores_avg_beta"
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
  detector.train_window_model.type_.value="$type_" \
  detector.train_window_model.domain_key="$domain_key" \
  detector.train_window_model.pzy_dist.value="$pzy_dist" \
  detector.train_window_model.pzy_dist.ne__standard.pzy_kl_n_samples="$pzy_kl_n_samples" \
  detector.train_window_model.pzy_dist.eq__gm.pzy_gm_n_components="$pzy_gm_n_components" \
  detector.train_window_model.pzy_dist.eq__gm.pzy_gm_softplus_scale="$pzy_gm_softplus_scale" \
  detector.train_window_model.pzy_dist.eq__vamp.pzy_vamp_n_components="$pzy_vamp_n_components" \
  detector.train_window_model.qz_x_conv1d_filters="$qz_x_conv1d_filters" \
  detector.train_window_model.qz_x_conv1d_kernel_sizes="$qz_x_conv1d_kernel_sizes" \
  detector.train_window_model.qz_x_conv1d_strides="$qz_x_conv1d_strides" \
  detector.train_window_model.qz_x_n_hidden="$qz_x_n_hidden" \
  detector.train_window_model.pzd_d_n_hidden="$pzd_d_n_hidden" \
  detector.train_window_model.px_z_conv1d_filters="$px_z_conv1d_filters" \
  detector.train_window_model.px_z_conv1d_kernel_sizes="$px_z_conv1d_kernel_sizes" \
  detector.train_window_model.px_z_conv1d_strides="$px_z_conv1d_strides" \
  detector.train_window_model.px_z_n_hidden="$px_z_n_hidden" \
  detector.train_window_model.time_freq.value="$time_freq" \
  detector.train_window_model.time_freq.eq__True.sample_normalize_x="$sample_normalize_x" \
  detector.train_window_model.time_freq.eq__True.sample_normalize_mag="$sample_normalize_mag" \
  detector.train_window_model.time_freq.eq__True.apply_hann="$apply_hann" \
  detector.train_window_model.time_freq.eq__True.n_freq_modes="$n_freq_modes" \
  detector.train_window_model.time_freq.eq__True.phase_encoding="$phase_encoding" \
  detector.train_window_model.time_freq.eq__True.phase_cyclical_decoding="$phase_cyclical_decoding" \
  detector.train_window_model.time_freq.eq__True.qz_x_freq_conv1d_filters="$qz_x_freq_conv1d_filters" \
  detector.train_window_model.time_freq.eq__True.qz_x_freq_conv1d_kernel_sizes="$qz_x_freq_conv1d_kernel_sizes" \
  detector.train_window_model.time_freq.eq__True.qz_x_freq_conv1d_strides="$qz_x_freq_conv1d_strides" \
  detector.train_window_model.time_freq.eq__True.px_z_freq_conv1d_filters="$px_z_freq_conv1d_filters" \
  detector.train_window_model.time_freq.eq__True.px_z_freq_conv1d_kernel_sizes="$px_z_freq_conv1d_kernel_sizes" \
  detector.train_window_model.time_freq.eq__True.px_z_freq_conv1d_strides="$px_z_freq_conv1d_strides" \
  detector.train_window_model.latent_dim="$latent_dim" \
  detector.train_window_model.type_.eq__rec.rec_unit_type="$rec_unit_type" \
  detector.train_window_model.type_.eq__rec.activation_rec="$activation_rec" \
  detector.train_window_model.type_.eq__rec.conv1d_pooling="$conv1d_pooling" \
  detector.train_window_model.type_.eq__rec.conv1d_batch_norm="$conv1d_batch_norm" \
  detector.train_window_model.type_.eq__rec.rec_weight_decay="$rec_weight_decay" \
  detector.train_window_model.weight_decay="$weight_decay" \
  detector.train_window_model.dropout="$dropout" \
  detector.train_window_model.dec_output_dist="$dec_output_dist" \
  detector.train_window_model.min_beta="$min_beta" \
  detector.train_window_model.max_beta="$max_beta" \
  detector.train_window_model.beta_n_epochs="$beta_n_epochs" \
  detector.train_window_model.loss_weighting.value="$loss_weighting" \
  detector.train_window_model.loss_weighting.eq__fixed.d_classifier_weight="$d_classifier_weight" \
  detector.train_window_model.learning_rate="$learning_rate" \
  detector.train_window_model.lr_scheduling="$lr_scheduling" \
  detector.train_window_model.lrs_pwc_red_factor="$lrs_pwc_red_factor" \
  detector.train_window_model.lrs_pwc_red_freq="$lrs_pwc_red_freq" \
  detector.train_window_model.optimizer.value="$optimizer" \
  detector.train_window_model.optimizer.eq__adamw.adamw_weight_decay="$adamw_weight_decay" \
  detector.train_window_model.softplus_scale="$softplus_scale" \
  detector.train_window_model.batch_size="$batch_size" \
  detector.train_window_model.n_epochs="$n_epochs" \
  detector.train_window_model.early_stopping_target="$early_stopping_target" \
  detector.train_window_model.early_stopping_patience="$early_stopping_patience" \
  detector.train_window_scorer.step_id="$train_window_scorer_id" \
  detector.train_window_scorer.scoring_method.value="$scoring_method" \
  detector.train_window_scorer.scoring_method.contains__agg_post.agg_post_dist.value="$agg_post_dist" \
  detector.train_window_scorer.scoring_method.contains__agg_post.agg_post_dist.eq__gm.agg_post_gm_n_components="$agg_post_gm_n_components" \
  detector.train_window_scorer.scoring_method.contains__mean_nll.mean_nll_n_samples="$mean_nll_n_samples" \
  detector.train_online_scorer.step_id="$train_online_scorer_id" \
  detector.train_online_scorer.scores_avg_beta="$scores_avg_beta" \
  detector.evaluate_online_scorer.step_id="$evaluate_online_scorer_id" \
  detector.evaluate_online_scorer.evaluator.ignored_anomaly_labels="$ignored_anomaly_labels" \
  detector.evaluate_online_scorer.evaluator.ignored_delayed_window="$ignored_delayed_window"
fi

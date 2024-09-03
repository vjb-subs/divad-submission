#!/bin/bash

# import utility functions
source ../utils.sh
source utils.sh

dataset_name=asd
step_name=build_features

# make_datasets args
dataset=$1
app_ids=$2
test_app_ids=$3
val_prop=$4

# build_features args
transform_chain=$5
regular_scaling_method=$6
transform_fit_normal_only=$7

make_datasets_id=$(get_make_datasets_id "$dataset" "$app_ids" "$test_app_ids" "$val_prop")
echo "make_datasets_id: ${make_datasets_id}"
build_features_id=$(get_build_features_id "$transform_chain" "$regular_scaling_method" "$transform_fit_normal_only")
echo "build_features_id: ${build_features_id}"

exathlon \
dataset="$dataset_name" \
step="$step_name" \
dataset.make_datasets.step_id="$make_datasets_id" \
dataset.make_datasets.data_manager.dataset="$dataset" \
dataset.make_datasets.data_manager.app_ids="$app_ids" \
dataset.make_datasets.data_manager.test_app_ids="$test_app_ids" \
dataset.make_datasets.data_manager.val_prop="$val_prop" \
dataset.build_features.step_id="$build_features_id" \
dataset.build_features.transform_chain.value="$transform_chain" \
dataset.build_features.transform_chain.contains__regular_scaling.regular_scaling.type_.value="$regular_scaling_method" \
dataset.build_features.transform_fit_normal_only="$transform_fit_normal_only"
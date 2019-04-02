#!/bin/bash
# You need to modify this path to your downloaded dataset directory
DATASET_DIR='/vol/vssp/cvpnobackup/scratch_4weeks/qk00006/dcase2019/task1/dataset_root'

# You need to modify this path to your workspace to store features and models
WORKSPACE='/vol/vssp/msos/qk/workspaces/dcase2019_task1'

# Hyper-parameters
GPU_ID=1
MODEL_TYPE='Cnn_9layers_AvgPooling'
BATCH_SIZE=32

# Calculate feature
python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --subtask='a' --data_type='development' --workspace=$WORKSPACE
python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --subtask='b' --data_type='development' --workspace=$WORKSPACE
python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --subtask='c' --data_type='development' --workspace=$WORKSPACE

# Calculate scalar
python utils/features.py calculate_scalar --subtask='a' --data_type='development' --workspace=$WORKSPACE
python utils/features.py calculate_scalar --subtask='b' --data_type='development' --workspace=$WORKSPACE
python utils/features.py calculate_scalar --subtask='c' --data_type='development' --workspace=$WORKSPACE

# Subtask A
CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --subtask='a' --data_type='development' --holdout_fold=1 --model_type=$MODEL_TYPE --batch_size=$BATCH_SIZE --cuda

CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py inference_validation --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --subtask='a' --data_type='development' --holdout_fold=1 --model_type=$MODEL_TYPE --iteration=5000 --batch_size=$BATCH_SIZE --cuda

# Subtask B
CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --subtask='b' --data_type='development' --holdout_fold=1 --model_type=$MODEL_TYPE --batch_size=$BATCH_SIZE --cuda

CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py inference_validation --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --subtask='b' --data_type='development' --holdout_fold=1 --model_type=$MODEL_TYPE --iteration=5000 --batch_size=$BATCH_SIZE --cuda

# Subtask C
CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --subtask='c' --data_type='development' --holdout_fold=1 --model_type=$MODEL_TYPE --batch_size=$BATCH_SIZE --cuda

CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py inference_validation --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --subtask='c' --data_type='development' --holdout_fold=1 --model_type=$MODEL_TYPE --iteration=5000 --batch_size=$BATCH_SIZE --cuda

# Plot statistics
python utils/plot_results.py --workspace=$WORKSPACE --subtask=a

############ END ############



export N_GPUS=4
# export CUDA_VISIBLE_DEVICES=0,1,2,6
export CUDA_VISIBLE_DEVICES=2,3,5,6
export LOCAL_RANK=0
# ray stop --force && ray start --head --include-dashboard=True --dashboard-port=8263

export BASE_MODEL="/NAS/chenfeng/models/Qwen/Qwen2.5-3B"

export DATA_DIR="/NAS/chenfeng/dataset/countdown"
export TRAIN_FILE_PATH=$DATA_DIR/CoT_train_lim2000.parquet
export TEST_FILE_PATH=$DATA_DIR/CoT_test_lim2000.parquet

export PROJECT_NAME=sft
export EXPERIMENT_NAME=countdown-qwen2.5-3b-coldstart-SFT-lr1e-6
export VLLM_ATTENTION_BACKEND=XFORMERS

export SWANLAB_API_KEY=YOUR_API_KEY_HERE

# export RESUME_ID=YOUREXPERIMENT_ID_HERE like lgmfk2t4ro1...
# export RESUME_START_STEP=210

mkdir -p ./outputs/$EXPERIMENT_NAME
mkdir -p ./checkpoints/sft/$EXPERIMENT_NAME
# bash ./examples/sft/gsm8k/train_gsm8k-sft-peft.sh 4 ./save/$EXPERIMENT_NAME
echo "Output saved to ./outputs/$EXPERIMENT_NAME/$(date +'%y%m%d-%H%M%S').nohupoutput"; \
nohup bash ./scripts/sft_countdown_4a20.sh $N_GPUS ./checkpoints/sft/$EXPERIMENT_NAME &>"./outputs/$EXPERIMENT_NAME/$(date +'%y%m%d-%H%M%S').nohupoutput"

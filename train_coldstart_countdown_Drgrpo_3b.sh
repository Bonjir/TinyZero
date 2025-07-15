# try 3B-2048 Dr.GRPO on 4A20 GPUs

export N_GPUS=4
export CUDA_VISIBLE_DEVICES=2,3,4,6
ray stop --force && ray start --head --include-dashboard=True --dashboard-port=8263

# from scratch
# export BASE_MODEL="/NAS/chenfeng/models/Qwen/Qwen2.5-3B"

# coldstart
export BASE_MODEL="/NAS/chenfeng/TinyZero-NOOOM-dr-grpo/checkpoints/sft/countdown-qwen2.5-3b-coldstart-SFT/global_step_61"

# resume
# export BASE_MODEL=/NAS/chenfeng/TinyZero-NOOOM-dr-grpo/checkpoints/TinyZero/countdown-qwen2.5-3b-dr-grpo/actor/global_step_90

export DATA_DIR="/NAS/chenfeng/dataset/countdown"
export ROLLOUT_TP_SIZE=4
export EXPERIMENT_NAME=cntdn-qwen2.5-3b--COLDSTART_cntdn_1e-5-DrGRPO
export VLLM_ATTENTION_BACKEND=XFORMERS

export SWANLAB_API_KEY=YOUR_API_KEY_HERE

# bash ./scripts/train_tiny_zero_a100_drgrpo_tang3.sh
# nohup bash ./scripts/train_tiny_zero_a100_drgrpo_tang3.sh &>./nohupoutput
mkdir -p ./outputs/$EXPERIMENT_NAME
echo "\
# Output saved to ./outputs/$EXPERIMENT_NAME/$(date +'%y%m%d-%H%M%S').nohupoutput"; \
nohup bash ./scripts/train_tiny_zero_drgrpo_4a20_tang3.sh &>"./outputs/$EXPERIMENT_NAME/$(date +'%y%m%d-%H%M%S').nohupoutput"

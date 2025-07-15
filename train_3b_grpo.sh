# This one works for 3B GRPO on 2A20 GPUs

export N_GPUS=2
export CUDA_VISIBLE_DEVICES=3,4
# ray stop --force && ray start --head --include-dashboard=True --dashboard-port=8263
export BASE_MODEL="/NAS/chenfeng/models/Qwen/Qwen2.5-3B"
export DATA_DIR="/NAS/chenfeng/dataset/countdown"
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b-grpo-better-format
export VLLM_ATTENTION_BACKEND=XFORMERS

export SWANLAB_API_KEY=YOUR_API_KEY_HERE

# bash ./scripts/train_tiny_zero_a100_grpo.sh
nohup bash ./scripts/train_tiny_zero_a100_grpo.sh &>./nohupoutput_grpo

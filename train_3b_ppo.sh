
export N_GPUS=2
export CUDA_VISIBLE_DEVICES=3,4
# ray stop --force && ray start --head --include-dashboard=True
export BASE_MODEL="/NAS/chenfeng/models/Qwen/Qwen2.5-3B"
export DATA_DIR="/NAS/chenfeng/dataset/countdown"
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b-ppo
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero_a100_ppo.sh
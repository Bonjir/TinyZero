# This one works for 1.5B PPO on 4A20 GPUs

export N_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,5,6
ray stop --force && ray start --head --include-dashboard=True
export BASE_MODEL="/NAS/chenfeng/models/Qwen/Qwen2.5-1.5B/Qwen/Qwen2.5-1.5B"
export DATA_DIR="/NAS/chenfeng/dataset/countdown"
export ROLLOUT_TP_SIZE=4
export EXPERIMENT_NAME=countdown-qwen2.5-1.5b-ppo
export VLLM_ATTENTION_BACKEND=XFORMERS

export SWANLAB_API_KEY=YOUR_API_KEY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# bash ./scripts/train_tiny_zero_a100_ppo.sh
nohup bash ./scripts/train_tiny_zero_4a20_1.5b_ppo.sh.sh &>"./outputs/ppo_$(date +'%y%m%d-%H%M%S').nohupoutput"

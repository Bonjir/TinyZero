
export N_GPUS=2
export CUDA_VISIBLE_DEVICES=0,1
ray stop --force && ray start --head --include-dashboard=True --dashboard-port=8263
# resume model path
export BASE_MODEL=/NAS/chenfeng/TinyZero-NOOOM-dr-grpo/checkpoints/TinyZero/countdown-qwen2.5-3b-dr-grpo/actor/global_step_90
export DATA_DIR="/NAS/chenfeng/dataset/countdown"
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b-dr-grpo
export VLLM_ATTENTION_BACKEND=XFORMERS

export SWANLAB_API_KEY=YOUR_API_KEY_HERE

export RESUME_ID=YOUR_EXPERIMENT_ID_HERE like yg7az62y5...
export RESUME_START_STEP=96

# nohup bash ./scripts/train_tiny_zero_a100_drgrpo_tang3.sh &>./nohupoutput
nohup bash ./scripts/train_tiny_zero_drgrpo_2a100_resume_tang3.sh &>"./outputs/drgrpo/$(date +'%y%m%d-%H%M%S').nohupoutput"

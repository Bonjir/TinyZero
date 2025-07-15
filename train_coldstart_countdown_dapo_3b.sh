# try 3B-2048 DAPO on 4A40 GPUs
# this is a training script for DAPO algorithm & length limit 2048

# similarity&differnce from dr-grpo:
# [v]token-mean
# [v]clip-higher
# [v]no-kl-loss
# [v]turn on norm_adv_by_std_in_grpo
# [v]no-kl-in-reward
# [v]overlong-buffer
# [x]dynamic-sampling


export N_GPUS=4
export CUDA_VISIBLE_DEVICES=1,2,3,4
ray stop --force && ray start --head --include-dashboard=True --dashboard-port=8263

# train from scratch
# export BASE_MODEL="/NAS/chenfeng/models/Qwen/Qwen2.5-3B"

# # coldstart
# export BASE_MODEL="/NAS/chenfeng/TinyZero-NOOOM-dr-grpo/checkpoints/sft/countdown-qwen2.5-3b-coldstart-SFT/global_step_61"

# resume
export BASE_MODEL=/NAS/chenfeng/TinyZero-NOOOM-dr-grpo/checkpoints/TinyZero/cntdn-qwen2.5-3b--COLDSTART_cntdn_DAPO/actor/global_step_30

export DATA_DIR="/NAS/chenfeng/dataset/countdown"
export ROLLOUT_TP_SIZE=4
export EXPERIMENT_NAME=cntdn-qwen2.5-3b--COLDSTART_cntdn_DAPO
export VLLM_ATTENTION_BACKEND=XFORMERS

export SWANLAB_API_KEY=YOUR_API_KEY_HERE

export ENABLE_RESUME=True
export RESUME_ID=YOUR_EXPERIMENT_ID
export RESUME_START_STEP=30

# params

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

loss_agg_mode="token-mean"

max_prompt_length=$((256))
max_response_length=$((1024 * 2))  # 2K
enable_overlong_buffer=True
overlong_buffer_len=256 # $((1024 * 1.5))  # 1.5K # maxlen = 2048 这个是buffer，而不是期望的长度
overlong_penalty_factor=1.0 

# bash ./scripts/train_tiny_zero_a100_drgrpo_tang3.sh
# nohup bash ./scripts/train_tiny_zero_a100_drgrpo_tang3.sh &>./nohupoutput
mkdir -p ./outputs/$EXPERIMENT_NAME
echo "\
# Output saved to ./outputs/$EXPERIMENT_NAME/$(date +'%y%m%d-%H%M%S').nohupoutput"; \
nohup /home/zhangyi/miniconda3/envs/wcf-zero-py3.10/bin/python3 -m verl.trainer.main_ppo \
    trainer.tracking.resume=$ENABLE_RESUME \
    trainer.tracking.resume_id=$RESUME_ID \
    trainer.tracking.start_step=$RESUME_START_STEP \
    \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=640 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.loss_agg_mode=$loss_agg_mode \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.critic_warmup=0 \
    trainer.logger=['swanlab'] \
    ++trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.project_name=TinyZero \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 2>&1 | tee verl_demo.log\
&>"./outputs/$EXPERIMENT_NAME/$(date +'%y%m%d-%H%M%S').nohupoutput"


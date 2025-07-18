#this is training script for ppo algorithm

/home/zhangyi/miniconda3/envs/wcf-zero-py3.10/bin/python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=320 \
    data.max_prompt_length=256 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    critic.optim.lr=1e-5 \
    critic.model.path=$BASE_MODEL \
    critic.ppo_micro_batch_size=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['swanlab'] \
    +trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.project_name=TinyZero \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 2>&1 | tee verl_demo.log


# /home/zhangyi/miniconda3/envs/wcf-zero-py3.10/bin/python3 -m verl.trainer.main_ppo \
#     data.train_files=$DATA_DIR/train.parquet \
#     data.val_files=$DATA_DIR/test.parquet \
#     data.train_batch_size=32 \
#     data.val_batch_size=64 \
#     data.max_prompt_length=256 \
#     data.max_response_length=1024 \
#     actor_rollout_ref.model.path=$BASE_MODEL \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.actor.ppo_mini_batch_size=16 \
#     actor_rollout_ref.actor.ppo_micro_batch_size=4 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.use_dynamic_bsz=True \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
#     critic.optim.lr=1e-5 \
#     critic.model.path=$BASE_MODEL \
#     critic.ppo_micro_batch_size=4 \
#     algorithm.kl_ctrl.kl_coef=0.001 \
#     trainer.logger=['swanlab'] \
#     +trainer.val_before_train=False \
#     trainer.default_hdfs_dir=null \
#     trainer.n_gpus_per_node=$N_GPUS \
#     trainer.nnodes=1 \
#     trainer.save_freq=10 \
#     trainer.test_freq=10 \
#     trainer.project_name=TinyZero \
#     trainer.experiment_name=$EXPERIMENT_NAME \
#     trainer.total_epochs=15 2>&1 | tee verl_demo.log
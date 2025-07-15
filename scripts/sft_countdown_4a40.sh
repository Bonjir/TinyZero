# Tested with 2 & 4 GPUs

set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_gemma_2b.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_FILE_PATH \
    data.val_files=$TEST_FILE_PATH \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    data.truncation=left \
    data.max_length=2048 \
    data.train_batch_size=32 \
    data.micro_batch_size=4 \
    data.truncation=pass \
    \
    model.partial_pretrain=$BASE_MODEL \
    \
    optim.lr=1e-6 \
    optim.weight_decay=0.01 \
    optim.warmup_steps_ratio=0.1 \
    \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger=['swanlab'] \
    trainer.total_epochs=1 \
    trainer.default_hdfs_dir=null \
    trainer.val_before_training=True \
    trainer.validate_every_n_steps=10 \
    $@
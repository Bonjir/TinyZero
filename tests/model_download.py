from modelscope import snapshot_download

# 指定缓存目录
cache_dir = "/NAS/chenfeng/models/Qwen/Qwen2.5-1.5B"

# 下载模型（示例：下载 damo/nlp_structbert_backbone_base_std）
snapshot_download(
    model_id="Qwen/Qwen2.5-1.5B",
    cache_dir=cache_dir
)

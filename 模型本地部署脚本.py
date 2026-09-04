#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export HF_TOKEN="我的令牌"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="models/Qwen/Qwen3-8B"
export HF_HUB_MAX_WORKERS=4
CUDA_VISIBLE_DEVICES=0 python 模型本地部署脚本.py


千问的：
models--Qwen--Qwen3-8B
b968826d9c46dd6066d109eabc6255188de91218

Llama的：
3-1-8B-Instruct
models--meta-llama--Llama-3.1-8B-Instruct
0e9e39f249a16976918f6564b8830bc894c89659


"""
import os
from huggingface_hub import snapshot_download

# ========= 配置 =========
REPO_ID = "models/Qwen/Qwen3-8B"
HF_HOME = "models/Qwen/Qwen3-8B" 
TOKEN = os.environ.get("HF_TOKEN", "")  
REVISION = None 

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HOME", HF_HOME)

os.environ.setdefault("HF_HUB_MAX_WORKERS", "2") 

print("HF_ENDPOINT =", os.environ["HF_ENDPOINT"])
print("HF_HOME     =", os.environ["HF_HOME"])
print("Downloading  =", REPO_ID)

snapshot_path = snapshot_download(
    repo_id=REPO_ID,
    revision=REVISION,
    token=TOKEN if TOKEN else None,
    resume_download=True,
    cache_dir=os.path.join(HF_HOME, "hub"),
)

print("\n✅ Download OK")
print("Snapshot path:")
print(snapshot_path)

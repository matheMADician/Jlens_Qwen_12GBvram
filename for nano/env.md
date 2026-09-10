## 環境

| 項目 | 版本 |
|---|---|
| OS | Ubuntu 22.04 LTS |
| GPU | NVIDIA GeForce RTX 3060 (12GB / 實測可用 ~11.6GB) |
| NVIDIA 驅動 | ≥ 550.x |
| CUDA | 12.1 |
| Python | 3.10 (conda/venv 隔離環境) |
| PyTorch | 2.5.1+cu121 |
| transformers | 5.16.0.dev0（從 source 安裝：`pip install git+https://github.com/huggingface/transformers`，Qwen2-Audio 在較新版本才穩定支援）|
| bitsandbytes | 0.50.1（4-bit NF4 量化）|
| accelerate | 1.14.0 |
| librosa | 0.11.0 |
| soundfile | 0.14.0 |
| jacobian-lens | Apache-2.0，來自 `anthropics/jacobian-lens`，`pip install -e .` 本地安裝 |
---
## please run all the test after setting up the environment

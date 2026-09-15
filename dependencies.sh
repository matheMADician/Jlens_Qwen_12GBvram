#!/usr/bin/env bash
# =============================================================================
#  setup_jlens_env.sh
#
#  用途：在已啟用的 conda 環境中，安裝 J-lens 實作所需的完整依賴。
#
#  前提：
#    - Ubuntu 22.04 LTS
#    - NVIDIA 驅動 >= 550.x，CUDA 12.1
#    - 使用者已自行建立並啟用 conda 環境（Python 3.10）
#    - 執行前請確認 `which python` 指向 conda 環境
#
#  用法：
#    conda activate <your_env>
#    bash setup_jlens_env.sh
#
#  選項：
#    SKIP_TORCH=1     跳過 PyTorch 安裝（若已裝好）
#    SKIP_JLENS=1     跳過 jacobian-lens 安裝
#    JLENS_DIR=...    指定 jacobian-lens 本地路徑（預設 ./jacobian-lens）
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 0. 基本設定
# -----------------------------------------------------------------------------
PYTHON_VERSION="3.10"
TORCH_VERSION="2.5.1"
CUDA_TAG="cu121"
TRANSFORMERS_REPO="git+https://github.com/huggingface/transformers"
JLENS_REPO="https://github.com/anthropics/jacobian-lens"
JLENS_DIR="${JLENS_DIR:-./jacobian_lens}"

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# -----------------------------------------------------------------------------
# 1. 環境檢查
# -----------------------------------------------------------------------------
log_info "檢查執行環境..."

# 確認在 conda 環境中
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
    log_error "未偵測到 conda 環境。請先執行：conda activate <your_env>"
    exit 1
fi
log_info "Conda 環境：${CONDA_DEFAULT_ENV}"

# 確認 Python 版本
PY_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PY_VER}" != "${PYTHON_VERSION}" ]]; then
    log_warn "目前 Python 版本為 ${PY_VER}，預期 ${PYTHON_VERSION}。若不相容請自行調整。"
fi

# 確認 nvidia-smi 可用
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log_error "找不到 nvidia-smi，請確認 NVIDIA 驅動已正確安裝。"
    exit 1
fi
log_info "NVIDIA 驅動：$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
log_info "GPU：$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1)"

# 升級 pip 工具鏈
log_info "升級 pip / setuptools / wheel ..."
python -m pip install --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# 2. PyTorch 2.5.1 + CUDA 12.1
# -----------------------------------------------------------------------------
if [[ "${SKIP_TORCH:-0}" != "1" ]]; then
    log_info "安裝 PyTorch ${TORCH_VERSION}+${CUDA_TAG} ..."
    pip install \
        torch=="${TORCH_VERSION}" \
        torchvision \
        torchaudio \
        --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
else
    log_warn "SKIP_TORCH=1，跳過 PyTorch 安裝。"
fi

# 驗證 PyTorch 與 CUDA
log_info "驗證 PyTorch CUDA 可用性..."
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA 不可用，請檢查驅動與 PyTorch 版本。"
print(f"  torch          : {torch.__version__}")
print(f"  CUDA (runtime) : {torch.version.cuda}")
print(f"  GPU            : {torch.cuda.get_device_name(0)}")
print(f"  VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
PY

# -----------------------------------------------------------------------------
# 3. HuggingFace transformers（source 安裝，Qwen2-Audio 穩定支援）
# -----------------------------------------------------------------------------
log_info "從 source 安裝 transformers（5.16.0.dev0 分支）..."
pip install --upgrade "${TRANSFORMERS_REPO}"

# 補齊 transformers 執行期常用依賴
log_info "安裝 transformers 執行期依賴（sentencepiece / safetensors / tokenizers 等）..."
pip install \
    sentencepiece \
    safetensors \
    tokenizers \
    huggingface_hub \
    einops

# -----------------------------------------------------------------------------
# 4. 量化與加速
# -----------------------------------------------------------------------------
log_info "安裝 bitsandbytes 0.50.1（4-bit NF4）..."
pip install "bitsandbytes==0.50.1"

log_info "安裝 accelerate 1.14.0 ..."
pip install "accelerate==1.14.0"

# -----------------------------------------------------------------------------
# 5. 音訊處理（Qwen2-Audio 相關）
# -----------------------------------------------------------------------------
log_info "安裝 librosa 0.11.0 / soundfile 0.14.0 ..."
pip install \
    "librosa==0.11.0" \
    "soundfile==0.14.0" \
    datasets \
    "torchcodec==0.2.1"

# torchcodec 需要系統 FFmpeg 動態函式庫（libavutil/libavcodec 等）。
if ! command -v ffmpeg >/dev/null 2>&1; then
    log_error "找不到 FFmpeg，torchcodec 無法解碼 FLEURS 音訊。"
    log_error "請先執行：sudo apt-get install -y ffmpeg"
    exit 1
fi

# soundfile 需要系統 libsndfile
if ! ldconfig -p | grep -q libsndfile; then
    log_warn "系統缺少 libsndfile，soundfile 可能無法載入。"
    log_warn "請執行：sudo apt-get install -y libsndfile1"
fi

# -----------------------------------------------------------------------------
# 6. jacobian-lens（本地可編輯安裝）
# -----------------------------------------------------------------------------
if [[ "${SKIP_JLENS:-0}" != "1" ]]; then
    if [[ ! -d "${JLENS_DIR}" ]]; then
        log_info "複製 jacobian-lens 至 ${JLENS_DIR} ..."
        git clone "${JLENS_REPO}" "${JLENS_DIR}"
    else
        log_warn "${JLENS_DIR} 已存在，略過 clone。"
    fi

    log_info "以 editable 模式安裝 jacobian-lens ..."
    pip install -e "${JLENS_DIR}"

    # 補齊官方參考實作所需依賴
    log_info "安裝 jacobian-lens 額外依賴（matplotlib / plotly / jupyter 等）..."
    pip install \
        matplotlib \
        plotly \
        jupyter \
        ipykernel \
        tqdm
else
    log_warn "SKIP_JLENS=1，跳過 jacobian-lens 安裝。"
fi

# -----------------------------------------------------------------------------
# 7. 安裝結果總覽
# -----------------------------------------------------------------------------
log_info "安裝完成，版本總覽："
python - <<'PY'
import importlib

pkgs = [
    ("torch",         "torch"),
    ("transformers",  "transformers"),
    ("bitsandbytes",  "bitsandbytes"),
    ("accelerate",    "accelerate"),
    ("librosa",       "librosa"),
    ("soundfile",     "soundfile"),
    ("datasets",      "datasets"),
    ("torchcodec",    "torchcodec"),
    ("jlens",         "jacobian-lens"),
]

for name, label in pkgs:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "unknown")
        print(f"  {label:<16}: {ver}")
    except Exception as e:
        print(f"  {label:<16}: [未安裝或匯入失敗] {e}")
PY

log_info "環境設定完成。建議執行 'python -c \"import jlens; print(jlens.__file__)\"' 驗證。"
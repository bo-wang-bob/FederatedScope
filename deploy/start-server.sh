#!/usr/bin/env bash
# Run using: bash backend/deploy/start-server.sh
# This script is for the GPU server, not the local CPU test environment.
set -euo pipefail
platform_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
platform_python="${PLATFORM_PYTHON:-python3}"

unset FS_PLATFORM_DEVICE
export FS_BACKDOOR_DEVICE=cuda
if [[ "${CUDA_VISIBLE_DEVICES-}" == "-1" ]]; then
    unset CUDA_VISIBLE_DEVICES
fi
# Preserve an administrator's explicit GPU index mask; do not hide GPUs here.
export FS_PLATFORM_RESOURCES="$platform_root/backend/resources"
export FS_PLATFORM_DATASETS="$FS_PLATFORM_RESOURCES/datasets"
export FS_PLATFORM_UPLOADS="$FS_PLATFORM_RESOURCES/uploaded_datasets"
export FS_FEDMIA_LOCAL_ROOT="$FS_PLATFORM_RESOURCES/fedmia_local"
export FS_BACKDOOR_BASE="$FS_PLATFORM_RESOURCES/backdoor/exp/sabre_newdataset"
export FS_BACKDOOR_DATA_ROOT="$FS_PLATFORM_DATASETS/MilitaryAircraft3D"
export FS_BACKDOOR_VIT_WEIGHTS="$FS_PLATFORM_RESOURCES/models/ViT-B-16.pt"
export FS_PLATFORM_MILITARY_VIT_WEIGHTS="$FS_BACKDOOR_VIT_WEIGHTS"
export FS_PLATFORM_AUGMENTED_MILITARY_VIT="$FS_PLATFORM_RESOURCES/caches/military_vit_default"
export FEDERATEDSCOPE_FRONTEND_DIST="$platform_root/frontend/dist"
export FEDERATEDSCOPE_GGEUR_LIGHTWEIGHT=1
export TORCH_HOME="$FS_PLATFORM_RESOURCES/torch"
export MPLCONFIGDIR="$platform_root/backend/exp/platform/mpl"
export MPLBACKEND=Agg
mkdir -p "$MPLCONFIGDIR"

# FS_PLATFORM_IMPORT_ROOTS may be set by the administrator as a JSON array.
# It is only needed for datasets outside FS_PLATFORM_DATASETS.
if [[ ! -f "$FEDERATEDSCOPE_FRONTEND_DIST/index.html" ]]; then
    echo 'Missing frontend/dist/index.html. Run npm ci && npm run build in frontend/.' >&2
    exit 1
fi
cd "$platform_root/backend"
"$platform_python" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable: check driver, container --gpus all and NVIDIA Container Toolkit'; x=torch.ones(1,device='cuda'); print('GPU:',torch.cuda.get_device_name(0)); torch.cuda.synchronize()"
exec "$platform_python" scripts/start_platform.py --host 0.0.0.0 --port 8002 --state-dir exp/platform

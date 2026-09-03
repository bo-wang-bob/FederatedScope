#!/usr/bin/env bash
set -euo pipefail

GROUP="${1:?usage: prepare_feature_cache.sh GROUP [METHOD]}"
METHOD="${2:-fedavg}"
PYTHON_BIN="${PYTHON_BIN:-/root/.local/share/mamba/envs/GGEUR/bin/python}"
DEVICE="${DEVICE:-1}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
SOURCE_CONFIG="${REPO_DIR}/scripts/example_configs/ggeur_final_5models/${GROUP}/${METHOD}.yaml"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-${REPO_DIR}/exp/distributed_feature_cache}"
OFFICEHOME_MANIFEST_ROOT="${OFFICEHOME_MANIFEST_ROOT:-${REPO_DIR}/exp/distributed_manifests/officehome_60c_lds01_seed42}"
DOMAINNET_MANIFEST_PATH="${DOMAINNET_MANIFEST_PATH:-${REPO_DIR}/exp/distributed_manifests/domainnet_4domains/domainnet_manifest.json}"
CACHE_DIR="${FEATURE_CACHE_ROOT%/}/${GROUP}"
LOG_DIR="${REPO_DIR}/exp/cache_warmup/${GROUP}"
LOG_PATH="${LOG_DIR}/prepare_feature_cache.log"
READY_MARKER="${CACHE_DIR}/.ggeur_feature_cache_ready.json"

test -f "${SOURCE_CONFIG}"
mkdir -p "${LOG_DIR}" "${CACHE_DIR}"
if [[ -f "${READY_MARKER}" ]]; then
  imported_marker="${READY_MARKER}.imported_$(date +%Y%m%d_%H%M%S)"
  mv "${READY_MARKER}" "${imported_marker}"
  echo "preserved_previous_marker=${imported_marker}"
fi

args=(
  -m federatedscope.main --cfg "${SOURCE_CONFIG}"
  federate.total_round_num 1
  federate.mode standalone
  use_gpu True
  device "${DEVICE}"
  eval.freq 0
  ggeur.feature_cache_dir "${CACHE_DIR}"
  ggeur.use_feature_cache True
  ggeur.require_complete_feature_cache False
  ggeur.reuse_augmented_feature_cache False
  ggeur.save_augmented_feature_cache False
  outdir "${REPO_DIR}/exp/cache_warmup/${GROUP}"
  expname cache_warmup
)

case "${GROUP}" in
  officehome_*)
    test -d "${OFFICEHOME_MANIFEST_ROOT}"
    args+=(
      data.root /root/autodl-tmp/datasets/OfficeHomeDataset_10072016
      ggeur.officehome_manifest_base "${OFFICEHOME_MANIFEST_ROOT}"
      ggeur.officehome_manifest_use_config_root True
    )
    ;;
  domainnet_*)
    test -f "${DOMAINNET_MANIFEST_PATH}"
    args+=(
      data.root /root/autodl-tmp/datasets/DomainNet
      ggeur.domainnet_manifest_path "${DOMAINNET_MANIFEST_PATH}"
    )
    ;;
  mdsent_*) args+=(data.root /root/autodl-tmp/datasets/sentiment) ;;
  *) echo "unsupported group: ${GROUP}" >&2; exit 2 ;;
esac
case "${GROUP}" in
  *_vit) args+=(ggeur.clip_model_path /root/.cache/clip/ViT-B-16.pt) ;;
  *_mixer) args+=(ggeur.timm_checkpoint_path /root/autodl-tmp/models/mixer_b16_224_complete.pth) ;;
  mdsent_*) args+=(ggeur.bert_model_path /root/autodl-tmp/models/nlptown_bert_base_multilingual_uncased_senti) ;;
esac

cd "${REPO_DIR}"
export FEDERATEDSCOPE_GGEUR_LIGHTWEIGHT=1
set +e
"${PYTHON_BIN}" "${args[@]}" >"${LOG_PATH}" 2>"${LOG_PATH}.stderr"
warmup_exit_code=$?
set -e

expected="$(
  awk '/^[[:space:]]*client_num:/{print $2; exit}' "${SOURCE_CONFIG}" |
    tr -d '\r'
)"
seen="$({
  grep -h 'GGEUR_TIMING_CLIENT .*stage=feature_extraction' \
    "${LOG_PATH}" "${LOG_PATH}.stderr" 2>/dev/null || true
} | sed -n 's/.*client=\([0-9][0-9]*\).*/\1/p' | sort -nu | wc -l)"
if [[ "${seen}" -ne "${expected}" ]]; then
  echo "cache preparation covered ${seen}/${expected} clients; see ${LOG_PATH}" >&2
  exit 1
fi
image_errors="$(
  {
    grep -hE 'Error loading image|Failed to load image' \
      "${LOG_PATH}" "${LOG_PATH}.stderr" 2>/dev/null || true
  } | wc -l
)"
if [[ "${image_errors}" -ne 0 ]]; then
  echo "cache preparation logged ${image_errors} image loading errors; see ${LOG_PATH}" >&2
  exit 1
fi
shopt -s nullglob
cache_files=("${CACHE_DIR}"/*.npz)
if [[ "${#cache_files[@]}" -eq 0 ]]; then
  echo "no raw feature cache files under ${CACHE_DIR}" >&2
  exit 1
fi
"${PYTHON_BIN}" - "${cache_files[@]}" <<'PY'
import sys

import numpy as np

for path in sys.argv[1:]:
    with np.load(path, allow_pickle=True) as cache:
        if 'paths' not in cache or 'features' not in cache:
            raise SystemExit(f'invalid feature cache keys: {path}')
        paths = cache['paths']
        features = cache['features']
        if len(paths) == 0 or features.ndim != 2 or \
                len(paths) != len(features):
            raise SystemExit(f'invalid feature cache shape: {path}')
        if not np.isfinite(features).all():
            raise SystemExit(f'non-finite feature cache values: {path}')
PY
if [[ "${warmup_exit_code}" -ne 0 ]]; then
  echo "accepted warmup exit_code=${warmup_exit_code} after complete client coverage, image, and NPZ validation" >&2
fi
printf '{\n  "group": "%s",\n  "method": "%s",\n  "client_num": %s,\n  "cache_files": %s,\n  "completed_at": "%s"\n}\n' \
  "${GROUP}" "${METHOD}" "${expected}" "${#cache_files[@]}" "$(date --iso-8601=seconds)" \
  >"${READY_MARKER}"
echo "feature_cache_ready=${CACHE_DIR}"

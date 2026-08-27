#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/FederatedScope}"
PYTHON_BIN="${PYTHON_BIN:-/root/.local/share/mamba/envs/GGEUR/bin/python}"
DOWNLOAD_JOB="${DOWNLOAD_JOB:-/root/autodl-tmp/resource_downloads/domainnet_job_20260723}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/datasets/DomainNet}"
SPLIT_LIST_DIR="${SPLIT_LIST_DIR:-$DOWNLOAD_JOB/txt}"
LABEL_LIST_DIR="${LABEL_LIST_DIR:-$DOWNLOAD_JOB/groundtruth_txt}"
SCRIPT_DIR="$REPO_DIR/scripts/distributed_scripts/ggeur_hierarchical_3machine"
MANIFEST="$REPO_DIR/exp/distributed_manifests/domainnet_4domains/domainnet_manifest.json"
PUBLISH_DIR="$REPO_DIR/exp/domainnet_resource_publish"
STATE_DIR="$REPO_DIR/exp/domainnet_resource_prepare"
LOG="$STATE_DIR/pipeline.log"
mkdir -p "$PUBLISH_DIR" "$STATE_DIR"

record() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >>"$LOG"; }

record WAIT_DOWNLOAD
while [[ ! -f "$DOWNLOAD_JOB/completed_at.txt" ]]; do
  if [[ -f "$DOWNLOAD_JOB/exit_code.txt" ]] && \
      [[ "$(cat "$DOWNLOAD_JOB/exit_code.txt")" != 0 ]]; then
    record DOWNLOAD_FAILED
    exit 10
  fi
  sleep 60
done
record DOWNLOAD_COMPLETE

for domain in clipart painting real sketch; do
  test -d "$DATA_ROOT/$domain"
done

"$PYTHON_BIN" "$SCRIPT_DIR/generate_domainnet_manifest.py" \
  --root "$DATA_ROOT" --output "$MANIFEST" \
  --domains clipart,painting,real,sketch \
  --split-list-dir "$SPLIT_LIST_DIR" \
  --label-list-dir "$LABEL_LIST_DIR" >>"$LOG" 2>&1
record MANIFEST_COMPLETE

for group in domainnet_vit domainnet_cnn domainnet_mixer; do
  marker="$REPO_DIR/exp/distributed_feature_cache/$group/.ggeur_feature_cache_ready.json"
  if [[ ! -f "$marker" ]]; then
    record "CACHE_START $group GPU0"
    DEVICE=0 PYTHON_BIN="$PYTHON_BIN" \
      bash "$SCRIPT_DIR/prepare_feature_cache.sh" "$group" fedavg \
      >>"$LOG" 2>&1
    record "CACHE_COMPLETE $group"
  fi
done

bundle="$PUBLISH_DIR/domainnet_4domains_cache_bundle.tar.gz"
tmp="$bundle.tmp"
cd "$REPO_DIR"
tar -czf "$tmp" \
  exp/distributed_manifests/domainnet_4domains \
  exp/distributed_feature_cache/domainnet_vit \
  exp/distributed_feature_cache/domainnet_cnn \
  exp/distributed_feature_cache/domainnet_mixer
mv "$tmp" "$bundle"
sha256sum "$bundle" | awk '{print $1}' >"$bundle.sha256"
cat >"$PUBLISH_DIR/domainnet_bundle.ready.json" <<EOF
{
  "bundle": "$(basename "$bundle")",
  "sha256": "$(cat "$bundle.sha256")",
  "completed_at": "$(date --iso-8601=seconds)"
}
EOF
record "BUNDLE_COMPLETE $(stat -c %s "$bundle") $(cat "$bundle.sha256")"
touch "$STATE_DIR/pipeline.complete"

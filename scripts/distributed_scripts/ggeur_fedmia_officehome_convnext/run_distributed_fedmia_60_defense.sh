#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export ATTACK_TYPE=fedmia
export RUN_ID="${RUN_ID:-officehome_60c_fedmia_defense_$(date +%Y%m%d_%H%M%S)}"
export FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-exp/headonly_system/cache/officehome_60c_fedmia_defense}"
exec bash "$ROOT_DIR/scripts/distributed_scripts/ggeur_officehome_60_defense/run_distributed_defense.sh"

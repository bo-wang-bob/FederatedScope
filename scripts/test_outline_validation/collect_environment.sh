#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/root/autodl-tmp/FederatedScope"
PYTHON_BIN="/root/.local/share/mamba/envs/GGEUR/bin/python"
OUTPUT_DIR="$PROJECT_DIR/exp/test_outline_validation/environment"
OUTPUT_FILE="$OUTPUT_DIR/root.txt"
mkdir -p "$OUTPUT_DIR"
{
  echo "ROLE=root"
  echo "OS=$(source /etc/os-release && echo "$PRETTY_NAME")"
  echo "KERNEL=$(uname -r)"
  echo "PYTHON=$($PYTHON_BIN -c 'import platform; print(platform.python_version())')"
  echo "PYTORCH=$($PYTHON_BIN -c 'import torch; print(torch.__version__)')"
  echo "GRPCIO=$($PYTHON_BIN -c 'import grpc; print(grpc.__version__)')"
  echo "PROTOBUF=$($PYTHON_BIN -c 'import google.protobuf; print(google.protobuf.__version__)')"
  echo "PROJECT_DIR=$PROJECT_DIR"
} >"$OUTPUT_FILE"
cat "$OUTPUT_FILE"
echo "STATUS=PASS"
echo "EVIDENCE_FILE=$OUTPUT_FILE"

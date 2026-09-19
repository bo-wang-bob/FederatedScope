# ViT training/evaluation + privacy + backdoor training and image inference.
# Build online, export with docker save, then load on the offline target.
# Build context is the directory containing sibling backend/ and frontend/.
FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/index.html frontend/tsconfig*.json frontend/vite.config.ts ./
COPY frontend/src ./src
COPY frontend/resource ./resource
RUN npm run build

FROM python:3.9.23-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FEDERATEDSCOPE_GGEUR_LIGHTWEIGHT=1 FS_PLATFORM_OFFLINE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=disabled \
    HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib HF_HOME=/tmp/huggingface \
    OMP_NUM_THREADS=2 FS_FEDMIA_LOCAL_ROOT=resources/fedmia_local
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY backend/deploy/requirements* /tmp/dependencies/
RUN python -m pip install --no-cache-dir pip==25.2 setuptools==80.9.0 wheel==0.45.1 \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 -r /tmp/dependencies/requirements-torch.txt \
    && python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 -r /tmp/dependencies/requirements-runtime.lock \
    && python -m pip check
WORKDIR /opt/platform/backend
COPY backend/federatedscope ./federatedscope
COPY backend/scripts ./scripts
COPY backend/deploy ./deploy
COPY backend/run.py backend/setup.py backend/README.md ./
COPY --from=frontend /build/dist /opt/platform/frontend/dist
RUN python scripts/check_platform_environment.py --strict
EXPOSE 8002
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "scripts/platform_healthcheck.py"]
USER 1000:1000
ENTRYPOINT ["python", "scripts/start_platform.py"]
CMD ["--host", "0.0.0.0", "--port", "8002", "--state-dir", "exp/platform"]

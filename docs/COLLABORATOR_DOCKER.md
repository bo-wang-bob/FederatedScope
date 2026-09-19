# 给协作者：环境镜像构建与 GPU 启动

## 协作者需要的文件

只发送 `deploy/Dockerfile.collaborator` 即可构建**运行环境镜像**：不依赖旧 v1/v2、不需要源码或额外 requirements/wheelhouse 文件。构建机须联网。内置 Python 3.9、CUDA 12.1 版 PyTorch 2.5.1/torchvision 0.20.1、全部锁定 Python 依赖、Pillow/HEIF 解码及 Node 22/npm。无需构建机带 GPU；不安装或改动宿主显卡驱动。

```bash
docker build --platform linux/amd64 -f Dockerfile.collaborator -t federated-train-env:20260919 .
docker save -o federated-train-env-20260919.tar federated-train-env:20260919
```

镜像不包含模型、图片数据、实验结果和前后端源码。它们仍单独放置成 `项目目录/backend/` 和 `项目目录/frontend/`。同目录的 `deploy/Dockerfile` 是另一个需要源码上下文的完整平台构建方案；此次给协作者使用独立的 `Dockerfile.collaborator`，避免混用。

## Ubuntu 宿主机

需要 NVIDIA 驱动、Docker、NVIDIA Container Toolkit。先导入镜像，在包含 frontend/、backend/ 的项目目录执行：

```bash
docker load -i /实际路径/federated-train-env-20260919.tar
docker run --rm -it --gpus all --shm-size=2g \
  -p 8002:8002 -v "$PWD:/workspace" -w /workspace \
  federated-train-env:20260919 bash
```

不要在只有 Dockerfile 的目录运行第二条；挂载目录应包含完整代码和 resources。资源与状态目录需可写。此环境镜像默认以 root 运行，因此新文件可能属于 root；如需改用宿主 UID，请先确保挂载目录、npm 与运行缓存可写。

## 容器内启动（原命令的核对版）

若 `frontend/dist/index.html` 已存在且是最新版本，直接启动；否则先在联网环境构建前端：

```bash
cd /workspace/frontend
npm ci
npm run build
cd /workspace
```

核对后的启动脚本保留原资源变量，新增上传数据目录、TORCH_HOME、可写 Matplotlib 缓存、GPU 检查，并清除旧本机 CPU 默认开关：

```bash
cd /workspace
# 可选：服务器路径导入的数据不在 backend/resources/datasets 内时配置。
# 必须写容器内路径，并在 docker run 时额外挂载对应目录。
# export FS_PLATFORM_IMPORT_ROOTS='["/data/datasets"]'
bash backend/deploy/start-server.sh
```

若宿主数据在 `/data/datasets`，docker run 额外加 `-v /data/datasets:/data/datasets:ro`；导入会复制到平台 uploads，不修改源文件。网页填容器内路径。默认数据目录无需设置 IMPORT_ROOTS。

浏览器访问 `http://服务器IP:8002`，仅向可信网络开放该端口。不是浏览器连接电脑的 localhost。非 Docker 部署也可在激活服务器 Python 环境后使用此脚本，设置 `PLATFORM_PYTHON=python` 可指定解释器。

没有把 CUDA_VISIBLE_DEVICES 强行限定为 0：平台部分现有预设使用 GPU 1，应与实际可见显卡数量一致。默认上传训练及军机/OfficeHome ViT 使用 GPU 0。既有 CPU 实验需重新选择 GPU 后预检，不原样复用 gpu=-1。

## 验证边界

当前仅核对依赖锁定、代码入口、静态启动命令和本机导入/回归测试。未在本机构建此镜像，未进行 GPU 运行验收；请协作者反馈完整构建日志。构建时 pip check 和模块导入检查必须通过，GPU 检查在目标服务器启动时执行。

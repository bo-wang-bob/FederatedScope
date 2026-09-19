# 2026-09-19 环境对比与服务器默认配置

## 与用户提供的 Dockerfile.v2 对比

v2 基于 federated-train-env:v1，离线补装 open-clip-torch 3.2.0、ftfy 6.3.1、regex 2025.11.3、wcwidth 0.2.14、dill 0.3.6；它不包含平台代码、资源或服务启动入口。未取得 v1/v2 镜像内部清单，不能仅凭此文件判断其他包是否已安装。

| 内容 | 当前要求/处理 |
| --- | --- |
| 图片上传和目录导入 | Pillow 11.3.0、pillow-heif 0.22.0；v2 没有显式安装或检查 |
| ViT、后门训练 | 继续使用 torch、torchvision、open-clip-torch、timm；设备配置清理不引入新 Python 包 |
| 隐私评测、绘图 | NumPy、SciPy、scikit-learn、Matplotlib 等；完整版本在 deploy/requirements-runtime.lock |
| 前端 | Node 22 构建，npm ci 安装 package-lock.json；没有新增 npm 依赖 |
| 完整平台镜像 | deploy/Dockerfile 补拷贝 run.py，新后门训练通过此入口启动 |
| GPU | 完整平台仍锁定 torch 2.5.1+cu121、torchvision 0.20.1+cu121；不改驱动，不安装 CPU 版 PyTorch |

## 使用哪个 Dockerfile

**本次发送给协作者：deploy/Dockerfile.collaborator。** 这是无需旧镜像、无需源码或外部 requirements 的独立环境构建文件；配套启动步骤见 COLLABORATOR_DOCKER.md。以下两种是其他构建方式，不是本次必须发送的附件。

- **已有 v2 环境**：deploy/Dockerfile.env.v3 只补图片解码依赖并检查导入，保留已有 PyTorch/CUDA；构建需要联网和已有 v2 镜像。不是离线 wheelhouse-v2 的直接替代。镜像仍以 bash 启动，需另挂载代码、资源并运行启动命令。
- **完整平台镜像**：deploy/Dockerfile 配合同目录的两份 requirements 文件，构建上下文必须包含同级 backend/、frontend/：

```bash
docker build -f backend/deploy/Dockerfile -t federatedscope-platform:20260919 .
```

运行需挂载完整 resources 和可写 exp/platform，启用 --gpus all。宿主机需 NVIDIA 驱动与 NVIDIA Container Toolkit。容器 UID 1000 必须能写缓存和状态目录；模型、数据、历史结果不在镜像中。

## CPU 启动配置清理

本机 Environment.ps1 不再设置 FS_PLATFORM_DEVICE=cpu、FS_BACKDOOR_DEVICE=cpu、CUDA_VISIBLE_DEVICES=-1。代码移除 FS_PLATFORM_DEVICE 的自动 CPU 覆盖；普通上传训练默认 GPU 0，隐私沿用 GPU 预设，后门默认 cuda。

显式设备参数、通用 FS_BACKDOOR_DEVICE 管理配置仍保留；服务器不要设为 cpu。历史实验的 gpu=-1、checkpoint 的 CPU 加载、固定评测计算和隔离测试脚本不是本机启动覆盖，不改写历史记录与算法口径。旧 CPU 实验在服务器重跑时重新选择 GPU 并预检。

已有进程只有重启后才采用新配置。本机缺少 GPU，不能验证 CUDA 训练；Docker 文件修改不等于目标镜像构建或 GPU 验收完成。完整平台镜像可用 `python scripts/check_platform_environment.py --gpu --strict` 验证。现有本机 Python 环境 pip check 无冲突。

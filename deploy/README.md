# 部署准备（尚未构建镜像）

当前范围：ViT + MilitaryAircraft-3D / OfficeHome + FedAvg / FedProx / 本架构，以及隐私回放、后门原图攻防推理。训练模型的独立评测与选图验证仍使用特征包和分类头。不会重跑准确率提升门槛。前后端及资源目录见 [运行说明](../docs/PORTABLE_PLATFORM.md)。

## 已准备

- `requirements-runtime.lock`：现有 Python 3.9.23 / torch 2.5.1+cu121 环境的 65 个精确版本，含后门需要的 open-clip-torch、timm、safetensors、Hugging Face Hub 及传递依赖。`requirements-torch.txt` 先从 PyTorch 官方 cu121 源安装；不使用旧 setup.py 解析部署依赖，不需要 `pip install -e .`。当前固定 ViT 路线不使用 torchaudio 或 transformers。
- `Dockerfile`：Node 22 按 package-lock.json 安装并构建前端；Python 3.9 运行后端，镜像包含代码、静态页面及依赖，数据、权重和历史实验外置挂载。构建阶段执行 pip check、锁定版本与依赖闭包检查，以及 open_clip / timm 等真实导入和 CPU 张量计算；构建不需要 GPU。运行时不会下载模型，必须提供 resources/models/ViT-B-16.pt。
- `Dockerfile.dockerignore`：构建上下文白名单，排除 Git、资源、实验、node_modules 等。
- `platform.sh`：本地镜像启动器，不构建、不下载、不安装宿主机软件。先断网执行 GPU 张量计算和资源预检，再启动同源页面与 API。资源只读、状态独立持久化、默认绑定本机端口，按本目录所有权标签停止容器，不清理其他进程或端口。
- `check_platform_deployment.py`：临时状态中执行三方法真实预检、缓存来源/哈希和 750 张图片验证；没有训练、增强生成或正式状态变更。`--require-cache` 要求默认本架构可直接复用独立缓存包。
- 前端构建启用 Vite manifest；预检逐一校验首页、动态模块、样式和图片，拒绝缺失文件及首页/清单版本混用，记录完整运行资源指纹。清单结构依据 [Vite 构建清单文档](https://vite.dev/guide/backend-integration)。
- `platform_offline_smoke.py`：单独启动临时 HTTP 服务和状态目录，复用现有资源，不复制资源、不连接正式服务。执行三方法各 2 轮、独立评测、选图预测、CSV/JSON/模型下载及服务重启检查，不判定准确率门槛。
- `FS_PLATFORM_OFFLINE=1`：训练/预检/评测进程拒绝 Python 网络连接与解析，避免意外下载。HTTP 主服务不受该进程内保护影响。它不是宿主机防火墙，不能替代容器断网验收。

## 合并前验证记录

前端使用新建的依赖目录完成 `npm ci`、生产构建、69 项测试，不依赖旧工作树的 node_modules。Linux 开发机上检查了精确依赖版本、实际 GPU 运算、三方法默认配置预检、750 张原图、525/225 训练测试样本及增强包复用。新增启动器用替身 Docker CLI 测试离线参数、失败阻断和容器归属检查；这不等同于真实容器生命周期通过。

补充检查已通过：前端 12 个运行文件完整且两次构建运行资源指纹一致；后端原有 30 项、部署检查 6 项、启动器 8 项测试通过。离线 worker 保护开启时三方法各 2 轮训练、225 样本独立评测、三次真实选图预测、20 行分类/分域 CSV、JSON 和模型下载通过；重启前后 6 个模型条目、3 个测试集保持一致。临时测试服务已关闭，正式状态目录没有变化。

**本 Dockerfile 尚未在干净镜像中完成构建与验收，未导出镜像。** 干净 Linux 镜像依赖安装、`pip check`、断网训练/隐私/后门流程及目标 RTX 5880 验证仍待执行。Python 3.9 和旧算法依赖是兼容性冻结，不代表已完成安全更新审查；已有服务器环境检查不能替代容器验收。

## 合并后的交付待办

1. 隐私真实资源已验证；40/20/20、本地 1 轮及模型验证口径保留。使用 [目录准备脚本](../docs/PACKAGE_DIRECTORY.md) 复制资源、独立增强包和历史模型；外置脚本适配只发生在副本，未知版本拒绝处理。
2. 重新运行前端 `npm ci && npm run build && npm test`；在 Linux 容器中再验证一次，不能用 Windows 构建替代。
3. 核定发布版本和预置模型记录，整理 `backend/resources`、独立增强缓存及完整选定任务目录。模型库必须有任务登记、特征包及清单，不能只复制分类头文件。
4. 获得打包指令后才构建镜像、做完整断网短流程、停止/失败/重启/移动路径检查，导出归档和逐文件哈希。三方法准确率门槛按用户指示暂不重验。
5. 目标机确认 NVIDIA Container Toolkit、Docker 权限和磁盘；首次导入后再做 RTX 5880 容器 GPU 与短流程验收。

2026-09-16 合入回归和后续真实隐私 60 客户端验证见 [合入记录](../docs/UPSTREAM_INTEGRATION.md) 和 [隐私验证](../docs/PRIVACY_REPLAY.md)。check_platform_deployment.py 主要检查军机默认配置；新增 check_platform_directory.py 检查已存模型与隐私资源。两者都不等于真实断网容器验收。

Docker 和源码统一使用 backend/resources/fedmia_local；显式 FS_FEDMIA_LOCAL_ROOT 可以覆盖，FS_PLATFORM_RESOURCES 改变时源码的隐私默认根随之调整。资源未就绪时隐私页显示提示，军机训练与评测仍可使用。

联网构建入口：在含 `backend/` 和 `frontend/` 的父目录运行：

```bash
docker build --platform linux/amd64 -f backend/deploy/Dockerfile -t fs-aircraft:prototype .
```

完成容器验收后，联网构建机执行 `docker save -o fs-aircraft-prototype.tar fs-aircraft:prototype`，把镜像 tar 与整个资源交付目录复制到 U 盘；离线目标机执行 `docker load -i fs-aircraft-prototype.tar`。离线目标机不执行 docker build 或 pip install；NVIDIA 驱动与 Container Toolkit 属于宿主机组件，不装进镜像。

导入正式镜像并准备资源后，Ubuntu 目标机运行（不依赖 Compose）：

```bash
mkdir -p backend/exp/platform
bash backend/deploy/platform.sh check
bash backend/deploy/platform.sh start
bash backend/deploy/platform.sh stop
```

先将发布目录放到可写磁盘；状态目录必须属于运行用户。默认 `FS_RESOURCES=backend/resources`、`FS_STATE=backend/exp/platform`（相对发布目录）、`FS_PORT=8002`、`FS_IMAGE=fs-aircraft:prototype`。显式设置的 FS_RESOURCES/FS_STATE 也相对发布目录解析，或使用绝对路径；不允许含逗号。端口冲突会报错，不杀占用进程。停服务后任务被记录为停止/中断，不自动续训，配置和结果保留。启动器暂按一个 GPU（宿主 GPU 0）映射，页面仍使用 GPU 0。

API 尚无身份认证，默认只能从目标机本机浏览器访问；不要擅自将端口绑定改为公网。

官方参考：[PyTorch 对应版本安装](https://pytorch.org/get-started/previous-versions/)、[NVIDIA 容器环境](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)、[Docker GPU 运行参数](https://docs.docker.com/engine/containers/resource_constraints/#gpu)。

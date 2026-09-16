# 部署准备（尚未打包）

当前范围：ViT 冻结特征 + MilitaryAircraft-3D + FedAvg / FedProx / 本架构；独立评测与选图验证仍使用特征包和分类头。不会重跑准确率提升门槛。前后端及资源目录见 [运行说明](../docs/PORTABLE_PLATFORM.md)。

## 已准备

- `requirements-runtime.lock`：从现有 Python 3.9.23 / torch 2.5.1+cu121 环境提取的 53 个精确版本，含传递依赖；依赖声明已核验无冲突。`requirements-torch.txt` 先从 PyTorch 官方 cu121 源安装。旧 setup.py 不用于部署依赖解析，不需要 `pip install -e .`。
- `Dockerfile`：前端在 Node 22 中按 lock 文件安装和构建；Python 3.9 运行后端，镜像仅放代码、静态页面及依赖，不放数据和旧实验。仅缓存链路不需要 torchaudio、open_clip 或 transformers；若其他人的更新引入原图特征提取，必须重新审查依赖，不能沿用本清单直接交付。
- `Dockerfile.dockerignore`：构建上下文白名单，排除 Git、资源、实验、node_modules 等。
- `platform.sh`：本地镜像启动器，不构建、不下载、不安装宿主机软件。先断网执行 GPU 张量计算和资源预检，再启动同源页面与 API。资源只读、状态独立持久化、默认绑定本机端口，按本目录所有权标签停止容器，不清理其他进程或端口。
- `check_platform_deployment.py`：临时状态中执行三方法真实预检、缓存来源/哈希和 750 张图片验证；没有训练、增强生成或正式状态变更。`--require-cache` 要求默认本架构可直接复用独立缓存包。
- 前端构建启用 Vite manifest；预检逐一校验首页、动态模块、样式和图片，拒绝缺失文件及首页/清单版本混用，记录完整运行资源指纹。清单结构依据 [Vite 构建清单文档](https://vite.dev/guide/backend-integration)。
- `platform_offline_smoke.py`：单独启动临时 HTTP 服务和状态目录，复用现有资源，不复制资源、不连接正式服务。执行三方法各 2 轮、独立评测、选图预测、CSV/JSON/模型下载及服务重启检查，不判定准确率门槛。
- `FS_PLATFORM_OFFLINE=1`：训练/预检/评测进程拒绝 Python 网络连接与解析，避免意外下载。HTTP 主服务不受该进程内保护影响。它不是宿主机防火墙，不能替代容器断网验收。

## 合并前验证记录

前端使用新建的依赖目录完成 `npm ci`、生产构建、69 项测试，不依赖旧工作树的 node_modules。Linux 开发机上检查了精确依赖版本、实际 GPU 运算、三方法默认配置预检、750 张原图、525/225 训练测试样本及增强包复用。新增启动器用替身 Docker CLI 测试离线参数、失败阻断和容器归属检查；这不等同于真实容器生命周期通过。

补充检查已通过：前端 12 个运行文件完整且两次构建运行资源指纹一致；后端原有 30 项、部署检查 6 项、启动器 8 项测试通过。离线 worker 保护开启时三方法各 2 轮训练、225 样本独立评测、三次真实选图预测、20 行分类/分域 CSV、JSON 和模型下载通过；重启前后 6 个模型条目、3 个测试集保持一致。临时测试服务已关闭，正式状态目录没有变化。

**没有构建镜像、导出镜像/资源包，也没有替换正式服务。** 本地 Docker 引擎未启动；干净 Linux 镜像依赖安装、`pip check`、无网络训练及目标 RTX 5880 验证仍要在合入后端更新后执行。Python 3.9 和旧算法依赖是兼容性冻结，不代表已完成安全更新审查。

## 合并后的交付待办

1. 最新隐私更新已在独立 integrated 工作树合入；40/20/20、本地 1 轮及模型验证口径已保留并回归。打包时补充可信的 fedmia_local 资源，核验外置脚本依赖；新增依赖必须更新锁文件。
2. 重新运行前端 `npm ci && npm run build && npm test`；在 Linux 容器中再验证一次，不能用 Windows 构建替代。
3. 核定发布版本和预置模型记录，整理 `backend/resources`、独立增强缓存及完整选定任务目录。模型库必须有任务登记、特征包及清单，不能只复制分类头文件。
4. 获得打包指令后才构建镜像、做完整断网短流程、停止/失败/重启/移动路径检查，导出归档和逐文件哈希。三方法准确率门槛按用户指示暂不重验。
5. 目标机确认 NVIDIA Container Toolkit、Docker 权限和磁盘；首次导入后再做 RTX 5880 容器 GPU 与短流程验收。

2026-09-16 已合入后端 `970fa95`、前端 `87b6a96`，合并版本前端 72 项、后端 39 项测试及军机真实短流程通过，详见 [合入记录](../docs/UPSTREAM_INTEGRATION.md)。隐私模块只做了测试桩接口与页面回归，尚无真实资源验证；现有部署预检仍只覆盖军机链路，不能据此认定隐私模块已可离线交付。

Docker 模板使用 `FS_FEDMIA_LOCAL_ROOT=resources/fedmia_local`；后续将可信脚本、两套配置/特征、OfficeHome 图片放入 backend/resources/fedmia_local，沿用只读资源挂载。源码直接运行保持上游默认 ../fedmia_local，也可用该变量指定相对 backend 的目录。资源未就绪时隐私页显示提示，军机训练与评测仍可使用。

未来构建入口（当前不要执行）：在含 `backend/` 和 `frontend/` 的父目录运行：

```bash
docker build --platform linux/amd64 -f backend/deploy/Dockerfile -t fs-aircraft:prototype .
```

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

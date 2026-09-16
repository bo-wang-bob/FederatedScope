# 跨域联邦学习 · 后端原型

分支：`prototype/backend`。对应前端：[prototype/frontend](https://github.com/bo-wang-bob/FederatedScope/tree/prototype/frontend)。

从已部署版本 `5ac3e453d69f33596c193e15daa808aaca52865b` 拆分。保留 `federatedscope/`、训练配置、脚本、后端测试及部署参考；不包含前端工程。算法、API 和配置文件与拆分源一致，不修改运行中的服务器。

## 启动现有环境

优先复用 4090lziy 已验证的 GGEUR Python/PyTorch 环境；先激活环境，再在本分支根目录运行：

```bash
python -m pip install --no-deps -e .
export FS_PLATFORM_RESOURCES=.
export FS_PLATFORM_DATASETS=data
python -m federatedscope.standalone_api.platform_app \
  --host 127.0.0.1 --port 8001 --state-dir exp/single_host_platform
```

以上相对路径以本后端项目根目录为基准，不依赖启动命令的当前目录。已有服务器可继续用原环境变量指定资源绝对路径，无需移动数据。状态目录应独立且可写，不要同时让两个服务管理同一个状态目录。8001 已有服务时另选未占用端口。

`setup.py` 保留原依赖声明，但包含历史版本约束；`--no-deps` 仅适用于已具备依赖的验证环境，不是全新环境安装承诺。不要为安装该原型而降级正在使用的训练环境。

检查：`curl http://127.0.0.1:8001/api/health`。无前端文件时 API 仍可独立运行，根页面不会提供界面。当前 API 没有用户认证，只绑定本机并使用 SSH 隧道或受控网络，禁止直接暴露公网。

## 对接独立前端

开发：前端使用 `FS_API_PROXY` 指向该服务。生产：先在 `prototype/frontend` 执行 `npm ci && npm run build`，在启动后端前设置：

```bash
export FEDERATEDSCOPE_FRONTEND_DIST=../FederatedScope-prototype-frontend/dist
```

后端即可从该目录提供页面。只指定已经审核的构建目录；不需要把两个分支相互合并。

## 功能与数据

- 单机模拟多客户端，保留联邦聚合、异构划分和算法比较。
- `/api/platform/catalog`、`library`、`jobs`：读取能力、模型/测试集和任务。
- `/api/platform/preflight`、`train`、`predict`、`evaluate`：预检、训练、单图预测和独立评测。
- 任务支持停止、恢复检查、仅清理自身进程及 JSON/CSV/模型/复现包导出。
- 基础特征必须完整；本架构可按配置新生成增强特征。单图验证使用关联冻结特征与分类器，不是重新提取原图特征；历史来源限制不能省略。

数据集、主干权重、训练模型、特征缓存和任务日志不随 Git 分支上传。资源路径及缓存覆盖变量见 [单机平台说明](docs/SINGLE_HOST_PLATFORM.md)。旧入口和历史方案说明保留在 [历史 README](docs/LEGACY_STANDALONE_README.md)，不是本原型的默认启动方式。

## 验证

```bash
python -m unittest discover -s tests -p 'test_single_host*.py' -v
```

拆分源已完成真实预测、独立评测和资源清理验收。本次核对后端源码与配置未变，不运行新的训练，也不为打包修改环境依赖。

## 军机演示接入（2026-09-15）

新增 `military_vit`，复用 `scripts/military_aircraft_3domain/configs/` 下的 FedAvg、FedProx、本架构配置。数据集为 MilitaryAircraft-3D，3 域（aerial/natural/recon）、5 类、750 张图，默认 15 个模拟客户端、划分种子 42、525 张训练图和 225 张测试图；ViT-B/16 冻结特征加 MLP 分类器。recon 是原数据方案中的确定性图像变换域，不代表真实侦察采集。

资源目录约定（只读复用，不复制进 Git）：

- 原图：`data/MilitaryAircraft3D`，可用 `FS_PLATFORM_DATASETS` 设置数据集父目录。
- ViT 权重：`pretrained_models/ViT-B-16.pt`，可用 `FS_PLATFORM_MILITARY_VIT_WEIGHTS` 覆盖。
- 原始特征：`$FS_PLATFORM_RESOURCES/exp/distributed_feature_cache/military_aircraft_vit_fixedsplit_v2`，可用 `FS_PLATFORM_CACHE_MILITARY_VIT` 覆盖。
- 原始缓存缺失会预检失败；本架构默认按本次配置生成增强特征，不自动接受来源不完整的历史增强缓存。基础缓存没有嵌入样本 ID，保留其来源限制说明。

独立部署模板：`scripts/federatedscope-prototype.service`，绑定 `127.0.0.1:8002`，独立状态目录 `prototype/backend/exp/platform`，前端来自 `prototype/frontend/dist`。仅通过 SSH 隧道访问；现有 8000/8001 服务不改动。默认军机训练使用原配置的 GPU 0。

```bash
python scripts/platform_acceptance.py --url http://127.0.0.1:8002 --group military_vit --method fedavg --rounds 2 --gpu 0
# 同样分别验证 fedprox 和 heterogeneous_solution
python -m scripts.platform_prediction_acceptance --url http://127.0.0.1:8002 --group military_vit --state exp/platform
```

`platform_acceptance.py` 还支持 `--name`、`--generated-per-sample`、`--generated-per-prototype`、`--target-per-class` 和 `--covariance-scale`。成功完成且实际生成增强特征的训练会自动进入 `/api/platform/catalog` 的 `augmentationSources`，供前端作为完整配置预设或严格缓存复用来源，不需要手工复制参数。

固定种子 42、相同划分和其余训练参数下，已保存 100 轮本架构增强量实验：每类目标 10、20、40、60；样本与原型候选生成数均为 20，协方差缩放 0.01。独立评测用于核对保存模型可重新加载，结果只代表当前这组运行，不能单独证明稳定提升。

25 项后端测试通过。三种方法各 2 轮真实验收结果：FedAvg 31.56%、FedProx 27.11%、本架构 60.44%；三者重新加载 final 模型后，225 张测试图的总体和分域准确率均与训练结果一致，任务进程已清理。本架构本次生成增强数据。此处只证明链路可运行，不能据短轮试跑声称稳定提升。

独立评测与单图测试复用保存的冻结 ViT 特征，并验证原图、样本及模型哈希；尚不支持任意上传图片重新运行 ViT。前端负责隐藏其他配置，后端仍保留原有能力及数据。

## 本地成员推理与可迁移路径

成员推理从后端同级的 `../fedmia_local/` 读取用户自行准备的脚本、特征和图片，目录内容不上传 Git：

```text
工作目录/
  FederatedScope-prototype-backend/
  FederatedScope-prototype-frontend/
  fedmia_local/
    show_fedmia_examples.py
    runs/no_defense/config.yaml
    runs/no_defense/ggeur_fedmia_features/
    runs/defense/config.yaml
    runs/defense/ggeur_fedmia_features/
    datasets/OfficeHomeDataset_10072016/
```

`FS_FEDMIA_LOCAL_ROOT` 可覆盖展示包目录。后端按客户端缓存攻击分数、样本映射、AUC 和分布；切换图片或防御状态不会重新训练。运行环境需要 torch、torchvision、numpy、scipy、yacs、pyyaml、pillow 以及项目已有依赖。

本地路径变量（资源、数据集、基础缓存、测试缓存、模型权重、前端构建目录、展示包）均支持绝对路径；相对路径以本后端根目录为基准。增强缓存沿用相对于 `FS_PLATFORM_RESOURCES` 的约定。程序内部解析成绝对路径是为了保证子进程和下载路径校验一致，不再写死盘符或个人服务器目录。历史训练日志中的路径不自动重写。

旧三机入口仍保留环境变量覆盖：远程 `FEDERATEDSCOPE_CLIENT_REPO` / `SUBSERVER_REPO` 及相应 `*_PYTHON`、`*_RESOURCE_REPO` 路径在目标 SSH 主机解析，默认使用登录目录下的 `FederatedScope-prototype-backend` 和其中的 `.venv`。现有三机部署应显式保留原环境路径；本地根节点默认使用当前后端及当前 Python 环境。systemd 模板用 `%h` 表示服务用户主目录；部署时按实际安装目录和 Python 环境设置，不能给 `WorkingDirectory` 填普通相对路径。

数据集、`fedmia_local/`、`exp/`、权重和缓存仅在本地保留；此分支只提交代码、测试与配置说明。

# FederatedScope 统一安全单机版

本分支面向单机联邦学习模拟，将异构处理、隐私保护和后门攻击防御串联为一条可配置的训练流程。不创建真实分布式客户端，服务端和多个逻辑客户端在同一进程中完成通信与训练模拟。

## 核心流程

一次实验由两个阶段组成：

1. 特征统计阶段：各客户端提取本地特征并交换必要的统计量，服务端聚合统计信息，客户端据此扩充本地特征分布。
2. 正常训练阶段：客户端执行本地训练，上传模型更新，服务端完成聚合并进入下一轮。

在此基础上提供三类能力：

- 异构解决方案：通过统一特征空间、类别统计聚合和本地特征扩充缓解跨客户端数据分布差异。
- 隐私保护方案：客户端上传参数前执行本地自适应裁剪和高斯加噪，裁剪状态不会在客户端之间共享。
- 攻击防御方案：在特征统计阶段过滤异常统计量，在正常训练阶段过滤异常模型更新。

隐私评估与后门攻击实验互斥。一次运行只能选择其中一种安全模式；执行后门攻击时不会同时执行隐私攻击。

## 环境

项目验证使用：

```bash
/root/.local/share/mamba/envs/pfedba/bin/python
```

安装当前源码：

```bash
/root/.local/share/mamba/envs/pfedba/bin/python -m pip install -e .
```

## 单机配置

核心配置位于 `scripts/standalone_configs/`：

- `heterogeneity.yaml`：仅运行异构解决方案。
- `privacy_protection.yaml`：启用本地自适应裁剪与加噪。
- `privacy_membership_attack.yaml`：成员关系隐私评估。
- `privacy_membership_attack_protected.yaml`：带本地隐私保护的成员关系评估。
- `privacy_property_attack.yaml`：客户端属性隐私评估。
- `privacy_property_attack_protected.yaml`：带本地隐私保护的属性评估。
- `privacy_reconstruction_attack.yaml`：训练数据重建评估。
- `privacy_reconstruction_attack_protected.yaml`：带本地隐私保护的数据重建评估。
- `backdoor_attack.yaml`：后门攻击模拟。
- `backdoor_attack_defended.yaml`：启用两个阶段攻击防御的后门实验。

运行示例：

```bash
/root/.local/share/mamba/envs/pfedba/bin/python run.py \
  --cfg scripts/standalone_configs/heterogeneity.yaml
```

运行前需要根据本机环境修改配置中的数据集目录、预训练模型目录和设备编号。

## 启动前后端实验平台

前端通过单机控制 API 启动或停止现有训练入口，使用快照和 SSE 接收真实训练事件。地图中的中央服务器、域子服务器和 60 个客户端是训练过程的可视化投影；后端仍由一个本机进程模拟，不会创建真实分布式客户端。

### 1. 准备运行环境

在仓库根目录安装 Python 源码依赖，并在前端目录安装 Node.js 依赖：

```bash
cd /root/project/FederatedScope
/root/.local/share/mamba/envs/pfedba/bin/python -m pip install -e .

cd frontend
npm install
```

数据目录必须包含 `Art`、`Clipart`、`Product` 和 `Real_World` 四个子目录。异构协同与后门实验还需要本地特征模型文件；隐私实验按其模板加载相应的本地图像模型。

### 2. 终端一：启动后端

```bash
cd /root/project/FederatedScope
export FEDERATEDSCOPE_DATA_ROOT=/path/to/OfficeHomeDataset_10072016
export FEDERATEDSCOPE_MODEL_PATH=/path/to/open_clip_vitb16.bin
export FEDERATEDSCOPE_API_STATE_DIR=/path/to/writable/experiment-state

/root/.local/share/mamba/envs/pfedba/bin/python -m \
  federatedscope.standalone_api.app --host 127.0.0.1 --port 8000
```

`FEDERATEDSCOPE_API_STATE_DIR` 可省略，默认使用 `exp/standalone_api`。可在另一个终端确认服务状态：

```bash
curl http://127.0.0.1:8000/api/health
```

### 3. 终端二：启动前端

```bash
cd /root/project/FederatedScope/frontend
npm run dev
```

浏览器访问 `http://127.0.0.1:5173/`。开发服务器默认把 `/api` 转发到 `http://127.0.0.1:8000`；若后端使用其他地址，启动前设置：

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

### 4. 启动首个实验

1. 在“场景与异构分析”中设置狄利克雷参数和随机种子，预览后应用场景。
2. 系统扫描实际四域图像，生成包含 60 个客户端精确样本路径的数据划分清单和数据指纹。
3. 在“实验配置”中选择单一实验类型，设置参数并执行“运行预检”。
4. 预检通过后启动实验，页面自动进入运行监控。
5. 可在监控页停止任务；任务状态、结构化指标、事件和完整训练日志均持久化到状态目录。

如果数据集、数据划分清单、模型、设备或磁盘空间不满足要求，后端会拒绝启动并返回具体预检结果，不会生成伪训练轮次。后门实验配置会强制关闭隐私攻击及上传加噪流程。

### 5. 生产构建

```bash
cd /root/project/FederatedScope/frontend
npm run build
npm run preview
```

构建产物位于 `frontend/dist/`。预览服务仍需要可访问的后端 API；跨地址部署时使用 `VITE_API_BASE_URL` 构建前端。

## 测试

```bash
/root/.local/share/mamba/envs/pfedba/bin/python -m pytest -q \
  tests/test_unified_security.py tests/test_standalone_api.py
```

该测试验证模式互斥、本地自适应加噪、隐私评估钩子、图像分支保护，以及两个训练阶段的异常客户端过滤。

## 目录

```text
federatedscope/                 核心框架和统一训练实现
federatedscope/standalone_api/  单机实验控制 API 与任务管理
scripts/standalone_configs/     单机实验配置
tests/                          核心回归测试
docs/                           使用说明
run.py                          运行入口
setup.py                        安装配置
```

更完整的配置说明见 `docs/统一安全单机方案使用说明.md`。

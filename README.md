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

## 测试

```bash
/root/.local/share/mamba/envs/pfedba/bin/python -m pytest -q \
  tests/test_unified_security.py
```

该测试验证模式互斥、本地自适应加噪、隐私评估钩子、图像分支保护，以及两个训练阶段的异常客户端过滤。

## 目录

```text
federatedscope/                 核心框架和统一训练实现
scripts/standalone_configs/     单机实验配置
tests/                          核心回归测试
docs/                           使用说明
run.py                          运行入口
setup.py                        安装配置
```

更完整的配置说明见 `docs/统一安全单机方案使用说明.md`。

# 隐私历史回放部署

本次交付包含真实 FedMIA 历史结果回放兼容，不启动新训练或生成特征。前端使用配套 `prototype/frontend`，资源独立部署，不能只复制代码。

## 资源目录与配置

默认放置于后端的 `resources/fedmia_local/`（与 Docker 一致）：

```text
fedmia_local/
  show_fedmia_examples.py
  runs/
    no_defense/config.yaml
    no_defense/ggeur_fedmia_features/client_*_features_round*.pt
    defense/config.yaml
    defense/ggeur_fedmia_features/client_*_features_round*.pt
  datasets/OfficeHomeDataset_10072016/<域>/<类别>/<图片>
```

设置 `FS_FEDMIA_LOCAL_ROOT=resources/fedmia_local`（相对 backend 根目录，不是当前 shell 目录）；可使用绝对路径覆盖。现有部署为 `/root/autodl-tmp/FederatedScope/resources/fedmia_local`，使用 `FEDERATEDSCOPE_GGEUR_LIGHTWEIGHT=1` 加载轻量数据模块。`FEDERATEDSCOPE_FRONTEND_DIST` 指向配套前端构建目录。

资源包含动态 Python 脚本及 PyTorch 序列化文件，只能加载可信来源。数据、特征、模型、运行日志不提交 Git。当前用户资源包含两份配置、240 个特征文件（60 客户端 × 2 轮 × 2 组）、15,590 张 OfficeHome 图片，已在现有 GGEUR 环境验证，无需联网下载或生成数据。

## 外置脚本的只读兼容补丁

用户原始脚本 `load_cfg()` 中的 `cfg.freeze()` 默认会向旧实验目录写入配置。服务调用该方法之前，须将这一行改为 `cfg.freeze(save=False, inform=False)`；后端后续冻结配置已禁用保存。此补丁不修改算法。

版本化补丁位于 [deploy/fedmia-readonly-config.patch](../deploy/fedmia-readonly-config.patch)。在含 `show_fedmia_examples.py` 的资源目录内备份原文件，再执行 `patch --forward -p1 < <backend绝对目录>/deploy/fedmia-readonly-config.patch`。已应用补丁的目录不要重复应用。原始压缩包应保留，修改后不要改写历史 config.yaml。

本次原脚本 SHA256 为 `530107bc5e7679cc3f49cbdc16a3f0030e59cf17e7b26a962b278efb7bf1f513`，部署修正版为 `24c47c2c434195bfbad196f05b953d73a9bbe01a4638ac7a5fa5180c5b24cd2e`。其他脚本版本必须审查后再适配，不盲目替换。

## 结果范围

交付目录准备脚本会在副本中补上脚本导入根、两组实验根和数据根的相对定位，并记录修改前后哈希；原压缩包、正在运行的资源和两份历史配置均不改动。详见 [目录规范](PACKAGE_DIRECTORY.md)。

- 当前无防御历史结果采用 indexed 统计、防御采用 global 统计，不能直接把其差异归因于防御效果。
- 回放适配检查两组重建的图片划分、各轮标签、样本顺序与分数数量，仅对可确认的共同前缀样本输出对照。混合/生成样本、不同轮次、不同划分、标签乱序和非法分数仍拒绝。
- 这批资源每客户端使用共同 27 个成员、27 个非成员；非成员来自保存的全局测试顺序前缀（当前为 Art 域），不代表全部 4,679 个测试样本。没有原始图片 ID 清单，图片按保存配置重建并核对标签。
- AUC、分布及 TPR 均使用相同共同样本集合；响应保留原始数量和 alignment 说明。TPR 使用实际 FPR≤1% 的可达阈值；27 个负样本下实际 FPR 为 0，不伪称精确 1%。
- 严格防御增益对比仍需攻击设置一致、样本身份映射完整的成套结果。

## 验证

```bash
python -m unittest tests.test_single_host_platform tests.test_single_host_privacy tests.test_privacy_replay -q
```

25 项回归通过；服务器独立进程真实回放验证全部 60 个客户端，图片无缺失，两份源配置哈希不变。正式服务客户端切换、成员/非成员图片、分布和指标均已验证；原有 46 个任务、24 个模型、12 个测试集保留。未重新运行准确率实验或验证增益门槛。

最终 Docker 镜像、断网容器及 RTX 5880 现场仍需独立验收；现有环境通过不等同于离线交付验收完成。

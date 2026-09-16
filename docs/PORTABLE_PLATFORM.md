# 军机三方法：可迁移运行

范围为 MilitaryAircraft-3D / 冻结 ViT 特征 / FedAvg、FedProx、本架构。独立评测和选图验证继续使用保存的特征与分类头。

## 默认参数

军机三方法默认 100 轮、15 个客户端、每轮全部参与、种子/划分种子 42、batch 16、学习率 0.00005。本架构本地轮数 1、每类目标 40、样本生成 20、原型生成 20、协方差缩放 0.01；两个基线保留本地轮数 5，不使用增强。

前端以 API catalog 为默认参数来源。切换算法恢复该算法默认值；恢复已编辑草稿保留编辑值。仍可手动调整通信轮数等参数。

本架构的 `augmentationMode=auto` 在预检时固定执行选择：优先查找匹配的完整生成记录，其次检查独立缓存包，否则按配置生成。复用必须通过文件哈希、样本来源、客户端、种子和生成参数检查；损坏或来源不匹配的缓存不会静默用于训练。正式启动使用预检固定的选择。

## 目录

```text
release/
  frontend/dist/
  backend/
    scripts/start_platform.py
    resources/
      datasets/MilitaryAircraft3D/
      exp/distributed_feature_cache/military_aircraft_vit_fixedsplit_v2/
      caches/military_vit_default/          # 可选：独立增强缓存包
        augmentation.json
        client_000001.pt ... client_000015.pt
      models/ViT-B-16.pt                    # 保留主干资源；缓存训练不加载它
      fedmia_local/                        # 隐私脚本、两组特征、OfficeHome 图片
    exp/platform/                          # 实验、模型、测试特征、日志
```

启动：`python backend/scripts/start_platform.py --host 0.0.0.0 --port 8002`。脚本路径按实际位置填写，可以从任意工作目录调用。默认状态目录为 backend/exp/platform，前端为兄弟目录 frontend/dist。

`FS_PLATFORM_RESOURCES`、`FS_PLATFORM_DATASETS`、`FS_PLATFORM_CACHE_MILITARY_VIT`、`FS_PLATFORM_MILITARY_VIT_WEIGHTS`、`FS_PLATFORM_AUGMENTED_MILITARY_VIT`、`FEDERATEDSCOPE_FRONTEND_DIST` 和 `--state-dir` 的相对值一律相对 backend，而不是命令执行目录；仍允许显式绝对路径覆盖。执行配置会记录相对于 backend 的路径。运行进程身份记录和诊断日志可能含实际绝对路径，它们不是重载资源的依赖。

## 缓存与历史模型迁移

从一个已经完成且来源可验证的生成任务导出独立缓存：

```bash
python backend/scripts/export_platform_cache.py --job <任务ID>
```

默认读取 backend/exp/platform，输出 backend/resources/caches/military_vit_default。输出目录必须不存在；原记录和文件保持不变。也支持 `--state` / `--output`。缓存包包含完整请求及逐客户端文件哈希，新平台无需导入旧任务即可复用它。

需要保留已有模型库时，复制所选任务的完整目录到 exp/platform/jobs/，包括 job.json、data_manifest.json、checkpoints、配置和结果。旧生成任务的绝对缓存路径按已登记任务 ID 安全重定位，重新核验每个文件。新模型/测试特征包保存固定的主干标识和 featureSpace；旧模型与新测试包混用时必须通过缓存特征指纹、类别、数据集及样本不交叉检查。

没有数据文件不应只靠复制源码启动。基础训练/测试特征仍必须完整；本改造不新增原图特征生成或任意图片上传。

## 验证

`python -m unittest discover -s tests -p 'test_single_host*.py' -v`。

`scripts/portable_platform_acceptance.py` 使用指定军机数据、缓存和一个已有生成任务，复制到独立临时目录，再检查三方法 2 轮训练、独立评测、生成/复用、整包迁移、旧模型兼容与单图预测。它不判定准确率提升指标，不修改正式平台。

本次已在 4090 环境完成上述真实短流程和独立缓存包+空状态目录运行。Docker 镜像制作及 RTX 5880 现场验收另行完成；本次没有替换正在运行的服务。

目录准备和迁移检查见 [项目交付目录](PACKAGE_DIRECTORY.md)。隐私默认路径统一为 resources/fedmia_local，并随 FS_PLATFORM_RESOURCES 解析；显式 FS_FEDMIA_LOCAL_ROOT 仍优先。旧 ../fedmia_local 布局须显式设置变量或复制到新目录。来源限制见 [隐私回放部署](PRIVACY_REPLAY.md)。目录整理不代表已构建或导出 Docker 镜像。

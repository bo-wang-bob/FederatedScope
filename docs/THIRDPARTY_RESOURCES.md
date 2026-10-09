# 三方测试资源接入

本次为增量接入，原 MilitaryAircraft-3D、Malimg、上传数据集、隐私和后门资源与实验记录不迁移、不删除、不改划分。

| 数据集 | 训练模型 |
| --- | --- |
| Office-Home | ViT、CNN、MLP-Mixer |
| Digits-3Domain | ViT、CNN |
| MDSent | RNN、LSTM |

前端默认展示完整后端目录；只有明确设置 `VITE_DEMO_SCOPE=aircraft` 才启用旧演示筛选。默认新训练仍优先选择军机数据集。

## 资源目录

所有路径相对 `backend/resources`，服务器启动仍使用 `backend/deploy/start-server.sh`，无需新增环境变量。

```text
resources/
  thirdparty_resources.json
  datasets/
    OfficeHomeDataset_10072016/
    digit_three_domain/        # 原固定 train/test 清单
    sentiment/
  pretrained_models/
    open_clip_vitb16.bin
    convnext_base-6075fbad.pth
    mixer_b16_224_complete.pth
    nlptown_bert_base_multilingual_uncased_senti/
  exp/
    distributed_feature_cache/ # OfficeHome、Digits ViT、MDSent 已有特征
    thirdparty/
      features/digit3_cnn/
```

`thirdparty_resources.json` 为可选资源映射，仅对列出的组生效；未列出的军机及上传数据仍使用原目录。`dataset`、`features`、`manifest` 均必须为资源根目录内的相对路径，禁止越界。

训练继续使用冻结特征与原分类头/聚合实现。缓存校验、训练/测试隔离、标签顺序验证不绕过。模型保存后可独立评测；图像组支持原图预览及冻结特征单图验证，文本组使用评测入口。

DomainNet 已按要求退出平台，不再提供其训练、测试入口或专用资源。`domainnet` 通用文件夹加载器仍供军机、隐私/后门和上传数据使用，不能按文件名包含该词批量删除。共享 ViT、CNN、MLP-Mixer 权重保留供其他数据集使用。

资源接入脚本 `scripts/add_thirdparty_resources.py` 只添加不存在的目标。同盘只读资源使用硬链接，打包读取到的仍是完整普通文件，不依赖原目录符号链接。不要原地编辑这些共享只读输入；需要变更时发布到新目录。

Digits CNN 在新目录按固定清单生成特征；原始划分、测试缓存及其他现有数据均保持不变。

`scripts/smoke_thirdparty_catalog.py` 必须使用全新的隔离状态目录；其单轮测试用于检查预检、训练、保存、评测和预测链路，不是准确率验收结论，不应混入正式实验记录。

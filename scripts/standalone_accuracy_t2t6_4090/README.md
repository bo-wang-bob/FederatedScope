# T2-T6 4090 最终运行包

本目录用于在 4090 根服务器上完成 T2-T6。正式实验均为单机独立运行，三种方法依次为 FedAvg、FedProx、平台，正式轮数均为 100。

缓存边界：三种方法可以读取相同的原始特征缓存，以避免重复运行 ViT、ConvNeXt、Mixer 或 BERT 特征提取；只有平台方法会生成并读取增强数据缓存。脚本会显式关闭 FedAvg、FedProx 的增强数据缓存，并在日志中检查这一点。

## 文件

- `final_experiment.json`：唯一的最终参数清单，同时保存现有正式结果及其来源。
- `preflight_4090.py`：检查 GPU、Python 3.9、依赖、数据集、预训练权重、配置和任务文件。
- `run_t2t6_4090.py`：先为平台执行 1 轮缓存预热，再按 FedAvg、FedProx、平台顺序完成正式训练。
- `summarize_results.py`：汇总准确率、绝对提升和纯训练用时。
- `run_all_4090.sh`：一条命令执行全部检查与训练。
- `sync_to_4090.ps1`：从当前 Windows 项目目录向 4090 同步当前 `federatedscope` 代码、本运行包、配置和任务文件；数据集、缓存和预训练权重不随代码重复传输。

## 在 4090 上运行

在 `FederatedScope` 项目主目录执行：

```bash
bash scripts/standalone_accuracy_t2t6_4090/run_all_4090.sh
```

仅运行一个测试用例：

```bash
/root/miniconda3/envs/fs/bin/python \
  scripts/standalone_accuracy_t2t6_4090/run_t2t6_4090.py \
  --python /root/miniconda3/envs/fs/bin/python \
  --only t2_digit3_vit
```

其余可选名称为 `t3_officehome_cnn`、`t4_officehome_mlp`、`t5_mdsent_rnn`、`t6_mdsent_lstm`。

## 输出

- 正式日志：`exp/t2t6_4090_final/logs/*_formal_*.log`
- 汇总结果：`exp/t2t6_4090_final/summary.json`
- 平台增强缓存：`exp/t2t6_4090_platform_cache/<测试用例>/`
- 任务自适分布：`exp/test_outline_validation/T-02至T-06/training_distributions/client_*.json`
- 环境检查：`exp/t2t6_4090_preflight.json`

脚本不在线下载模型。这样可以避免正式测试时间被模型下载卡顿；若预检显示某个权重缺失，应先补齐该文件，再开始正式训练。

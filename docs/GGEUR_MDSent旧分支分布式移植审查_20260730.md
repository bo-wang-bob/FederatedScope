# GGEUR MDSent 旧分支分布式移植审查（2026-07-30）

## 结论

本次移植以 `origin/feature/ggeur-backdoor-research` 为行为基准，重点参考：

- `79ae3d7`：加入 MDSent 文本数据和冻结 BERT 特征提取；
- `6654a33`：加入 RNN/LSTM 及其对比实验配置。

旧实现不是端到端训练 BERT，也不是训练完整文本 RNN/LSTM。它的实际计算链为：

1. 使用本地预训练的
   `nlptown/bert-base-multilingual-uncased-sentiment`；
2. 冻结 BERT 全部参数，只提取 CLS 的 768 维句向量；
3. 将句向量视为长度为 1 的序列；
4. 只训练单层 RNN 或 LSTM（hidden=256）及四分类线性层；
5. GGEUR 只在冻结句向量空间生成增强特征；
6. FedAvg/FedProx/FedOpt 聚合轻量头参数；
7. FedProto 在轻量头的 256 维隐藏表示空间计算逐类原型。

当前实现已恢复以上语义，并将它变成
“120 个逻辑客户端 → 4 个子服务器 → 1 个根服务器”的三机分层训练。

## 旧分支与当前分布式实现的逐项对照

| 项目 | 旧分支 | 当前分布式实现 |
|---|---|---|
| 数据集 | MDSent 四域：books、dvd、electronics、kitchen | 相同 |
| 标签 | rating 1/2/4/5 映射为四类 | 相同 |
| 划分 | 每域先 train/test，再按 Dirichlet α=0.01 切客户端 | 相同；分布式客户端用 `data_idx` 取同一份逻辑切片 |
| 文本编码器 | 冻结 BERT | 相同；编码结果可跨轮缓存 |
| 池化 | CLS，768 维 | 相同 |
| 训练模型 | 单层 RNN/LSTM + Linear | 相同 |
| 输入序列长度 | 1 | 相同 |
| hidden | 256 | 相同 |
| 本地步数 | 每轮 1 个 local epoch | 相同 |
| RNN Adam LR | 1e-3 | 相同 |
| LSTM Adam LR | 0.1 | 已恢复为相同 |
| GGEUR 每类目标数 | 200 | 相同 |
| 模型通信 | 轻量头状态 | 相同；不传 BERT 参数 |
| 根聚合 | 客户端样本数加权 | 两级样本数加权，数学上等价 |
| FedProto | 隐藏表示逐类均值和计数 | 已恢复；两级保留逐类计数，数学上等价 |

## 审查中发现并修复的问题

### 1. LSTM 正式配置学习率漂移

旧分支 LSTM 使用 Adam `lr=0.1`，当前五种方法的 LSTM 源配置曾被统一改成
`1e-3`。这不是分布式所需修改，会改变结果。现已将
`mdsent_lstm/{fedavg,fedprox,fedproto,fedopt,ggeur}.yaml`
全部恢复为 `0.1`。

### 2. FedProto 在分布式移植中丢失了核心算法链

此前的分布式代码没有完整执行以下链路：

- 客户端每轮在 RNN/LSTM hidden 表示上计算逐类原型；
- 上传原型及逐类样本计数；
- 子服务器按逐类计数汇总；
- 根服务器再次按逐类计数汇总；
- 下一轮向客户端广播全局原型并计算原型正则项。

现已补齐。子服务器不再使用“整个客户端训练样本数”错误加权原型，而是保留
`class mean + class count` 这组充分统计量，因此两级汇总与扁平汇总完全一致。

### 3. 旧分支 FedProto 广播包会污染 `state_dict`

旧分支把 `fedproto_global_prototypes` 直接追加到普通模型 `state_dict`，客户端
`load_state_dict` 会遇到额外键。当前移植保留其算法含义，但将传输包明确分为：

```text
{
  "mlp": <RNN/LSTM head state_dict>,
  "fedproto_global_prototypes": <class -> hidden prototype>
}
```

客户端更新也使用同样的显式信封，避免元数据被当作模型参数。

### 4. 分布式模型广播缺少张量反序列化

`Message.transform(to_list=True)` 会把 `model_para` 中的每个张量叶子序列化为
bytes。子服务器按设计原样转发，但此前普通训练回调没有在
`load_state_dict` 前递归还原张量。加载失败只写 debug 日志，客户端随后会继续
训练自己的旧本地模型，因此表面上能完成 100 轮，却不是真正的逐轮 FedAvg。

现已在客户端普通训练回调入口统一递归还原参数树；根服务器接收子服务器
FedProto 原型时也执行相同的张量还原。新增测试覆盖“模型状态序列化 →
递归反序列化 → `load_state_dict`”完整参数一致性。

## 新的三机不可变配置

已生成但未启动：

```text
run_id = mdsent_reference_distributed_20260730_v1
```

其中只包含用户指定的两个 GGEUR case：

- `mdsent_rnn_ggeur`
- `mdsent_lstm_ggeur`

配置审计结果：

- 每个 case 120 个逻辑客户端；
- 4 个子服务器，每个负责 30 个客户端；
- 100 个正式训练轮次；
- 根服务器：`10.112.81.135:60050`；
- 子服务器：`10.129.248.111:61000-61003`；
- 客户端 1-60：`10.129.222.189:20000-20059`；
- 客户端 61-120：`10.112.81.135:20000-20059`；
- 所有训练业务端点均为真实 `10.x`；
- 客户端配置为 CPU head-only；
- SSH 不出现在训练业务端点中，只用于管理。

准确率队列 `final_remaining_20260723_v1` 继续保持暂停，本次没有启动远程进程。

## 为什么之前一次要三小时以上

已归档的 100 轮 RNN 日志给出的实际耗时如下：

| case | 每轮中位数 | 每轮均值 | 100 轮训练阶段 |
|---|---:|---:|---:|
| FedAvg | 125.5 s | 129.9 s | 3.61 h |
| FedOpt | 117.7 s | 118.1 s | 3.28 h |
| FedProto（旧的错误移植） | 116.8 s | 116.6 s | 3.24 h |
| FedProx | 117.3 s | 120.4 s | 3.35 h |
| GGEUR | 121.6 s | 123.5 s | 3.43 h |

这不是在训练完整 BERT，也不是网络断连。主要原因是：

1. 120 个逻辑客户端被实现为 120 个独立 Python/gRPC 进程；
2. 客户端只分布在两台计算机上，每台同时承载 60 个 CPU 训练进程；
3. 每轮需要处理约 58,142 个原始样本，GGEUR 约 95,200 个增强样本；
4. 每一轮都有 120 客户端同步屏障，耗时由最慢客户端决定；
5. 第三台机器只做子聚合，不承担客户端头部训练，不能把计算时间直接除以三；
6. 已设置 OMP/MKL/OpenBLAS/NUMEXPR 每进程单线程，因此不是 BLAS 线程爆炸，
   而是客户端进程数量、CPU 调度和同步屏障的固定代价。

所以“三机”提高的是拓扑真实性、并发承载和分层聚合能力，不会自动让
120 个客户端的同步训练比旧的单进程模拟更快。

若必须将 100 轮压回 1-2 小时，同时保持算法和样本不变，优先方案是增加承担
客户端计算的物理节点，或实现同一进程内的多客户端批量执行器；直接减少轮数、
减小样本数或增大 batch size 都会改变正式实验口径，应单独作为加速变体。

## 结果有效性处置

- 已归档的五个 RNN case 都经过了缺少模型广播反序列化的旧路径，不能作为
  正式分布式聚合结果；应保留为历史失败尝试，并使用新的 run_id 重跑。
- 其中 `mdsent_rnn_fedproto` 还额外缺少旧分支真实的隐藏表示原型交换。
- 尚未完成的 LSTM case 应使用已恢复的 `lr=0.1` 新配置运行。
- 如果后续要做五方法完整对比，应从修复后的源配置重新生成新的 run_id，
  不复用 `final_remaining_20260723_v1` 的不可变生成配置。

## 本地验证

在 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` 环境下完成：

- Python 静态编译检查；
- 嵌套模型参数样本加权测试；
- 模型张量经过两级消息序列化后的子服务器与根服务器加权测试；
- FedProto 客户端到子服务器的逐类计数汇总测试；
- FedProto 子服务器到根服务器的二级汇总等价性测试；
- FedProto 原型确认为 head hidden 维度而非 BERT 输入维度；
- 模型张量经过消息层序列化和递归还原后可严格加载且数值不变；
- RNN/LSTM 正式源配置学习率一致性测试；
- 新 run 的 120 客户端、4 子服务器和全 `10.x` 业务端点审计。

验证结果：`hierarchical tests: PASS`。

## 启动前资源阻塞

本地执行以下资源预检：

```powershell
python scripts/validate_mdsent_setup.py `
  --data-root data/sentiment `
  --model-dir pretrained_models/nlptown_bert_base_multilingual_uncased_senti
```

当前结果为失败：8G 工作区的 `data/sentiment` 下缺少 `books` 等四个域目录，
同时工作区中也未找到上述 BERT 模型目录。新 run 因此保持未启动状态。

这属于 8G 节点资源缺失，不是三台机器的训练业务连接故障。正式启动前必须：

1. 将 MDSent 四域恢复到
   `D:/Projects/FederatedScope/data/sentiment/{books,dvd,electronics,kitchen}`；
2. 将冻结 BERT 模型恢复到
   `D:/Projects/FederatedScope/pretrained_models/nlptown_bert_base_multilingual_uncased_senti`；
3. 通过同一预检；
4. 再检查 4090 上对应的
   `/root/autodl-tmp/datasets/sentiment` 与
   `/root/autodl-tmp/models/nlptown_bert_base_multilingual_uncased_senti`。

不得用缺失资源时产生的启动失败记录替代算法或连接验证。

# `datapoison` 与 `attack` 分支合并指南

> 基于仓库 `bo-wang-bob/FederatedScope` 在 2026-08-14 的远端分支状态编写。
>
> - `datapoison`: `a74b23cd146b2daab74b9be6e79cf9ff0a0703ba`
> - `attack`: `d660d5f4c8ecce8ecb62f4e2fb064421b87c25aa`
> - 共同祖先: `e354126c26f93809a19d89c9ad05d6399eaab43e`

## 1. 结论先行

建议从 `origin/attack` 创建新版集成分支，再把 `origin/datapoison` 合入，并对 GGEUR 客户端、GGEUR 服务端和自适应 DP 实现进行人工重构。不要对冲突文件整体选择 `ours` 或 `theirs`。

选择 `attack` 作为基线的原因是：

1. 它保留了共同祖先中较新的 GGEUR 缓存、OfficeHome manifest 和客户端评估逻辑；`datapoison` 的大文件改写删除了其中一部分能力。
2. 它的自适应加噪发生在客户端上传边界，以 `local - global` 的模型更新为保护对象，并将裁剪历史保留在客户端本地，更符合“本地用户上传参数时增加自适应加噪”的目标。
3. 它已经包含 FedMIA、Meta-PPA、GRNN、自适应加噪和两个新增单元测试，且大部分新增代码是独立模块，适合作为隐私实验底座。
4. `datapoison` 的核心价值集中在 GGEUR 路径上的后门/数据投毒实现、攻击评估，以及 MultiMetric 两阶段防御。这些能力可以按明确的钩子移植到 `attack` 的主流程中。

### 1.1 本次合并的范围

新版只支持 `federate.mode: standalone`：单个 Python 进程中创建多个逻辑客户端，模拟特征交换、客户端本地训练和服务端聚合。不新增真实多进程、多端口或多机器客户端实现，也不把分布式 quorum、故障注入和跨机器 collector 作为新版主流程依赖。

攻击实验严格分成两种互斥模式：

- 后门模式：运行 A3FL/CERBERUS/SABRE/LabelFlip/LIE、MultiMetric 两阶段防御，以及可选的自适应上传加噪；不得同时启动 FedMIA、PPA 或 GRNN。
- 隐私模式：运行 FedMIA、PPA 或 GRNN，并对比是否启用自适应上传加噪；不得同时启动任何后门/数据投毒方法，默认也不启用仅用于后门防御的 MultiMetric。

自适应加噪是隐私防御，不属于“隐私攻击”，因此它可以在后门模式中作为额外上传保护开启；但后门模式不计算任何隐私攻击指标。

### 1.2 目标主流程

公共 GGEUR 阶段应固定为：

```text
阶段 0：GGEUR 特征统计与交换
  Server 广播初始化信息
    -> Client 提取特征并生成本地统计
    -> backdoor 模式可选：数据投毒修改统计/标签
    -> Client 上传 means/covs/counts/prototypes
    -> backdoor 模式可选：MultiMetric-Stats 过滤异常统计
    -> privacy 模式：直接使用干净统计，不运行 MultiMetric-Stats
    -> Server 聚合全局协方差/原型并回传
    -> Client 完成特征增强

阶段 1..T：standalone 模拟的正常联邦训练
  Server 依次驱动同一进程中的逻辑 Client
    -> Client 加载 global state
    -> Client 本地训练并形成 update delta
    -> 可选：自适应裁剪 + 高斯噪声（最后一道上传边界）
    -> Server 收集所有逻辑客户端更新并聚合
```

随后根据实验模式进入不同分支：

```text
后门模式：
  后门/投毒训练 -> 可选 DP -> MultiMetric-Model -> 聚合 -> ACC + ASR
  privacy attack plugins 必须为空

隐私模式：
  正常本地训练 -> 可选 DP -> FedMIA/PPA/GRNN -> 普通聚合
  backdoor method 必须为空，MultiMetric 默认关闭
```

MultiMetric 是服务端后门防御，不应与后门攻击方法或隐私攻击插件共用同一个配置开关。

## 2. 两个分支的代码总结

### 2.1 `datapoison` 分支

该分支从共同祖先之后有 7 个主要开发提交和 2 个合并提交，核心改动集中在 17 个文件，约新增 6938 行、删除 1679 行。

主要能力：

- 后门与数据投毒：
  - A3FL：新增通用 trainer wrapper、攻击客户端采样、GGEUR 特征头路径的触发器搜索、毒化训练、更新增强和 ASR 评估。
  - CERBERUS：在 GGEUR MLP 头路径实现干净锚点、影子/同伴模型、触发器优化和毒化训练。
  - SABRE：实现全图加性触发器、毒化特征池、锚点约束和触发器优化。
  - Label Flip：可分别投毒阶段 0 的统计标签和阶段 1 的训练标签，并支持 update reversal。
  - LIE/ALIE：在服务端掌握同轮更新后执行协调式模型投毒。
- 后门防御：
  - 阶段 0 的 `MultiMetric-Stats`：基于协方差 trace、Frobenius norm、跨类原型一致性构造稳健特征，使用 MAD、白化和 Mahalanobis 风格分数过滤统计异常客户端。
  - 阶段 1 的 `MultiMetric-Model`：基于 Manhattan、Euclidean、Cosine 三类更新差异，动态加权并选择低异常分数客户端聚合；支持阈值模式和跨轮 EMA。
  - 同一服务端还实现 FLAME、FoolsGold、Multi-Krum、Trimmed Mean、AlignIns、MARS 等防御。
- 自适应加噪：
  - 新增 `cfg.adaptive_dp` 和 `AdaptiveDPController`。
  - 上传前对 MLP 的 `delta = local - global` 进行裁剪和高斯加噪。
- 兼容普通 FederatedScope 后门路径：
  - 新增 `a3fl_utils.py`、`a3fl_trainer.py`，修改 trainer/worker builder 和基础 Client。
  - 将 `federatedscope.main` 抽出可调用的 `main()`。

当前不足：

- `ggeur_client.py` 和 `ggeur_server.py` 是大幅改写，覆盖了另一分支依赖的同一主流程。
- `cfg.adaptive_dp` 使用进程级全局 controller。在 standalone 中，这会让多个逻辑客户端共享 raw norm 和裁剪状态，破坏客户端隔离。新版应为每个逻辑 client 实例持有独立 controller。
- `expected_clients=0` 会经 `max(1, int(...))` 变为 1，容易导致每个 round 的 clip bound 在收到第一个客户端 norm 后就更新，而不是等待预期客户端数。
- 当前 GGEUR 上传顺序先做 DP、再做 update reversal。update reversal 会在加噪后再次改写更新，破坏“DP 是最后上传边界”的设计，应调整为攻击变换先执行、DP 最后执行。
- 分支没有为 GGEUR A3FL/CERBERUS/SABRE、两阶段 MultiMetric 或自适应 DP 新增专门测试；仓库原有 `test_backdoor_attack.py` 只覆盖旧的通用后门路径。
- 阶段 0 上传的 means/covs/prototypes 没有纳入 DP；当前保护范围仅是训练阶段模型更新。

### 2.2 `attack` 分支

该分支在共同祖先后由一个大型提交构成，修改 68 个文件，约新增 14212 行、删除 91 行。

主要能力：

- 隐私攻击：
  - GRNN 梯度重构，包含 GGEUR 图像分支适配、原始参考图像、PSNR 和攻击成功率汇总。
  - 模块化 FedMIA 插件：black-box loss、gradient cosine/diff/norm、loss series、average cosine、FedMIA-I、FedMIA-II。
  - Meta-PPA/property inference。
  - standalone hook 与真实分布式 reporter/collector 两套接入方式；本次新版只采用 standalone hook。
- 隐私防御：
  - 在标准 `cfg.dp` 下增加 client-update 级 fixed/adaptive clipping 与 Gaussian noise。
  - `LocalAdaptiveClipper` 的 raw norm 历史保留在客户端进程内。
  - 支持 fixed LDP-Fed、DPFL update noise、DPFL gradient noise 和自适应 clipping 对照。
  - 默认不上传 raw norm/sanitized norm，仅上传可选的公开噪声统计。
- GGEUR 运行增强：
  - 缓存 fingerprint、缓存热启动、统计恢复。
  - OfficeHome manifest 与随机固定采样。
  - 客户端评估回传以及 DP 噪声方差汇总。
- 新增测试：
  - `tests/test_distributed_fedmia.py`
  - `tests/test_distributed_ppa.py`

当前不足：

- 不含 `datapoison` 的 GGEUR 后门攻击和 MultiMetric 两阶段防御。
- `cfg.dp.accountant` 虽可写 `rdp`，当前代码主要通过 Opacus sigma helper/经验统计工作；仅设置字符串并不等于已经完成端到端隐私会计。
- 自适应阈值由真实更新 norm 驱动。`private_clip_update=True` 目前只是元数据/意图标记，没有自动把 quantile/EMA 更新变成严格私有机制。
- 部分示例 YAML 仍保留未注册的旧 `adaptive_dp:` 块；严格配置合并时可能失败，应删除或提供兼容迁移层。
- standalone hook 与 distributed collector 可被配置同时启用，可能重复采集、重复写结果或重复 finalization。新版主流程应完全禁用 distributed collector，只保留其独立模块和已有单元测试以避免回归。

## 3. 实际 Git 冲突与语义冲突

执行以下 dry-run：

```bash
git merge-tree --write-tree origin/attack origin/datapoison
```

Git 报告 4 个真实冲突文件：

| 文件 | 冲突类型 | 处理原则 |
| --- | --- | --- |
| `federatedscope/contrib/worker/ggeur_client.py` | content | 以 `attack` 的缓存和隐私上传骨架为主，保留 standalone 路径并移植 `datapoison` 的攻击状态与训练钩子 |
| `federatedscope/contrib/worker/ggeur_server.py` | content | 以 `attack` 的 standalone 阶段/eval/隐私 hook 骨架为主，移植后门攻击状态、MultiMetric-Stats 与 MultiMetric-Model |
| `federatedscope/core/privacy/__init__.py` | add/add | 合并导出，不保留两个互相矛盾的说明 |
| `federatedscope/core/privacy/adaptive_dp.py` | add/add | 以本地 `LocalAdaptiveClipper` 为核心，按需保留纯函数，不保留跨客户端全局 controller |

此外，以下文件虽然 Git 能自动合并，仍必须人工审查：

| 文件 | 风险 |
| --- | --- |
| `federatedscope/core/configs/cfg_attack.py` | 两边都扩展同一 `cfg.attack`；需做字段并集、校验与 legacy 映射 |
| `federatedscope/core/configs/cfg_ggeur.py` | `datapoison` 删除了另一边仍在使用的缓存、manifest、quorum 等配置；自动合并会静默保留删除结果 |
| `federatedscope/core/auxiliaries/worker_builder.py` | GRNN 必须在 GGEUR registry 前特殊分派；普通 A3FL 才交给 `BackdoorServer` |
| `federatedscope/attack/worker_as_attacker/server_attacker.py` | A3FL 采样与 GRNN/FedMIA 重构逻辑共存，需确认类级职责没有交叉 |

`attack` 有而 `datapoison` 当前缺失、且 standalone 新版仍需保留的重要 GGEUR 配置包括：

- augmented feature cache：`reuse_augmented_feature_cache`、`save_augmented_feature_cache`、cache dir/version。
- 单机大客户端数模拟：`max_cross_client_prototypes_per_class`、`cross_client_prototype_seed`。
- OfficeHome：domain filter、split strategy、random sample 参数、manifest path。
- 隐私攻击适配：local decoy、MLP regularization、`grnn_ce_only_image_branch`。

`min_statistics_clients`、`min_augmentation_clients`、`min_train_updates`、阶段故障注入和真实 distributed launcher 不属于本次新增范围。为保证 `attack` 分支原有测试，可以保留现有代码和配置，但合并后的 standalone 主流程不得依赖它们。

## 4. 推荐的统一模块边界

### 4.1 不要再把所有逻辑堆进 callback

建议在合并时形成以下内部接口，即使第一版仍保留在同一文件中，也应按这些职责分段：

```python
# client-side
prepare_round0_statistics()      # 特征提取与干净统计
apply_statistics_attack(stats)  # LabelFlip 等阶段 0 攻击
train_local_model(global_state)  # 正常训练或后门攻击训练
apply_model_attack(update)       # update reversal 等上传前攻击
protect_upload(update)           # 自适应 DP，必须是最后的模型变换
build_upload_envelope(...)       # 统一序列化

# server-side
parse_upload_envelope(content)
filter_statistics(stats_by_client)  # MultiMetric-Stats
filter_model_updates(updates)        # MultiMetric-Model
aggregate_filtered_updates(updates)
evaluate_clean_and_backdoor_metrics()

# privacy-mode only
run_privacy_attack(server_visible_upload)
```

后门模式不得调用 `run_privacy_attack()`；隐私模式不得调用 `apply_statistics_attack()`、`apply_model_attack()`、MultiMetric 或 ASR 评估。

### 4.2 统一上传消息协议

当前代码同时存在：

```text
(sample_size, model_para)
(sample_size, model_para, defense_stats)
(sample_size, model_para, gradients, last_batch_data)
```

而 `datapoison` 还会把 `a3fl`、`cerberus`、`sabre`、`label_flip` 元数据塞进模型字典。若各回调继续按 `len(content)` 自行猜测，合并后很容易把 defense stats 当成 gradients，或把攻击元数据当成 state_dict 参数。

建议统一为一个兼容 envelope：

```python
{
    "version": 1,
    "sample_size": int,
    "model": {
        "mlp": state_dict | None,
        "cnn": state_dict | None,
        "prompt": state_dict | None,
    },
    "attack_meta": {
        "method": str,
        "trigger": optional_serialized_trigger,
        "metrics": dict,
    },
    "privacy_meta": {
        "mechanism": str,
        "noise_std": float,
        "noise_variance": float,
        # 默认不得包含 raw_norm 或本地样本
    },
    "observer_payload": {
        # 仅 privacy 模式允许；backdoor 模式必须为空
        # DP 开启时必须是服务端实际可见版本
        "gradients": optional,
        "reference": optional,
    },
}
```

为保留旧测试和旧配置，服务端 parser 在一个版本周期内兼容旧的 2/3/4 元组；所有后续代码只消费 parser 产生的规范对象。

### 4.3 配置轴与实验模式必须解耦

建议新增顶层 `security.mode: none | backdoor | privacy`，并将现有配置含义固定为：

- `security.mode: backdoor` 时，`attack.attack_method` 表示 `a3fl`、`cerberus`、`sabre`、`label_flip` 或 `lie`；隐私插件必须禁用。
- `security.mode: privacy` 时，使用 `attack.modular_attacks` + `attack.attack_plugins` 表示 `fedmia_i`、`fedmia_ii`、`meta_ppa` 或 GRNN；后门方法必须为空。
- `dp.*`：客户端上传隐私防御。
- `ggeur.defense_method: multi_metrics`：仅 backdoor 模式使用的训练阶段后门防御。
- `ggeur.multi_metrics_stats_defense: true`：仅 backdoor 模式使用的统计阶段后门/数据投毒防御。

不要在一次 run 中同时表达 SABRE 和 FedMIA。旧配置中的 `attack_method: ggeur_fedmia` 可以通过兼容层映射为 `security.mode: privacy`；旧的 A3FL/SABRE 等配置映射为 `security.mode: backdoor`。检测到两类攻击同时启用时应立即报错。

建议新增配置校验：

1. `federate.mode` 必须为 `standalone`；其他模式在新版安全组合入口直接报错。
2. backdoor method 与 privacy attack plugin 不得同时启用。
3. backdoor 模式必须关闭 FedMIA/PPA/GRNN hook 和所有 distributed collector。
4. privacy 模式必须关闭 A3FL/CERBERUS/SABRE/LabelFlip/LIE、MultiMetric-Stats、MultiMetric-Model 和 ASR 评估。
5. 同一轮只能选择一个服务端 robust aggregator；`multi_metrics` 与 FLAME/MARS 等不能同时隐式生效。
6. `dp.enabled=True` 时必须有 `protect_ggeur_update=True`，否则明确报错或警告，而不是静默不保护。
7. `upload_private_stats=False` 时 envelope 中禁止出现 raw norm、原始梯度、原始图片或 feature vector。
8. 阶段 0 的统计保护范围需显式配置。当前若不对 means/covs/prototypes 加噪，文档必须写清“模型更新 DP，不覆盖特征统计交换”。

## 5. 逐文件合并方法

### 5.1 `ggeur_client.py`

以 `attack` 版本为骨架，必须保留：

- augmented cache fingerprint、load/save、统计恢复与 buffer 释放。
- OfficeHome/local-only 数据处理。
- `recursive_param2tensor` 等已有 helper 以保证原分支单元测试，但新版 standalone 主路径不依赖网络序列化。
- FedMIA/PPA/GRNN 只接入独立的 privacy-mode callback，不接入 backdoor-mode callback。
- local adaptive clipper、fixed/DPFL/adaptive 三类上传保护。
- standalone 客户端评估和 DP summary。

从 `datapoison` 移植：

- A3FL/CERBERUS/SABRE/LabelFlip/LIE 的初始化状态。
- 阶段 0 的 label-flip statistics 钩子。
- 三类触发器构造、搜索、共享、特征毒化与本地 ASR 评估 helper。
- 后门专用训练分支与 attack metadata。
- 非 GGEUR A3FL trainer 相关调用仍留在基础 Client 路径，不与 GGEUR callback 重复执行。

backdoor 模式训练阶段的强制顺序：

```text
加载 global state
  -> 保存 global snapshot
  -> 解析 server 下发的 trigger/peer state
  -> 执行本地后门/数据投毒训练
  -> 执行 update reversal 等最终攻击变换
  -> 计算最终 delta
  -> DP clip + noise
  -> 构造并发送 envelope
```

特别注意：`datapoison` 当前是先 `_apply_adaptive_dp_to_upload()` 再 `_apply_update_reversal_to_upload()`，合并时必须反转这一顺序。

privacy 模式走另一条独立路径：正常训练 -> 可选 DP -> privacy attack hook。该路径不得初始化后门 trigger、攻击标签或 ASR evaluator。

### 5.2 `ggeur_server.py`

以 `attack` 版本的 standalone 阶段控制为骨架，保留：

- 客户端评估和 DP summary。
- FedMIA/PPA/GRNN standalone hook，但仅在 privacy 模式初始化。
- 真实 distributed collector 保留为旧代码兼容项，不接入新版组合入口。

从 `datapoison` 移植：

- A3FL/CERBERUS/SABRE/LabelFlip/LIE 服务端状态。
- shared trigger 和 peer model 的广播/更新。
- clean accuracy 与 ASR 的评估频率控制。
- `_multi_metrics_filter_statistics()`，接在统计 payload 验证之后、全局 covariance/prototype 聚合之前。
- `_multi_metrics_aggregate_model_params()`，接在模型 payload 规范化之后、FedAvg 之前。
- 必要的无防御 FedAvg helper 与统一更新向量抽取。

backdoor 模式下服务端的模型更新顺序：

```text
解析 envelope
  -> 只保留有效的模型 state_dict
  -> 可选 LIE 服务端攻击变换（实验模式）
  -> MultiMetric 评分与筛选
  -> 对筛选后的更新执行 FedAvg/FedOpt
  -> 更新 shared trigger/peer model
  -> clean/ASR/client eval
```

MultiMetric 不能读取 `attack_meta.client_is_attacker` 或配置中的 attacker id 来做实际筛选；这些信息只能用于 oracle baseline 或离线评估，否则会造成结果泄漏。

privacy 模式的服务端只执行普通聚合和 privacy attack hook，不初始化共享 trigger、MultiMetric 或后门评估器。

### 5.3 `adaptive_dp.py` 与配置

建议保留 `attack` 的：

- `subtract_states()`、`add_delta()`、`sanitize_update()`。
- `LocalAdaptiveClipper`。
- `is_ggeur_client_update_dp_enabled()` 和 `get_ggeur_client_update_dp_cfg()`。

可以从 `datapoison` 保留独立、无状态的纯函数测试思路，例如 `clip_update()`、`update_clip_bound()` 和 variance summary，但不要恢复 `_GLOBAL_CONTROLLERS` 作为 GGEUR 主路径。

配置统一使用 `cfg.dp`。若必须兼容已有 `adaptive_dp:` YAML，可新增一次性迁移函数：

```text
adaptive_dp.use                  -> dp.enabled + dp.protect_ggeur_update
adaptive_dp.initial_clip         -> dp.clipping.initial_clip
adaptive_dp.target_quantile      -> dp.clipping.target_quantile
adaptive_dp.ema                  -> dp.clipping.ema
adaptive_dp.min_clip/max_clip    -> dp.clipping.min_clip/max_clip
adaptive_dp.noise_multiplier     -> dp.noise_multiplier
```

迁移时打印 deprecation warning；不要让两个 namespace 同时生效。

隐私声明注意事项：

- 当前机制可以称为“客户端更新级自适应裁剪与高斯扰动”或“经验型本地 DP 防御”。
- 在自适应阈值本身未私有化、端到端 accountant 未验证、阶段 0 统计未保护前，不应声称整个 GGEUR 流程满足严格 `(epsilon, delta)-DP`。
- 固定 `(round, client_id, seed)` 会产生可复现噪声，适合测试，不适合重复发布同一私有数据的生产场景；生产模式需使用安全随机源并持久化隐私会计状态。

### 5.4 builder 与普通 FederatedScope 路径

`worker_builder.get_server_cls()` 的优先级应为：

1. `federate.method == ggeur and attack_method == grnn`：`GGEURPassiveServer`。
2. GGEUR registry：返回合并后的 `GGEURServer/GGEURClient`，再根据 `security.mode` 只初始化后门路径或隐私路径之一。
3. 普通 `dlg/ig/grnn`：`PassiveServer`。
4. 普通 `backdoor/a3fl`：`BackdoorServer`。

这样 GGEUR+A3FL 不会错误切换到通用 `BackdoorServer`，同时旧的非 GGEUR A3FL 测试仍能工作。

`server_attacker.py` 中 A3FL receiver sampling 与 GRNN reconstruction 基本位于不同类中，可以同时保留代码，但一次 standalone run 只能实例化其中一种攻击路径。必须为以下边界加测试：

- `sample_client_num < active_attackers`。
- benign pool 不足。
- `attacker_id` 为单值、列表、字符串时的解析一致性。
- GRNN/FedMIA 扩展 tuple 不影响普通 backdoor 的二元上传。

## 6. 推荐的实际 Git 操作

```bash
git fetch origin
git switch -c feature/ggeur-unified-security-standalone origin/attack

# 先观察提交与差异
git log --oneline --left-right origin/attack...origin/datapoison
git diff --stat origin/attack...origin/datapoison

# 保留一次真实 merge，便于追溯两个分支历史
git merge --no-ff --no-commit origin/datapoison

# 查看冲突和三个 stage
git status --short
git diff --cc
git show :1:federatedscope/contrib/worker/ggeur_client.py > /tmp/ggeur_client.base.py
git show :2:federatedscope/contrib/worker/ggeur_client.py > /tmp/ggeur_client.attack.py
git show :3:federatedscope/contrib/worker/ggeur_client.py > /tmp/ggeur_client.datapoison.py
```

不要运行：

```bash
git checkout --ours federatedscope/contrib/worker/ggeur_client.py
git checkout --theirs federatedscope/contrib/worker/ggeur_server.py
```

这会完整丢掉另一分支的核心能力。

推荐在同一个未提交 merge 中按以下顺序解决：

1. `cfg_attack.py`、`cfg_ggeur.py`、`cfg_differential_privacy.py` 和 legacy 配置迁移。
2. `adaptive_dp.py` 与 `privacy/__init__.py`。
3. 统一 upload envelope/parser。
4. `ggeur_client.py`。
5. `ggeur_server.py`。
6. worker/trainer builder、普通 Client、`server_attacker.py`。
7. 示例 YAML 和测试。

完成后：

```bash
git diff --check
/root/.local/share/mamba/envs/pfedba/bin/python -m compileall -q federatedscope
/root/.local/share/mamba/envs/pfedba/bin/python -m pytest -q \
  tests/test_ggeur_unified_security.py \
  tests/test_distributed_fedmia.py tests/test_distributed_ppa.py
/root/.local/share/mamba/envs/pfedba/bin/python -m pytest -q \
  tests/test_backdoor_attack.py tests/test_rec_opt_attack.py \
  tests/test_rec_IG_opt_attack.py tests/test_CRA_gan_attack.py
git add federatedscope scripts tests docs
git commit -m "Merge GGEUR heterogeneity, privacy, and backdoor pipelines"
```

如果希望减小一次提交的审查压力，可以在 merge commit 完成后再用后续提交分别重构 message protocol、补测试和补配置；不建议通过 cherry-pick 两个大文件的多个历史提交来代替真实 merge，因为 `datapoison` 的 7 个提交连续重写同一区域，会重复解决相同冲突。

## 7. 新版集成配置示例

### 7.1 后门实验

后门实验只运行 SABRE、可选上传 DP 和两阶段 MultiMetric；明确关闭所有隐私攻击：

```yaml
security:
  mode: backdoor

federate:
  method: ggeur
  mode: standalone
  client_num: 10
  sample_client_num: 10

ggeur:
  use: true
  statistics_round: 0
  defense_method: multi_metrics
  multi_metrics_stats_defense: true
  multi_metrics_stats_min_clients: 4
  multi_metrics_stats_keep_ratio: 0.75
  multi_metrics_min_clients: 4
  multi_metrics_keep_ratio: 0.5

attack:
  attack_method: sabre
  attacker_id: [1, 2]
  modular_attacks: false
  attack_plugins: []
  distributed_fedmia: false
  distributed_ppa: false
  sabre:
    start_round: 1
    poison_epochs: 10

dp:
  enabled: true
  level: client_update
  mechanism: gaussian
  protect_ggeur_update: true
  upload_private_stats: false
  noise_multiplier: 1.0
  clipping:
    type: adaptive
    initial_clip: 0.5
    target_quantile: 0.7
    ema: 0.9
    min_clip: 0.02
    max_clip: 5.0
```

注意：DP 噪声会改变 MultiMetric 的三类距离分布。合并后必须在“无攻击+有 DP”场景重新标定 keep ratio/threshold，不能直接沿用无噪声阈值，否则会把正常客户端噪声误判为后门更新。

### 7.2 隐私实验

隐私实验单独运行 FedMIA，并关闭所有后门攻击和 MultiMetric：

```yaml
security:
  mode: privacy

federate:
  method: ggeur
  mode: standalone
  client_num: 10
  sample_client_num: 10

ggeur:
  use: true
  statistics_round: 0
  defense_method: ''
  multi_metrics_stats_defense: false

attack:
  attack_method: ''
  attacker_id: -1
  modular_attacks: true
  attack_plugins: [fedmia_i, fedmia_ii]
  distributed_fedmia: false
  distributed_ppa: false

dp:
  enabled: true
  level: client_update
  mechanism: gaussian
  protect_ggeur_update: true
  upload_private_stats: false
  noise_multiplier: 1.0
  clipping:
    type: adaptive
    initial_clip: 0.5
    target_quantile: 0.7
    ema: 0.9
    min_clip: 0.02
    max_clip: 5.0
```

隐私实验建议分别执行 `dp.enabled: false` 和 `true` 两次，比较攻击 AUC/TPR；不要在同一次运行中加入 SABRE/A3FL 等后门攻击。

## 8. 必须新增的测试矩阵

### 8.1 快速单元测试

1. 配置：两个分支已有 YAML 都能 load；legacy `adaptive_dp` 能迁移或给出清晰错误；`federate.mode != standalone` 被新版组合入口拒绝。
2. DP 数学：delta 计算、global+delta 重建、固定/自适应 clip、noise variance、非浮点 buffer 保持不变。
3. DP 隔离：两个 client 的 clip history 不共享；默认上传 metadata 不含 raw norm。
4. 消息协议：2/3/4 元组和 v1 envelope 都能解析成同一规范结构。
5. MultiMetric-Stats：构造 4+ 客户端和一个统计 outlier，验证过滤和“不可全删”fallback。
6. MultiMetric-Model：构造正常更新与一个方向/幅度 outlier，验证筛选后聚合结果。
7. A3FL/CERBERUS/SABRE：触发器 shape、裁剪范围、攻击轮判断、攻击 metadata 与 state_dict 分离。
8. 模式互斥：SABRE+FedMIA、A3FL+GRNN 等组合必须在启动前报错。
9. builder：privacy 模式 GGEUR+GRNN、backdoor 模式 GGEUR+A3FL、普通 A3FL、普通 DLG 的 class dispatch。

### 8.2 standalone 短流程测试

至少使用 4 个客户端、2~3 个训练 round，覆盖：

后门模式：

| 后门攻击 | DP | MultiMetric-Stats | MultiMetric-Model | 隐私攻击 | 期望 |
| --- | --- | --- | --- | --- | --- |
| 关 | 关 | 关 | 关 | 关 | GGEUR standalone 基线流程一致 |
| SABRE | 关 | 开 | 开 | 关 | 两阶段防御执行且输出 ACC/ASR |
| SABRE | 开 | 开 | 开 | 关 | DP 是最后上传变换，不产生隐私攻击指标 |
| LabelFlip | 关 | 开 | 关 | 关 | 阶段 0 异常统计可被过滤 |
| A3FL | 开 | 关 | 开 | 关 | message parser 不混淆攻击 metadata 与模型参数 |

隐私模式：

| 隐私攻击 | DP | 后门攻击 | MultiMetric | 期望 |
| --- | --- | --- | --- | --- |
| FedMIA | 关 | 关 | 关 | 输出无防御 AUC/TPR 基线 |
| FedMIA | 开 | 关 | 关 | 只观察加噪后的 update，输出防御后 AUC/TPR |
| PPA | 开/关 | 关 | 关 | 不生成 ASR 或 trigger 状态 |
| GRNN | 开/关 | 关 | 关 | 不初始化 A3FL/CERBERUS/SABRE |

每个组合至少断言：

- 阶段状态按 `statistics -> augmentation -> training -> finish` 前进。
- 同一 client/round 只上传一次模型更新。
- server 聚合输入是 DP 后、MultiMetric 筛选前的规范 state_dict。
- backdoor 模式只输出 clean ACC/ASR，不输出 privacy attack metric。
- privacy 模式只输出 clean ACC/privacy metric，不输出 ASR。
- 无 NaN/Inf、无空聚合、无等待已被过滤客户端造成的死锁。

### 8.3 两个分支的回归测试

必须保留并运行 `attack` 新增的两个 collector 单元测试，即使新版主流程不使用真实分布式客户端：

```bash
pytest -q tests/test_distributed_fedmia.py tests/test_distributed_ppa.py
```

以及仓库现有攻击回归：

```bash
pytest -q tests/test_backdoor_attack.py \
  tests/test_rec_opt_attack.py \
  tests/test_rec_IG_opt_attack.py \
  tests/test_CRA_gan_attack.py
```

`datapoison` 当前没有对应的新增测试文件，所以需要将其关键行为固化成上面的新测试，不能把“原分支能跑实验脚本”当成自动化回归保证。

## 9. 验收门槛

合并分支只有同时满足以下条件才算完成：

- 两个原分支的已有测试全部通过。
- 新增的 standalone GGEUR、DP、MultiMetric、后门模式和隐私模式测试通过。
- `attack` 分支的 cache/manifest 能力仍可在 standalone 中使用；distributed/quorum/fault-injection 代码不作为新版验收项。
- 旧 2/3/4 元组上传在兼容期内可解析；新版内部只使用规范 envelope。
- DP 是客户端发送前最后一个模型更新变换；privacy 模式的攻击 hook 只能看到该 defended update。
- backdoor 模式不会初始化或执行 FedMIA/PPA/GRNN，privacy 模式不会初始化或执行后门攻击、MultiMetric 或 ASR 评估。
- MultiMetric-Stats 在全局统计聚合前运行，MultiMetric-Model 在 FedAvg/FedOpt 前运行。
- 防御逻辑不使用真实 attacker id；oracle filtering 只能由单独配置显式启用。
- 文档明确阶段 0 特征统计是否受隐私保护，并准确描述当前 DP 保证级别。
- 单机至少 4 个逻辑客户端的 backdoor 与 privacy 短流程分别能够结束，且不会交叉产生另一模式的指标或状态。

## 10. 本次审阅的验证状态

- 已创建并实现分支 `feature/ggeur-unified-security-standalone`，以
  `origin/attack` 为起点保留真实 merge 关系，再合入
  `origin/datapoison`。
- 已用 `/root/.local/share/mamba/envs/pfedba/bin/python` 完成核心代码
  `compileall`，并通过 `git diff --check`。
- 新增的统一安全测试、FedMIA collector 测试和 PPA collector 测试共
  15 项通过；覆盖模式互斥、standalone 限制、逐客户端自适应裁剪、
  defended upload 可见性、GRNN 图像分支、MultiMetric 两阶段和 builder
  分派。
- `scripts/attack_exp_scripts` 下 12 份现有 GGEUR YAML 均已验证能够加载并
  freeze；legacy `adaptive_dp`、`fed_smp.use: false`、mixup 和图像增强配置
  保留兼容入口。
- FEMNIST 的 4 个原有端到端攻击测试已通过测试收集；本机没有其测试数据，
  因此没有把未实际执行的长流程标记为通过。

# 隐私模块合入记录

更新：FedMIA 外置资源已提供并部署，60 个客户端真实回放检查通过；必要兼容补丁、共同样本限制及当前打包要求见 [隐私回放部署](PRIVACY_REPLAY.md)。下文保留首次合入时的历史状态。

2026-09-16：将当前 portable 工作树的修改合入最新远端代码，使用独立工作树，不改原工作树、正式数据或运行中的服务。合并验证阶段没有提交或推送；随后按用户要求，将合并结果提交并发布至现有 prototype/frontend、prototype/backend，不创建新的远端分支。没有构建镜像或导出交付包。

| 新工作分支 | 工作树（主仓库下） | 纳入的远端基线 |
|---|---|---|
| prototype/integrated-backend | output/integrated/backend | prototype/backend @ 970fa95d9ea9301231d8c4e31843245632454410 |
| prototype/integrated-frontend | output/integrated/frontend | prototype/frontend @ 87b6a96f6c7b79285767726bdb83edde3647278f |

## 合并结果

- 保留最新隐私成员推理页面、攻防指标/分布对照及 FedMIA 接口；后门入口仍预留。
- 保留本地军机三方法默认配置：100 轮；本架构本地 1 轮、40/20/20、自动匹配并固定缓存来源。页面以 API 默认参数为准，已编辑草稿不被覆盖。
- 统一军机资源为 backend/resources，状态为 backend/exp/platform，前端为同级 frontend/dist；路径解析复用远端 paths.py，持久化配置仍使用相对路径。保留缓存迁移、特征分类头验证、离线 worker 保护和部署检查。
- 隐私资源缺失返回 configured=false；页面按需显示资源目录，不生成替代结果，不影响准确率入口。增加查询参数校验、攻防客户端配对、无效分数/数量不匹配拒绝、并发缓存锁；本层冻结配置不回写原始 config.yaml。

## 验证结果

- 前端：独立 npm ci、72 项测试、TypeScript 与生产构建通过（仍有大 chunk 提示）。
- 后端：Linux 39 项 single-host 测试通过，含 6 项新增隐私 HTTP 合约测试；部署预检 6 项与启动器 8 项通过。
- 合并版本在 4090 的临时目录完成三方法各 2 轮真实缓存训练；225 样本独立评测与训练结果一致，选图预测、CSV/JSON/模型下载、重启保留模型库、任务资源清理通过。本架构实际选择 reuse，没有重新生成增强数据。
- 部署预检检查 53 个精确依赖版本、GPU 张量计算、12 个前端运行文件、750 张图片及默认增强包通过。首次运行缺少临时状态目录被正确阻断，创建目录后通过。
- 报告保存在主仓库 output/integrated/integrated-preflight.json 和 integrated-offline-smoke.json。没有重验准确率提升门槛。

## 隐私资源与打包待办

用户确认外置资源位置暂时未知，留到打包时补充；这不是代码合并阻塞。源码默认继续兼容同级 ../fedmia_local，可用相对 backend 的 FS_FEDMIA_LOCAL_ROOT 覆盖。Docker 模板已设 resources/fedmia_local，复用现有只读 resources 挂载。

资源目录必须包含 show_fedmia_examples.py、runs/{no_defense,defense}/config.yaml、各自 ggeur_fedmia_features/、datasets/OfficeHomeDataset_10072016/。需提供可信来源文件：上游使用动态脚本和 unsafe_load=True，不能加载未知来源包。

当前隐私测试使用明确的测试桩，只验证接口、指标计算与页面交互，不代表真实 FedMIA 推断通过。拿到外置脚本后仍需核验其依赖、读写路径、样本与攻防特征对应关系、真实结果及断网运行；现有依赖锁和部署预检仅证明军机链路，不证明隐私资源完整。最后再做干净 Docker 构建、容器生命周期/断网验收及 RTX 5880 现场验证。

# 后门更新合并与部署边界

- 来源：`backend_backdoor` 的 `4614991`，对应前端 `fronted_backdoor` 的 `276da12`。
- 保留便携路径、停止/恢复、安全校验、正确 ASR 分母及离线 DNS 修复。
- 新军机后门资源放入 `resources/backdoor/exp/sabre_newdataset/`，包含实验子目录、分类头、触发器、配置与 `testset_images/index.csv`、样本图片。
- 军机原始图片复用 `resources/datasets/MilitaryAircraft3D/`；支持 `FS_PLATFORM_DATASETS` 和 `FS_BACKDOOR_DATA_ROOT` 显式覆盖。不存在新实验目录时继续使用已打包的 `resources/backdoor/exp/sabre/`。
- 新增 `scripts/backdoor/precompute_trigger_predictions.py` 可在资源准备阶段显式生成展示抽样缓存。页面选样不自动启动预计算，避免无人管理的 GPU 进程及 Docker 只读资源写入失败；缓存缺失则随机抽样。
- 有缓存时保留远端的加权展示选样，并在前端注明不代表整体指标；此样本集合不得用作无偏总体准确率或 ASR 结论。
- Git 合并不包含仓库外的新军机后门权重、触发器和预测缓存。部署前必须核对这些资源，不能以已有训练模型代替后门实验权重。
- 本次仅合并及代码回归，不更新运行中的服务器、Docker 镜像或 U 盘；旧离线包仍是旧版本，后续需要重新构建、打包并验收。

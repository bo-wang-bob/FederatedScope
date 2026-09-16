# 项目交付目录

此目录是代码与资源的完整副本，不是已构建的 Docker 镜像。现有服务器目录和运行服务保持不变。

```text
FederatedScope-delivery/
├── README.md
├── DIRECTORY.json                 # 来源标签及隐私脚本适配记录
├── FILES.sha256.json              # 准备时的逐文件大小和 SHA256
├── frontend/
│   ├── src/、resource/、package*.json、构建配置
│   └── dist/                      # 生产页面，含 .vite/manifest.json
└── backend/
    ├── federatedscope/、scripts/、deploy/、docs/、tests/
    ├── resources/                 # Docker 中只读挂载
    │   ├── datasets/MilitaryAircraft3D/
    │   ├── datasets/OfficeHomeDataset_10072016/  # OfficeHome 训练/验证原图
    │   ├── exp/distributed_feature_cache/military_aircraft_vit_fixedsplit_v2/
    │   ├── exp/distributed_feature_cache/officehome_vit/  # 四域训练及测试缓存
    │   ├── caches/military_vit_default/   # 独立 40/20/20 增强缓存
    │   ├── models/ViT-B-16.pt
    │   ├── backdoor/exp/               # 后门模型、触发器、测试图和缓存
    │   └── fedmia_local/
    │       ├── show_fedmia_examples.py
    │       ├── runs/{no_defense,defense}/
    │       │   ├── config.yaml
    │       │   └── ggeur_fedmia_features/
    │       └── datasets/OfficeHomeDataset_10072016/
    └── exp/platform/jobs/          # 配置、分类头、测试特征、结果和日志
```

## 路径规则

- 整体复制，不单独拿出 frontend 或 backend；不使用指向原服务器的软链接。
- 源码资源默认 backend/resources；隐私默认 resources/fedmia_local；状态默认 backend/exp/platform；页面默认同级 frontend/dist。相对环境变量和 `--state-dir` 按 backend 根解析，与终端当前目录无关。
- Docker 挂载按启动脚本所在交付目录解析。换机器前不要携带旧的 FS_PLATFORM_*、FS_FEDMIA_LOCAL_ROOT、FEDERATEDSCOPE_FRONTEND_DIST 绝对路径环境变量。
- 历史配置、日志、source.zip 保留原始证据，可能含旧绝对路径，但不用于寻找当前数据或模型。新任务重新解析资源，历史模型按任务 ID 查找。不要批量替换历史 JSON/模型，避免破坏哈希。
- 副本排除 process.json、service.lock、字体缓存、Git、node_modules、Python 缓存。隐私脚本只适配已审查版本的导入/实验/数据路径和只读加载；原 config.yaml 不改写。
- Docker 镜像和离线宿主机安装包需要单独准备，不在本目录中。

## 准备副本

先构建配套前端 dist。以下路径参数相对当前终端目录；输出必须是全新的、不嵌套于来源的目录：

```bash
python backend/scripts/prepare_platform_directory.py \
  --output /目标位置/FederatedScope-delivery \
  --frontend /来源/frontend \
  --dataset /来源/MilitaryAircraft3D \
  --features /来源/military_aircraft_vit_fixedsplit_v2 \
  --officehome-dataset /来源/OfficeHomeDataset_10072016 \
  --officehome-features /来源/officehome_vit \
  --backdoor /来源/backdoor \
  --weights /来源/ViT-B-16.pt \
  --privacy /来源/fedmia_local \
  --state /来源/exp/platform \
  --cache-job d74589340f344f6cbba7f18e7df97e01
```

只复制，不移动或覆盖来源。运行中/未清理的任务、软链接、复制时文件变化、未知隐私脚本均阻断。失败目录带 PREPARATION_INCOMPLETE，不能部署；修复原因后另选新目录。

OfficeHome 两项参数需要同时提供；不提供时仍可准备原军机交付范围。训练图片与隐私回放图片分别保留，不默认视为同一版本。OfficeHome / ViT 开放 FedAvg、FedProx、本架构，默认 GPU 0、60 客户端、3 通信轮、本地 1 轮；本架构沿用该数据集的 50/50/50 增强配置，按配置生成增强特征，不重新提取图片特征。军机 40/20/20 默认配置不变。

OfficeHome 完整执行短测（隔离状态目录，三方法各两轮，不验收准确率门槛）：

```bash
python backend/scripts/platform_offline_smoke.py --group officehome_vit --report /外部位置/officehome-smoke.json
```

## 复制或换目录后检查

后门默认读取 `backend/resources/backdoor/exp/sabre`，复用隐私的 `resources/fedmia_local/datasets/OfficeHomeDataset_10072016`，ViT 权重读取 `resources/models/ViT-B-16.pt`。可用 `FS_BACKDOOR_BASE`、`FS_BACKDOOR_DATA_ROOT`、`FS_BACKDOOR_VIT_WEIGHTS` 覆盖，相对值按 backend 根解析。不修改归档配置中的原始路径证据；缺少本地权重时明确报错，不联网下载。后门运行记录保存在 `exp/platform/backdoor`，打包时同样禁止运行中任务。

后门页面执行已保存模型的真实原图推理，不重新训练；沿用来源分支的各实验模型及各自触发器。选中样本的对照不代表整个测试集的攻防指标。ASR 分母排除目标类别样本。

文件校验只需 Python；完整检查需配套依赖，仅在临时目录写评测/推理结果，不训练、不重新生成特征：

```bash
python /新位置/FederatedScope-delivery/backend/scripts/check_platform_directory.py --files-only
python /新位置/FederatedScope-delivery/backend/scripts/check_platform_directory.py --report /外部位置/directory-check.json
python /新位置/FederatedScope-delivery/backend/scripts/check_platform_deployment.py --gpu --require-cache
```

完整检查包括全部已存分类头的实际评测及选图推理、测试图片、独立增强包和隐私客户端回放；第三条检查三方法默认配置与 750 张军机原图。不重验准确率提升门槛。隐私回放来源限制见 backend/docs/PRIVACY_REPLAY.md。

FILES.sha256.json 是整理时的快照校验。服务运行后任务正常变化，不能误认为复制损坏；再次交付需新建快照。报告放在交付目录外。

已有 Python 环境时从任意目录运行 `python /新位置/FederatedScope-delivery/backend/scripts/start_platform.py --host 127.0.0.1 --port 8002`。

Docker 构建上下文是含 backend/、frontend/ 的根目录。依赖、代码、页面进镜像；resources 和 exp/platform 外部挂载。构建/导出镜像、完全断网容器及 RTX 5880 现场验收仍需单独完成，操作模板见 backend/deploy/README.md。目录检查不等于最终离线部署验收。

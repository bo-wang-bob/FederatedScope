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
    │   ├── exp/distributed_feature_cache/military_aircraft_vit_fixedsplit_v2/
    │   ├── caches/military_vit_default/   # 独立 40/20/20 增强缓存
    │   ├── models/ViT-B-16.pt
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
  --weights /来源/ViT-B-16.pt \
  --privacy /来源/fedmia_local \
  --state /来源/exp/platform \
  --cache-job d74589340f344f6cbba7f18e7df97e01
```

只复制，不移动或覆盖来源。运行中/未清理的任务、软链接、复制时文件变化、未知隐私脚本均阻断。失败目录带 PREPARATION_INCOMPLETE，不能部署；修复原因后另选新目录。

## 复制或换目录后检查

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

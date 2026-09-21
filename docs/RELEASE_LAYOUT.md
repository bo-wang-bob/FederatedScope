# 源码与资源交付目录

当前推荐的打包入口。无需 Docker；不附带 Python、Node、虚拟环境或驱动。
前后端保持同级，不搬动正在运行的安装目录。

本次交付已移除 OfficeHome 原图、特征缓存、旧隐私/后门实验资源与结果；页面不再提供该数据集。军机、上传数据集、共享模型权重及军机隐私/后门资源保留。内部通用配置模板供上传训练复用，不包含 OfficeHome 数据。

```text
release/
├─ frontend/
│  ├─ src/、resource/、tests/、package*.json、构建配置
│  └─ dist/                         # 已构建页面，正式运行不需要 Node 服务
├─ backend/
│  ├─ federatedscope/、scripts/、deploy/、docs/、tests/
│  ├─ setup.py、run.py
│  ├─ resources/
│  │  ├─ datasets/MilitaryAircraft3D/
│  │  ├─ exp/distributed_feature_cache/  # 军机 ViT 特征
│  │  ├─ caches/military_vit_default/    # 默认增强缓存
│  │  ├─ models/ViT-B-16.pt
│  │  ├─ torch/hub/checkpoints/convnext_base-6075fbad.pth
│  │  ├─ fedmia_local/                  # 当前完整隐私资源包
│  │  ├─ backdoor/exp/                  # 当前完整后门资源包
│  │  └─ uploaded_datasets/             # 有上传时保留，包含原图及转换图
│  └─ exp/platform/                     # 训练、模型、评测、隐私、后门实验记录
├─ README.md
└─ FILES.sha256.json                    # 文件校验清单，不是运行依赖
```

## 整理与导出

在前后端的父目录执行。先在 frontend 中运行 `npm ci`、`npm run build`，然后回到父目录。

```bash
python backend/scripts/prepare_current_release.py
python backend/scripts/prepare_current_release.py --output ../release-ready
python ../release-ready/backend/scripts/check_platform_directory.py --files-only
```

第一条只检查目录与关键资源是否存在，不复制文件，也不代替训练预检。
导出前停止本平台服务和任务，不进行上传；目标必须是全新目录且不能嵌套在前后端内。
脚本不改写资源，完整复制当前资源与各模块实验记录，逐文件校验；活动任务或复制期间变化会使导出失败。
失败目录保留 `PREPARATION_INCOMPLETE`，不要使用它部署，请排查后换新目录重新导出。
导出成功后压缩整个 `release-ready` 即可，不要只压缩两个分支的 Git 源码。

排除 `.git`、`node_modules`、虚拟环境、`.env*`、Python/测试缓存，以及进程身份文件和服务锁。
外层本机 `Environment.ps1`、Start/Stop 脚本、服务日志等不会混入交付目录。
日志与历史实验配置保留原始内容用于追溯；其中的旧路径不等于当前部署路径，不要手工全局替换。
资源不进 Git；更新代码时请继续保留 resources 与 exp/platform，勿用空目录覆盖。

## 运行

在目标机激活已准备的 Python 环境；所需包见 `backend/deploy/requirements-runtime.lock`。
本说明不安装驱动、不修改 CUDA、不强制 CPU；默认 GPU 行为保持不变。
若此前设置过 `FS_PLATFORM_*`、`FS_BACKDOOR_*` 或 `CUDA_VISIBLE_DEVICES` 等本机覆盖，请先核对并清除不适用于目标机的值。

在 release 目录执行（Linux、Windows 相同）：

```bash
python backend/scripts/start_platform.py --host 127.0.0.1 --port 8002
```

浏览器打开 `http://127.0.0.1:8002`。局域网访问时改用 `--host 0.0.0.0`，仅向可信网络开放。
默认从兄弟目录 frontend/dist 提供页面，后端路径相对 backend 解析；不要同时启动两个实例共享一个状态目录。
resources/uploaded_datasets 和 exp/platform 必须可写，其余资源不要随意修改。
停止服务用启动终端的 Ctrl+C；先停止实验再退出。迁移后先运行一次预检，再验收训练、模型评测及隐私/后门页面。

已有 ViT 模型保持“保存特征＋分类头”验证；上传原图训练与测试使用上传训练链路。
无标签测试仅输出预测，不计算准确率。图片格式及标注要求见 backend/docs/uploaded-dataset-formats.md。
GPU 实际运行与算法准确率需在目标 GPU 机器验收，目录校验通过不代表算法验收通过。

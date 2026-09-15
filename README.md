# 跨域联邦学习 · 后端原型

分支：`prototype/backend`。对应前端：[prototype/frontend](https://github.com/bo-wang-bob/FederatedScope/tree/prototype/frontend)。

从已部署版本 `5ac3e453d69f33596c193e15daa808aaca52865b` 拆分。保留 `federatedscope/`、训练配置、脚本、后端测试及部署参考；不包含前端工程。算法、API 和配置文件与拆分源一致，不修改运行中的服务器。

## 启动现有环境

优先复用 4090lziy 已验证的 GGEUR Python/PyTorch 环境；先激活环境，再在本分支根目录运行：

```bash
python -m pip install --no-deps -e .
export FS_PLATFORM_RESOURCES=/root/autodl-tmp/FederatedScope
export FS_PLATFORM_DATASETS=/root/autodl-tmp/datasets
python -m federatedscope.standalone_api.platform_app \
  --host 127.0.0.1 --port 8001 --state-dir /path/to/prototype-state
```

以上资源路径是现有服务器示例，需要按实际环境设置；状态目录应独立且可写，不要同时让两个服务管理同一个状态目录。8001 已有服务时另选未占用端口。拆分与推送本身不启动、停止或覆盖任何已有服务。

`setup.py` 保留原依赖声明，但包含历史版本约束；`--no-deps` 仅适用于已具备依赖的验证环境，不是全新环境安装承诺。不要为安装该原型而降级正在使用的训练环境。

检查：`curl http://127.0.0.1:8001/api/health`。无前端文件时 API 仍可独立运行，根页面不会提供界面。当前 API 没有用户认证，只绑定本机并使用 SSH 隧道或受控网络，禁止直接暴露公网。

## 对接独立前端

开发：前端使用 `FS_API_PROXY` 指向该服务。生产：先在 `prototype/frontend` 执行 `npm ci && npm run build`，在启动后端前设置：

```bash
export FEDERATEDSCOPE_FRONTEND_DIST=/absolute/path/to/prototype/frontend/dist
```

后端即可从该目录提供页面。只指定已经审核的构建目录；不需要把两个分支相互合并。

## 功能与数据

- 单机模拟多客户端，保留联邦聚合、异构划分和算法比较。
- `/api/platform/catalog`、`library`、`jobs`：读取能力、模型/测试集和任务。
- `/api/platform/preflight`、`train`、`predict`、`evaluate`：预检、训练、单图预测和独立评测。
- 任务支持停止、恢复检查、仅清理自身进程及 JSON/CSV/模型/复现包导出。
- 基础特征必须完整；本架构可按配置新生成增强特征。单图验证使用关联冻结特征与分类器，不是重新提取原图特征；历史来源限制不能省略。

数据集、主干权重、训练模型、特征缓存和任务日志不随 Git 分支上传。资源路径及缓存覆盖变量见 [单机平台说明](docs/SINGLE_HOST_PLATFORM.md)。旧入口和历史方案说明保留在 [历史 README](docs/LEGACY_STANDALONE_README.md)，不是本原型的默认启动方式。

## 验证

```bash
python -m unittest discover -s tests -p 'test_single_host*.py' -v
```

拆分源已完成真实预测、独立评测和资源清理验收。本次核对后端源码与配置未变，不运行新的训练，也不为打包修改环境依赖。

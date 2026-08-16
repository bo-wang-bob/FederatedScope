# 全域智汇前端

跨军事域联邦学习单机实验平台。综合态势页使用地图展示中央服务器、四个域子服务器和 60 个逻辑客户端；训练仍由 FederatedScope 单机进程执行，三级结构不改变后端聚合逻辑。

## 启动后端控制服务

在仓库根目录执行：

```bash
export FEDERATEDSCOPE_DATA_ROOT=/path/to/OfficeHomeDataset_10072016
export FEDERATEDSCOPE_MODEL_PATH=/path/to/open_clip_vitb16.bin
export FEDERATEDSCOPE_API_STATE_DIR=/path/to/writable/experiment-state
/root/.local/share/mamba/envs/pfedba/bin/python -m federatedscope.standalone_api.app \
  --host 127.0.0.1 --port 8000
```

`FEDERATEDSCOPE_API_STATE_DIR` 可省略，默认使用仓库中的 `exp/standalone_api`。数据目录未就绪时，能力接口仍可访问，场景页使用内置 OfficeHome 数据规模生成可复现预览；实验预检会拒绝启动并返回明确原因。数据目录就绪后，场景预览直接扫描四域实际训练样本，应用场景时持久化精确客户端划分清单，训练入口严格重放该清单。

后端启动后可检查：

```bash
curl http://127.0.0.1:8000/api/health
```

## 启动前端

```bash
npm install
npm run dev
```

访问 `http://127.0.0.1:5173/`。开发服务器会把 `/api` 转发到 `http://127.0.0.1:8000`。使用其他地址时设置：

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

首次运行依次进入“场景与异构分析”应用场景，再进入“实验配置”执行运行预检并启动。启动成功后会自动跳转到运行监控；后端日志保存在状态目录的 `runs/<experimentId>/runner.log`，实验记录页也可查看日志尾部和当次场景快照。

## 页面

- 综合态势：全屏地图和三级训练链路。
- 场景与异构分析：设置狄利克雷参数，查看四域 60 个客户端的样本数量和类别比例，并固化可重放的场景快照。
- 实验配置：使用快捷方案或精细参数配置 FedAvg、FedProx、异构解决、隐私保护或后门攻防实验；支持预检、导出配置和启动任务。
- 运行监控：订阅后端快照和结构化 SSE 事件，展示客户端状态、阶段防御结果和当前实验类型的真实指标，并支持停止任务。
- 实验记录：筛选持久化实验，查看配置、场景、日志和最终指标，并可基于同一配置重新运行。

隐私攻击效果、对照分析和系统设置不作为独立页面提供。隐私与后门实验使用互斥配置，后门任务不会执行隐私攻击或隐私保护。

## 构建与测试

```bash
npm test
npm run build
npm run test:e2e
```

端到端测试显式使用固定种子场景适配器，并通过网络拦截验证启动跳转；产品默认连接真实后端，不会在连接失败后继续生成伪实时轮次。

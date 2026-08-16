# 全域智汇前端

跨军事域联邦学习单机实验平台。综合态势页使用地图展示中央服务器、四个域子服务器和 60 个逻辑客户端；训练仍由 FederatedScope 单机进程执行，三级结构不改变后端聚合逻辑。

## 启动后端控制服务

在仓库根目录执行：

```bash
export FEDERATEDSCOPE_DATA_ROOT=/path/to/OfficeHomeDataset_10072016
export FEDERATEDSCOPE_MODEL_PATH=/path/to/open_clip_vitb16.bin
/root/.local/share/mamba/envs/pfedba/bin/python -m federatedscope.standalone_api.app \
  --host 127.0.0.1 --port 8000
```

数据目录未就绪时，能力接口仍可访问，场景页使用内置 OfficeHome 数据规模生成可复现预览；实验预检会拒绝启动并返回明确原因。数据目录就绪后，场景预览直接扫描四域实际训练样本，并与训练入口采用相同种子和划分算法。

## 启动前端

```bash
npm install
npm run dev
```

访问 `http://127.0.0.1:5173/`。开发服务器会把 `/api` 转发到 `http://127.0.0.1:8000`。使用其他地址时设置：

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

## 页面

- 综合态势：全屏地图和三级训练链路。
- 场景与异构分析：设置狄利克雷参数，查看四域 60 个客户端的样本数量和类别比例，并固化场景快照。
- 实验配置：配置 FedAvg、FedProx、异构解决、隐私保护或后门攻防实验，导出配置并启动任务。
- 运行监控：订阅后端快照和 SSE 事件，展示适用于当前实验类型的指标，并支持停止任务。
- 实验记录：读取后端持久化的实验状态、规范化配置和最终指标。

隐私攻击效果、对照分析和系统设置不作为独立页面提供。隐私与后门实验使用互斥配置，后门任务不会执行隐私攻击或隐私保护。

## 构建与测试

```bash
npm test
npm run build
npm run test:e2e
```

端到端测试显式使用固定种子场景适配器，并通过网络拦截验证启动跳转；产品默认连接真实后端，不会在连接失败后继续生成伪实时轮次。

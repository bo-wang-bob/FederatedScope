# 跨域联邦学习 · 前端原型

分支：`prototype/frontend`。对应后端：[prototype/backend](https://github.com/bo-wang-bob/FederatedScope/tree/prototype/backend)。

从已部署版本 `5ac3e453d69f33596c193e15daa808aaca52865b` 拆分，原 `frontend/` 工程提升到本分支根目录；应用源码、依赖锁文件、测试和图片保持不变。包含训练、模型验证、独立评测、对比、地图模拟及隐私/后门预留入口。其他设计文档保留作历史参考，以本说明为准。

## 启动

使用 Node.js 22.12+ 和 npm，在本分支根目录运行：

```bash
npm ci
npm run dev
```

打开 `http://127.0.0.1:5173/`，`/api` 默认代理到后端 `http://127.0.0.1:8001`。本机连接现有服务器隧道时：

```powershell
$env:FS_API_PROXY='http://127.0.0.1:18001'
npm run dev
```

Linux/macOS：`FS_API_PROXY=http://127.0.0.1:18001 npm run dev`。保持同源代理，不需要修改后端跨站保护。

## 构建与验证

```bash
npm test -- --pool=threads --maxWorkers=1
npm run build
```

生产产物为 `dist/`。在后端设置 `FEDERATEDSCOPE_FRONTEND_DIST` 指向该目录后启动 API，即可由同一服务提供页面及接口。不要把本分支合并覆盖后端分支；二者是分别维护的交付快照。

`npm run dev:design` / `npm run build:design` 保留不连接 API 的设计预览。`/demo` 为明确标记的前端地图模拟，不代表真实训练。

拆分源版本已通过 61 项前端测试和生产构建，并完成真实单图预测、独立评测与导出验收；本次只调整分支目录和说明，没有重新训练。历史一体化服务器专用的 `scripts/publish_static.py` 不纳入独立前端包。

## 资源范围

不含 `node_modules`、构建产物、数据集、模型、特征缓存、任务记录或密钥。页面图片为原工程已有静态素材，来源见 [图片说明](src/design/media/README.md)。真实模型与测试图像从后端获取，历史特征来源限制仍由页面披露。

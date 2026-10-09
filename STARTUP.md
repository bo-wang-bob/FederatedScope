# FederatedScope 前后端原型 · 启动手册

本机启动「跨域协同训练」平台前后端的完整步骤。**照抄即可**，命令均已实测。

- 前端：`prototype/frontend` 分支，Vite + React，端口 **5173**
- 后端：`prototype/backend` 分支（本机为 detached worktree），Python 平台，端口 **8001**
- 两个分支是**分别维护的交付快照，不要互相合并**。

---

## 0. 环境前提（先确认这三件事）

| 项目 | 本机路径 / 值 |
|---|---|
| 后端解释器 | `C:/Users/pc/miniconda3/envs/cerp/python.exe`（torch 2.5.1+cu121，RTX 4090 可用） |
| 前端运行时 | Node `C:/Users/pc/.workbuddy/binaries/node/versions/22.22.2-6/node.exe` |
| 后端工作区 | `C:/Users/pc/Desktop/sf/FederatedScope-backend`（detached worktree，指向 `origin/prototype/backend`） |
| 前端工作区 | `C:/Users/pc/Desktop/sf/FederatedScope`（仓库根，`prototype/frontend`） |

### 0.1 Git Bash 的 PATH 是坏的（每次开新 shell 都要先导出）

`git / ls / grep / curl` 等在默认 PATH 下会 `command not found`。每条命令前先执行：

```bash
export PATH="/c/Users/pc/.workbuddy/binaries/PortableGit/versions/1.2.0/bin:/c/Windows/System32:/c/Windows:/usr/bin:/bin:$PATH"
```

### 0.2 代理会让 git / npm / curl 假超时

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
# 测本地端口一律带 --noproxy '*'
curl -s --noproxy '*' http://127.0.0.1:8001/api/health
```

### 0.3 环境变量必须是 Windows 风格路径

`C:\...`（反斜杠）而不是 Git Bash 的 `/c/...`，否则 Python 会拼出 `/c/...\Art` 一类混合路径而失败。
**唯一例外**是 `FS_PLATFORM_IMPORT_ROOTS`，因为它是 JSON，用正斜杠 `C:/...` 免转义。

---

## 1. 如果后端工作区还不存在：创建 detached worktree

```bash
cd /c/Users/pc/Desktop/sf/FederatedScope
git fetch origin
git worktree add --detach "C:/Users/pc/Desktop/sf/FederatedScope-backend" origin/prototype/backend
```

> 不要在仓库根直接 `checkout prototype/backend`——那会换掉前端工作区。
> 嵌套分支名（`prototype/*`）在本机 git 下写入不可靠，所以用 detached worktree。

---

## 2. 启动后端（端口 8001）

```bash
export PATH="/c/Users/pc/.workbuddy/binaries/PortableGit/versions/1.2.0/bin:/c/Windows/System32:/c/Windows:/usr/bin:/bin:$PATH"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

cd /c/Users/pc/Desktop/sf/FederatedScope-backend

export FS_PLATFORM_RESOURCES='C:\Users\pc\sf\FL'
export FS_BACKDOOR_BASE='C:\Users\pc\Desktop\sf\FederatedScope-backend\exp\sabre_newdataset'
export FS_BACKDOOR_VIT_WEIGHTS='C:\Users\pc\sf\FL\open_clip_vitb16.bin'
export FS_PLATFORM_IMPORT_ROOTS='["C:/Users/pc/sf/FL/military_aircraft_staging"]'
export FS_BACKDOOR_DEVICE=cuda
export PYTHONIOENCODING=utf-8

"C:/Users/pc/miniconda3/envs/cerp/python.exe" -m federatedscope.standalone_api.platform_app \
  --host 127.0.0.1 --port 8001 \
  --state-dir 'C:\Users\pc\Desktop\sf\FederatedScope-backend\exp\platform'
```

后台常驻写法（日志落文件，方便排查）：

```bash
... > /c/Users/pc/Desktop/sf/FederatedScope/.workbuddy/tmp/backend_run.log 2>&1 &
```

### 各环境变量作用

| 变量 | 作用 | 不配的后果 |
|---|---|---|
| `FS_PLATFORM_RESOURCES` | 资源根（数据集的父目录） | 找不到数据资产 |
| `FS_BACKDOOR_BASE` | 后门实验结果组的基准目录 | 自动发现兜底，可能找不到结果组 |
| `FS_BACKDOOR_VIT_WEIGHTS` | ViT-B/16 open_clip 权重 | **训练直接 409**（仓库里没有 `resources/models/`） |
| `FS_PLATFORM_IMPORT_ROOTS` | 「服务器路径」导入白名单（JSON 数组） | 路径导入 403 |
| `FS_BACKDOOR_DEVICE` | 计算设备（`cuda` / `cpu`） | 默认可能走 CPU，慢很多 |

---

## 3. 启动前端（端口 5173）

首次（或 `node_modules` 缺失时）先装依赖：

```bash
cd /c/Users/pc/Desktop/sf/FederatedScope
npm ci --registry=https://registry.npmmirror.com     # 约 9 分钟 / 239 包
```

启动 dev server：

```bash
export PATH="/c/Users/pc/.workbuddy/binaries/PortableGit/versions/1.2.0/bin:/c/Windows/System32:$PATH"
export PATH="/c/Users/pc/.workbuddy/binaries/node/versions/22.22.2-6:$PATH"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

cd /c/Users/pc/Desktop/sf/FederatedScope
node node_modules/vite/bin/vite.js --host 0.0.0.0 --port 5173
```

前端把 `/api` 反代到后端（`FS_API_PROXY`，默认 `http://127.0.0.1:8001`）。

访问：

- 主页 / 平台：<http://127.0.0.1:5173/>
- 后门研究页：<http://127.0.0.1:5173/?view=backdoor>
- 逐样本对照：<http://127.0.0.1:5173/?view=backdoorCompare&job=<任务id>>

---

## 4. 启动后自检（四步，能定位到具体哪一层断了）

```bash
# 1. 后端存活（commit 应等于后端分支 HEAD）
curl -s --noproxy '*' http://127.0.0.1:8001/api/health

# 2. 后门测试集（期望 exported:true；本机军机组 total=225 / 5 类）
curl -s --noproxy '*' http://127.0.0.1:8001/api/platform/backdoor/testset

# 3. 训练可用性（期望 runnable:true，datasets/testsets 非空，attacks 四个）
curl -s --noproxy '*' http://127.0.0.1:8001/api/platform/backdoor/training

# 4. 经 vite 代理（验证前后端联通）
curl -s --noproxy '*' http://127.0.0.1:5173/api/health
```

本次实测输出：

```
1. {"status":"ok","mode":"single-host","commit":"810d4b2..."}
2. exported=True | total=225 | classNames=['B-52','C-130','C-17','F-15','F-16']
3. runnable=True | attacks=['数据投毒','模型投毒','常见后门','新型后门']
   datasets=1 | testsets=1 | attack=model_poisoning
4. 代理 ok
```

---

## 5. 停止服务

```bash
# 查占用端口的 PID
netstat -ano | grep ":8001" | grep LISTENING
netstat -ano | grep ":5173" | grep LISTENING
```

```powershell
# 用 PowerShell 杀进程（taskkill //PID 在 Git Bash 里参数会被转义坏）
Stop-Process -Id <PID> -Force -ErrorAction SilentlyContinue
```

---

## 6. 常见坑（踩过的）

1. **改了后端代码必须重启进程。**
   Python 启动时就把 `platform_backdoor_training.py` 等模块载入内存，**文件落盘不会热更**。
   典型症状：新加的前端字段/下拉框不出现。一步定位——看 `/training` 的 `data` 里有没有新键，
   没有就是旧进程，重启即可，不必怀疑前端。
   （前端 `.tsx` 改动只需浏览器刷新，Vite 按需从磁盘编译。）

2. **改了环境变量也要重启进程**，同理。

3. **`rounds` 不会把攻击窗口推到后半程。** 模板里写死 `start_round: 40`，
   平台只在 `start_round >= rounds` 时才重算成 `rounds//2`。长训练前需临时把模板
   `start_round` 改成哨兵大数触发重算，`POST` 返回后立即还原。

4. **页面首次挑图必然是随机抽样。** `platform_backdoor._start_precompute` 是死代码；
   要加权展示得手跑 `scripts/backdoor/precompute_trigger_predictions.py`，
   再调 `POST /api/platform/backdoor/testset/apply` 清进程内缓存。

5. **模板是烟测配方**（冻结 ViT + 纯线性头 + 1 step/round + lr 1e-4），
   5 类任务干净精度上限约 **0.67**，加轮数无效；想更高得改模板（无 API 入口）。

6. **别清 `<state>/mpl`**（matplotlib 字体缓存，预置了 `fontlist-v330.json`）；
   本机有外部 safe-delete 保护，批量删除会挂死，服务内部一律「改名挪进 `.trash`」。

7. **`exp/`、`.workbuddy/`、`node_modules/` 都在 `.gitignore` 内**，产物不污染分支。

---

## 7. 本机数据资产（不随 Git 分发，换机器要单独拷）

| 路径 | 内容 |
|---|---|
| `C:\Users\pc\sf\FL\MilitaryAircraft3D` | 军机原始数据，3 域 × 5 类 × 50 张 |
| `C:\Users\pc\sf\FL\military_aircraft_staging` | 按平台要求重排的 `train\<类>` / `test\<类>`（525 / 225 张） |
| `C:\Users\pc\sf\FL\open_clip_vitb16.bin` | ViT-B/16 open_clip 权重（后门训练用） |
| `FederatedScope-backend\exp\sabre_newdataset` | 本机新训军机结果组 |
| `FederatedScope-backend\exp\platform` | 平台状态目录（`--state-dir`） |

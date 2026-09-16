# 后门研究（Backdoor Lab）—— 工作总结与运行手册

> 适用分支
> - 后端：`backend_backdoor`（本机 worktree `C:\Users\pc\sf\protype-backend`，HEAD `1d75bee`）
> - 前端：`fronted_backdoor`（主工作区 `C:\Users\pc\sf\protype`）
>
> 最后更新：2026-09-15

---

## 一、本次做了什么

### 1. 前端跑起来 + 分支结构梳理

- 前端分支 `fronted_backdoor`，栈为 Vite 7 + React 19 + TS 5.9 + antd 6 + echarts 6 + react-router 7。
- 入口链：`src/main.tsx` → `RootApp.tsx`（按 `import.meta.env.MODE` 分流 design / platform）→ `platform/PlatformApp.tsx`。
- **路由是 query 驱动**：所有页面挂在 `/*` 上，用 `?view=home|train|jobs|experience|evaluate|compare|privacy|backdoor` 切换，逻辑在 `platform/navigation.tsx` 的 `resolveView`。改路由不要动 `<Route>`。
- 约一半代码是死代码（`src/App.tsx`、`src/pages/*`、`src/components/*`、`src/store/useAppStore.ts`、`src/mock/*`、`src/api/experimentApi.ts` 等），只有单元测试还在引用，看代码可跳过。
- `npm ci` + `npm run build`（`tsc -b && vite build`）均通过。

### 2. 后门绘图脚本脱离 FL 仓库

原先 `scripts/backdoor/plot_predictions.py` 必须在另一个仓库 `C:\Users\pc\sf\FL` 根目录执行，原因是两个叠加问题：

1. **仓库根路径算错一级**：脚本在 `scripts/backdoor/` 中，`dirname(__file__)/..` 得到 `scripts/` 而非仓库根，导致 `import federatedscope` 掉回 site-packages（0.3.0，无 `ggeur_server`）。FL 能跑纯属巧合——`import eval` 命中了 `FL/scripts/eval.py`，它位于 `scripts/` 下，`..` 正好是仓库根。
   → 改为 `_find_repo_root()`：向上逐级找含 `federatedscope/` 目录的那层。
2. **后端分支的 `GGEURServer` 是裁剪版**（3909 行 vs FL 5387 行，缺 80 个方法）。绘图只用到 `_load_a3fl_test_loaders` 和 `_restore_a3fl_tensor`，已按 FL 实现补进 `scripts/backdoor/eval.py` 的 `GGEUREvalServer`，**没有改动 `federatedscope/`**。

另新增 `--data-root` 参数，用于覆盖 `config.yaml` 里的 `data.root`。

### 3. 前后端对接：后门研究页

前端新增「后门研究」页（侧边栏入口，不再是“未接入”占位）：

流程 = 进页面自动随机抽 20 张测试图（缩略图网格，可点选增删、按域/类别筛选、换一批）→ 点「生成三连对比」→ 后端约 7 秒出图 → 展示 **clean / triggered / defense 三张一组**的结果图 + 结论段落 + 逐样本对照表（被劫持的样本标红）。

**改动清单**

| 仓库 | 文件 | 说明 |
|---|---|---|
| 后端 | `scripts/backdoor/run_group.py`（新增） | 一次加载测试集与特征提取器，依次套用多个 run 的 MLP head + trigger，产出 `clean.png / triggered.png / defense.png / defense_clean.png + result.json`。复用单个 server，两个 run 只要 ~7 秒 |
| 后端 | `federatedscope/standalone_api/platform_backdoor.py`（新增） | `BackdoorService`：独立 job 目录 + 单任务锁，spec.json + 子进程 + job.json 轮询，自动发现 attack/defense run |
| 后端 | `federatedscope/standalone_api/platform_app.py`（改） | 挂载 `/api/platform/backdoor/*` 路由 |
| 前端 | `src/platform/backdoor.tsx` + `backdoor.css`（新增） | `BackdoorLab` 页面，风格沿用 studio.css 深色变量与 `platform-panel` / `platform-stats` |
| 前端 | `navigation.tsx` / `extensions.tsx` / `PlatformApp.tsx` / `api.ts`（改） | 注册视图、移出 planned、渲染、加类型 |

---

## 二、环境准备

### Python 环境

后端依赖很轻。本机用的是已有 conda 环境：

```
C:\Users\pc\miniconda3\envs\cerp\python.exe   # 3.10.20 + torch 2.5.1+cu121，CUDA 可用
```

必需包：`torch`、`numpy`、`yaml`、`PIL`、`matplotlib`（绘图用）、`psutil`（平台服务用，本机原本缺，已补装）。

```bat
python -m pip install psutil matplotlib
```

### 前端环境

```bat
cd C:\Users\pc\sf\protype
npm ci
```

---

## 三、标准启动方式（推荐）

### 1. 启动后端（Windows / 本机）

```bat
cd C:\Users\pc\sf\protype-backend
set FS_PLATFORM_RESOURCES=C:\Users\pc\sf\FL
set FS_PLATFORM_DATASETS=C:\Users\pc\sf\FL
set FS_BACKDOOR_DATA_ROOT=C:\Users\pc\sf\FL\OfficeHomeDataset_10072016
python -m federatedscope.standalone_api.platform_app --host 127.0.0.1 --port 8001 --state-dir C:\Users\pc\sf\protype-backend\exp\platform
```

> **路径必须是 Windows 风格 `C:\...`**。用 Git Bash 的 `/c/...` 会让 Python 拼出 `/c/...\Art` 这类混合路径而失败。

### 2. 启动前端

```bat
cd C:\Users\pc\sf\protype
npm run dev          # http://127.0.0.1:5173/ ，局域网 http://10.129.248.111:5173/
```

前端 dev server 会把 `/api` 代理到 `FS_API_PROXY` 或默认 `http://127.0.0.1:8001`（`changeOrigin: false`）。若后端不在 8001：

```bat
set FS_API_PROXY=http://127.0.0.1:18002
npm run dev
```

### 3. 打开页面

浏览器访问 `http://127.0.0.1:5173/?view=backdoor`，左侧进入「后门研究」。

### 4. 健康检查

```bat
curl http://127.0.0.1:8001/api/health
curl http://127.0.0.1:8001/api/platform/backdoor/testset
```

> 若在本机用 curl 测本地端口，环境里带 HTTP 代理，需加 `--noproxy '*'`，否则返回 502。

---

## 四、只跑命令行脚本（不启服务）

### 4.1 三连图一次生成（推荐，快）

由服务内部调用，也可单独跑：

```bat
cd C:\Users\pc\sf\protype-backend
python scripts\backdoor\run_group.py --spec <spec.json 路径>
```

`spec.json` 字段：

```json
{
  "base": "C:/Users/pc/sf/protype-backend/exp/sabre",
  "runs": { "attack": "sabre_vit_42", "defense": "sabre_vit_defense_42" },
  "ids": ["Clipart_00256", "..."],
  "output": "C:/.../输出目录",
  "device": "cuda",
  "dataRoot": "C:/Users/pc/sf/FL/OfficeHomeDataset_10072016"
}
```

### 4.2 原始单 run 绘图脚本

```bat
cd C:\Users\pc\sf\protype-backend
python scripts\backdoor\plot_predictions.py --base exp\sabre --device cuda --ids-file exp\sabre\ids_txt
```

数据集路径不同加 `--data-root <路径>`。注意：

- 已存在同名输出图的 run 会被**跳过**，要重画需删旧图或换 ids 文件名（输出后缀取自 ids 文件名）。
- 首次运行会往 `{base}\testset_images` 导出全部 4679 张测试图（约 81 MB），有 `.done` 标记后不再重复。
- 耗时约 1 分 50 秒（时间花在导出图片上），`run_group.py` 走内存不落盘，只要 7 秒。

---

## 五、环境变量速查

| 变量 | 默认值 | 说明 |
|---|---|---|
| `FS_BACKDOOR_BASE` | `<repo>/exp/sabre` | 实验根目录，下面每个子目录是一个 run |
| `FS_BACKDOOR_DATA_ROOT` | 空（`config.yaml` 的 `data.root`） | 数据集根目录，覆盖配置 |
| `FS_BACKDOOR_DEVICE` | `cuda` | 推理设备，无 GPU 填 `cpu` |
| `FS_BACKDOOR_RUNS` | 自动发现 | 形如 `attack=sabre_vit_42,defense=sabre_vit_defense_42` |
| `FS_PLATFORM_RESOURCES` | `/root/autodl-tmp/FederatedScope` | 平台资源根 |
| `FS_PLATFORM_DATASETS` | `/root/autodl-tmp/datasets` | 数据集根 |
| `FS_API_PROXY` | `http://127.0.0.1:8001` | 前端 dev 代理目标 |
| `FEDERATEDSCOPE_FRONTEND_DIST` | `<repo>/frontend/dist` | 后端托管前端静态产物的路径 |

run 自动发现规则：`base` 下含 `*_final_mlp_head.pt` 的子目录为候选，优先取 seed 后缀（默认 42）一致的一组，名字含 `defense` 的归 defense，其余第一个归 attack。

---

## 六、接口清单

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/platform/backdoor/testset` | 测试集元数据：已导出数、总数、类别名、各域计数、attack/defense run |
| GET | `/api/platform/backdoor/testset/<id>/image` | 单张测试图缩略图 |
| POST | `/api/platform/backdoor/pick` | 随机挑图，body `{count, seed, domain?, className?}` |
| POST | `/api/platform/backdoor/jobs` | 提交生成任务，body `{ids[], name?, idempotencyKey}` |
| GET | `/api/platform/backdoor/jobs/<id>` | 查询任务状态与结果 |
| GET | `/api/platform/backdoor/jobs/<id>/image/<clean\|triggered\|defense\|defenseClean>` | 结果图 |
| GET | `/api/platform/backdoor/jobs/<id>/logs` | 任务日志 |
| POST | `/api/platform/backdoor/jobs/<id>/stop` | 终止任务 |

> 与平台其它接口一致，提交任务**必须带 `idempotencyKey`**，否则 422。

---

## 七、部署到其它服务器（Linux）注意事项

1. **数据不随 Git 分发**。`exp/` 已被 gitignore，需要单独拷贝：
   - `exp/sabre/sabre_vit_42/`、`exp/sabre/sabre_vit_defense_42/`（含 `config.yaml`、`*_final_mlp_head.pt`、`trigger` 文件）
   - Office-Home 数据集（注意域目录必须是 `Real_World` 下划线形式，不是 `Real World`）
   - CLIP 权重（`open_clip_vitb16.bin`）
2. **`config.yaml` 里有硬编码 Windows 路径**，换机器必须改或靠环境变量覆盖：
   - `ggeur.feature_cache_dir`（本机 `C:\Users\pc\sf\FL\exp\feature_cache`）
   - `ggeur.clip_model_path`（本机 `C:\Users\pc\sf\FL\open_clip_vitb16.bin`）
3. Linux 启动命令示例：

```bash
cd /path/to/protype
export FS_BACKDOOR_BASE=/path/to/exp/sabre
export FS_BACKDOOR_DATA_ROOT=/root/autodl-tmp/datasets/OfficeHomeDataset_10072016
export FS_BACKDOOR_DEVICE=cuda
python -m federatedscope.standalone_api.platform_app --host 0.0.0.0 --port 8001 \
  --state-dir /path/to/exp/platform
```

4. 前端生产构建后，可用 `FEDERATEDSCOPE_FRONTEND_DIST` 指向 `dist/`，由后端直接托管，省掉一层代理。

---

## 八、已知问题与待办

- **训练仍跑不了（本机）**：没有 MilitaryAircraft3D 数据与 ViT 冻结特征缓存，`exp/distributed_feature_cache` 不存在，所有 group `cacheFound=False`，预检会失败。后门研究页不依赖训练，不受影响。
- **后端一个小瑕疵**：军机组预检报错文案复用了 DomainNet（`DomainNet root not found`），实际是军机数据集。
- **代码尚未提交**，两边都有未提交改动：
  - 后端：`M federatedscope/standalone_api/platform_app.py`、`?? platform_backdoor.py`、`?? scripts/backdoor/`
  - 前端：`M PlatformApp.tsx / api.ts / extensions.tsx / navigation.tsx`、`?? backdoor.tsx / backdoor.css`
- 若要清理 worktree：`git worktree remove C:/Users/pc/sf/protype-backend`（不影响主仓库）。

---

## 九、实测结果参考

同一批 20 张随机测试图：

| 阶段 | 结果 |
|---|---|
| clean | 准确率 75%（15/20） |
| triggered | ASR 65%（13/20 被劫持为 Alarm_Clock），准确率降至 35% |
| defense | ASR 0/20，准确率回升至 90% |

浏览器端到端跑通过一次（20 张缩略图 → 生成 → 3 张结果图 + 21 行逐样本表），`tsc` 与 `vite build` 均通过，页面无报错（唯一 404 是 favicon，项目本身没有）。

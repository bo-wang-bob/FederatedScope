@echo off
REM ---------------------------------------------------------------------------
REM 后门研究前后端一键启动
REM
REM   后端 (-backend_backdoor) : http://127.0.0.1:8001
REM   前端 (fronted_backdoor)  : http://localhost:5173  -> ?view=backdoor
REM
REM 后端默认实验目录优先级: exp\sabre_newdataset (军机三域新实验) > exp\sabre (旧实验)。
REM 会自动在「后端仓库 exp\」「状态目录同级 exp\」「进程工作目录 exp\」里找, 因此
REM 正常情况下不需要设置 FS_BACKDOOR_BASE。
REM
REM   想强制指定:  set FS_BACKDOOR_BASE=<某个实验目录>
REM   想加搜索位置: set FS_BACKDOOR_SEARCH_ROOTS=<目录1>;<目录2>
REM   切回旧实验:  set FS_BACKDOOR_BASE=%FRONTEND_DIR%\exp\sabre
REM ---------------------------------------------------------------------------

set PY=C:\Users\pc\miniconda3\envs\cerp\python.exe
set NODE=C:\Users\pc\.workbuddy\binaries\node\versions\22.22.2-3\node.exe

set BACKEND_REPO=C:\Users\pc\sf\protype
set FRONTEND_DIR=C:\Users\pc\Desktop\logs\FederatedScope
set STATE_DIR=%FRONTEND_DIR%\exp\platform

REM --- MilitaryAircraft3D 原图根目录 (config.yaml 的 data.root 是相对路径) ---
set FS_BACKDOOR_DATA_ROOT=C:\Users\pc\sf\FL\MilitaryAircraft3D
set FS_BACKDOOR_DEVICE=cuda

start "backend 8001" cmd /k "cd /d %BACKEND_REPO% && "%PY%" -m federatedscope.standalone_api.platform_app --host 127.0.0.1 --port 8001 --state-dir "%STATE_DIR%""
start "frontend 5173" cmd /k "cd /d %FRONTEND_DIR% && "%NODE%" node_modules/vite/bin/vite.js --host 0.0.0.0"

echo backend  : http://127.0.0.1:8001
echo frontend : http://localhost:5173/?view=backdoor

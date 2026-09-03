# 三机完整 MLP 往返 QPS 验证方案

## 固定三机链路

本验证沿用 2026-07-22 已成功完成正式实验的实验室内部拓扑：

```text
管理链路：
本地控制端
  -> 8G 管理机（唯一入口）
      -> ssh root@10.112.81.135
      -> ssh pc@10.129.248.111

业务链路：
8G / 10.129.222.189
  -> 第三台子服务器 / 10.129.248.111
  -> 4090 根服务器 / 10.112.81.135
```

- `bjb*.seetacloud` 不属于这组三机，不得用于状态判断、部署或压测。
- 本地到 8G 的连接只承担管理；从 8G 打开的 SSH 跳转通道也只承担管理。
- 模型状态上传、下载、ACK 和迟到边缘节点接入必须全部使用上述真实
  `10.x` 地址，禁止通过 SSH 隧道承载业务流量。
- 自动化审计必须记录管理路由为
  `8g_jump_to_10.112.81.135` 和
  `8g_jump_to_10.129.248.111`。

## 指标口径

本测试不运行完整训练。每个事务必须完成：

1. 客户端向子服务器请求模型；
2. 子服务器返回完整 MLP 字节；
3. 客户端把同一份完整 MLP 上传给子服务器；
4. 子服务器读取完整上传并返回 ACK。

严格指标为：

```text
roundtrip_qps = 成功完成下载和上传的事务数 / GO 到最后完成事务的窗口秒数
```

达标条件：

```text
aggregate_qps >= 10000
success_ratio = 1.0
payload_bytes_per_download >= 133380
payload_bytes_per_upload >= 133380
业务连接只使用真实 10.x IP
```

不能使用 `payload: omitted`、仅 ACK 请求、loopback 或 SSH 转发结果作为正式证据。

## 当前硬件前检

当前链路：

| 节点 | 链路 |
|---|---:|
| 4090 `10.112.81.135` | 10000 Mbps |
| 第三台 `10.129.248.111` | 400 Mbps WLAN |
| 8G `10.129.222.189` | 200 Mbps WLAN |

当前线性 MLP 为 33345 个 float32 参数，即每个方向 133380 bytes。执行：

```powershell
python scripts\check_mlp_qps_bandwidth.py `
  --target-qps 10000 `
  --client-link-mbps 200 `
  --subserver-link-mbps 400
```

前检会以退出码 2 拒绝正式压测。当前协议开销假设下，物理上限约为 178
transactions/s；实际值通常更低。增加同一 WLAN 网卡后的进程数不能突破该上限。

## 网络升级后的三机分片

建议8G和第三台升级到25GbE。使用独立端口，避开正式训练的60050和61000–61003：

```text
4090接收分片：62010–62014
第三台接收分片：62020–62024
8G：10组客户端负载，每组连接一个接收分片
```

每个接收分片管理1000个逻辑客户端，总计10000个同步事务。4090和第三台各承担
5000个事务，8G作为统一负载发生器。所有连接均使用真实10.x地址。

升级后先执行：

```powershell
python scripts\check_mlp_qps_bandwidth.py `
  --target-qps 10000 `
  --client-link-mbps 25000 `
  --subserver-link-mbps 25000
```

只有 `feasible_with_headroom=true` 才进入正式压测。

## 单分片命令

接收端：

```text
python scripts/benchmark_headonly_mlp_upload_window.py server
  --host 0.0.0.0
  --port <PORT>
  --clients 1000
  --include-download
  --payload-file <SERIALIZED_MLP_PT>
  --output <RESULT_JSON>
```

8G负载端：

```text
python scripts/benchmark_headonly_mlp_upload_window.py client
  --host <REAL_10X_IP>
  --port <PORT>
  --clients 1000
  --include-download
  --payload-file <SERIALIZED_MLP_PT>
```

十个接收分片必须先启动，再并行启动十组客户端。禁止使用管理SSH隧道作为
`--host`。正式证据必须使用同一份真实序列化MLP文件；不传 `--payload-file`
时仅用于代码smoke test的等尺寸字节负载。

## 汇总

把十个接收分片的 JSON 放到同一目录后执行：

```powershell
python scripts\summarize_mlp_roundtrip_qps.py `
  results\subserver_*.json `
  --target-qps 10000 `
  --output results\aggregate_summary.json
```

汇总器使用“最早GO到最晚完成”的全局窗口，不会把各分片局部QPS直接相加。

## 与正式训练的隔离

- 正式压测不得与三机训练队列同时运行；
- 压测前确认60050、61000–61003上的正式Case已经停止；
- QPS仅使用62010–62024；
- 保留每个分片JSON、客户端stdout/stderr、三机网卡计数器和命令审计；
- 测试后停止全部分片，不修改正式训练完成标记。

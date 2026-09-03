# 测试大纲逐步验证脚本

本目录把测试大纲中的操作封装为脚本。Word 文档只给出脚本文件的绝对路径和执行命令，脚本负责产生结构化证据。

- Windows 客户端机/子服务器机：`run_case_step.ps1`
- Linux 根服务器机：`run_case_step.sh`
- Windows 环境采集：`collect_environment.ps1`
- Linux 环境采集：`collect_environment.sh`
- 用例与模型/数据集/方法映射：`case_catalog.json`

`-PlanOnly` 或 `--plan-only` 仅检查参数映射并生成计划证据，不启动训练。正式测试不得使用计划模式；正式步骤只有在相应配置、完成标记或汇总文件实际存在时才输出 `STATUS=PASS`。

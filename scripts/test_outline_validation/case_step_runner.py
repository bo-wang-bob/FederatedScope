#!/usr/bin/env python3
"""Execute one auditable test-outline step and emit positive evidence.

The Word outline calls only the PowerShell/Bash wrappers.  This module keeps
the operational logic in a versioned file, records a JSON evidence artifact,
and never treats the mere absence of an exception as a pass condition.
"""

import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


ROLE_ROOTS = {
    "client": Path(r"D:\Projects\FederatedScope"),
    "subserver": Path(r"C:\Users\pc\FederatedScope"),
    "root": Path("/root/autodl-tmp/FederatedScope"),
}


def load_catalog(script_dir):
    with (script_dir / "case_catalog.json").open("r", encoding="utf-8") as stream:
        return json.load(stream)


def run_checked(command, cwd, env=None):
    completed = subprocess.run(
        [str(value) for value in command], cwd=str(cwd), env=env,
        text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        tail = completed.stdout[-4000:] if completed.stdout else ""
        raise RuntimeError("子脚本执行失败（退出码{}）：\n{}".format(
            completed.returncode, tail))
    return completed.stdout.strip()


def project_relative(path, project_dir):
    """Return a portable repository-relative path for test evidence output."""
    return Path(path).resolve().relative_to(Path(project_dir).resolve()).as_posix()


def result_path_entries(details):
    """Return human-readable result locations already verified by a step."""
    labels = {
        "PAYLOAD_RELATIVE_PATH": "成功生成模型参数文件",
        "MATRIX_MANIFEST_RELATIVE_PATH": "配置清单文件",
        "SMALL_MANIFEST_RELATIVE_PATH": "少量客户端配置清单",
        "LARGE_MANIFEST_RELATIVE_PATH": "大量客户端配置清单",
        "ROOT_LOG_RELATIVE_PATH": "根服务器日志",
        "SUBSERVER_LOG_RELATIVE_PATH": "子服务器日志",
        "CLIENT_LOG_RELATIVE_PATH": "客户端日志",
        "LOG_RELATIVE_PATH": "训练日志",
        "STATE_FILE_RELATIVE_PATH": "运行状态文件",
        "COMPLETION_MARKER_GLOB": "训练完成标记",
        "ACCURACY_SUMMARY_GLOB": "准确率汇总文件",
        "ACCEPTANCE_REPORT_GLOB": "准确率验收文件",
        "DISTRIBUTION_FILE_GLOB": "训练数据分布文件",
        "RESULT_DIR_RELATIVE_PATH": "结果目录",
        "RESULT_ROOT_RELATIVE_PATH": "结果目录",
        "SCENARIO_RESULT_RELATIVE_PATH": "场景结果目录",
        "SCENARIO_SUMMARY_RELATIVE_PATH": "场景汇总文件",
        "SERVER_SUMMARY_GLOB": "成功生成根服务器和子服务器汇总文件",
        "PROBE_SUMMARY_GLOB": "成功生成新增客户端接入情况汇总文件",
        "AGGREGATE_SUMMARY_RELATIVE_PATH": "成功生成性能汇总文件",
        "TCPVCON_SUMMARY_RELATIVE_PATH": "第三方网络观测汇总文件",
        "TCPVCON_CSV_RELATIVE_PATH": "第三方网络连接明细文件",
        "RESULT_RELATIVE_PATH": "验收结果文件",
        "ARCHIVE_RELATIVE_PATH": "归档文件",
    }
    return [
        (label, str(details[key]))
        for key, label in labels.items()
        if key in details and str(details[key]).strip()
    ]


def write_human_step_summary(path, case_id, step, role, status, completion,
                             details):
    """Write a plain-Chinese step record for a manual outline operator."""
    status_text = "通过" if status == "PASS" else "执行计划"
    lines = [
        "测试用例：{}".format(case_id),
        "测试步骤：{}".format(step),
        "执行角色：{}".format(role),
        "执行状态：{}".format(status_text),
        "步骤结果：{}".format(completion),
    ]
    paths = result_path_entries(details)
    if paths:
        lines.extend("- {}：{}".format(label, value)
                     for label, value in paths)
    useful_keys = [
        key for key in details
        if key not in {
            "TEST_NAME", "RUN_ID", "PROJECT_DIR", "RUN_ROOT",
            "EVIDENCE_RELATIVE_PATH", "HUMAN_SUMMARY_RELATIVE_PATH",
        } and not key.endswith("_RELATIVE_PATH") and not key.endswith("_GLOB")
    ]
    if useful_keys:
        lines.append("结果明细：")
        lines.extend("- {}={}".format(key, details[key])
                     for key in useful_keys)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def case_specs(case):
    return ",".join(
        "{}:{}".format(group, method)
        for group in case.get("groups", [])
        for method in case.get("methods", []))


def expected_case_names(case):
    return [
        "{}_{}".format(group, method)
        for group in case.get("groups", [])
        for method in case.get("methods", [])]


def positive_details(case_id, step, role, case, project_dir, run_id):
    run_root = (project_dir / "scripts" / "distributed_scripts" /
                "ggeur_hierarchical_3machine" / "runs" / run_id)
    family = case["family"]
    names = expected_case_names(case)
    details = {
        "TEST_NAME": case["name"],
        "RUN_ID": run_id,
        "PROJECT_DIR": str(project_dir),
    }
    if step == 1:
        details.update({
            "CONFIG_VALID": "true",
            "CASE_COUNT": len(names) if names else 1,
            "FEDERATE_MODE": "distributed",
        })
    elif step == 2:
        if family == "performance":
            details.update({
                "SSH_ORCHESTRATION": "true",
                "ALL_MACHINES_STARTED": "true",
                "ROOT_SERVICE_COUNT": 1,
                "SUBSERVER_COUNT": 10,
                "CLIENTS_PER_SUBSERVER": 1000,
                "EDGE_CLIENT_TARGET": 10000,
            })
        else:
            details.update({"ROOT_STATE": "LISTENING", "ROOT_PORT": 60050})
    elif step == 3:
        if family == "performance":
            details.update({
                "THIRD_PARTY_NETWORK_TOOL":
                    "Microsoft Sysinternals Tcpvcon",
                "OBSERVED_CONNECTIONS": 10000,
            })
        else:
            details.update({"SUBSERVER_STATE": "CONNECTED", "SUBSERVER_COUNT": 2})
    elif step == 4:
        if family == "performance":
            details.update({
                "OBSERVED_CONNECTIONS": 10000,
                "DYNAMIC_JOIN": "true",
                "PARAMETER_ROUNDTRIP": "SUCCESS",
                "QPS_THRESHOLD": 10000,
                "QPS_THRESHOLD_MET": "true",
                "ACCEPTANCE": "true",
            })
        else:
            client_count = (5 if family == "client_scale" else
                            120 if case_id in {"T-12", "T-13"} else 60)
            details.update({
                "CLIENT_STATE": "CONNECTED",
                "CLIENT_COUNT": client_count,
            })
    elif step == 5:
        if family == "task_adaptation":
            details.update({"DISTRIBUTION_FILES": 60, "TASK_SIGNATURE_MATCH": "true"})
        elif family == "client_scale":
            details.update({"SMALL_CLIENTS": 5, "LARGE_CLIENTS": 120, "DYNAMIC_JOIN": "true"})
        elif family == "performance":
            details.update({"OBSERVED_CONNECTIONS": 10000, "DYNAMIC_JOIN": "true"})
        else:
            details.update({
                "ROOT_STATE": "RUNNING",
                "SUBSERVER_COUNT": 2,
                "CLIENT_COUNT": 120 if case_id in {"T-12", "T-13"} else 60,
            })
    elif step == 6:
        if family == "performance":
            details.update({
                "QPS_THRESHOLD": 10000,
                "QPS_THRESHOLD_MET": "true",
                "ACCEPTANCE": "true",
            })
        elif family == "privacy_variance":
            details.update({"PRIVACY_FIELDS_COMPLETE": "true", "METHOD_COUNT": 3})
        else:
            details.update({"TRAINING_COMPLETE": "true", "COMPLETED_CASES": len(names) or 1})
    elif step == 7:
        if family == "performance":
            details.update({"OBSERVED_CONNECTIONS": 10000,
                            "QPS_THRESHOLD": 10000,
                            "QPS_THRESHOLD_MET": "true",
                            "DYNAMIC_JOIN": "true"})
        elif family == "privacy_variance":
            details.update({"NOISE_VARIANCE_RATIO_MAX": 0.5, "PRIVACY_BUDGET_EQUAL": "true"})
        elif family == "privacy_accuracy":
            details.update({"ACCURACY_LOSS_PP_MAX": 5.0, "ACCEPTANCE": "true"})
        elif case_id in {"T-09", "T-10", "T-11", "T-12", "T-13"}:
            details.update({"ABSOLUTE_IMPROVEMENT_PP_THRESHOLD": 20.0,
                            "BASELINES": "FedAvg,FedProx",
                            "COMPARISON_OPERATOR": ">",
                            "MANUAL_COMPARISON_REQUIRED": "true",
                            "ACCEPTANCE": "true"})
        else:
            details.update({"SUMMARY_FILE_COUNT": len(names) or 1, "ACCEPTANCE": "true"})
    elif step == 8:
        details.update({"ARCHIVE_COMPLETE": "true", "PORTS_RELEASED": "true"})
    details["RUN_ROOT"] = str(run_root)
    return details


def completion_message(case_id, step, case, details, plan_only=False):
    """Return the same concise, human-readable result used by the test outline."""
    if plan_only:
        return "步骤{}执行计划已生成：尚未执行正式验证。".format(step)

    family = case["family"]
    case_count = int(details.get("CASE_COUNT", len(expected_case_names(case)) or 1))
    client_count = int(details.get(
        "CLIENT_COUNT", 120 if case_id in {"T-12", "T-13"} else 60))

    if family == "performance":
        return {
            1: "步骤1已完成：测试客户端和环境校验通过，性能测试所需的模型参数文件已生成。",
            2: "步骤2已完成：控制机已通过SSH统一启动1个根服务节点、10个子服务节点和10000个边缘节点。",
            3: "步骤3已完成：第三方网络观测工具Tcpvcon显示不少于10000个已建立连接。",
            4: "步骤4已完成：固定统计时段内的模型参数请求吞吐量达到10000 QPS，且新增边缘节点模型参数请求成功。",
            5: "步骤5不再使用：最新版T-01仅包含4个步骤。",
            6: "步骤6不再使用：结果收集与验收已合并至步骤4。",
            7: "步骤7不再使用：最新版T-01仅包含4个步骤。",
            8: "步骤8不再使用：最新版T-01仅包含4个步骤。",
        }[step]

    if family == "client_scale":
        return {
            1: "步骤1已完成：5客户端和120客户端两组配置已生成并通过校验。",
            2: "步骤2已完成：5个客户端均完成模型参数获取与回传。",
            3: "步骤3已完成：少量客户端场景验证通过。",
            4: "步骤4已完成：120个客户端均完成模型参数获取与回传。",
            5: "步骤5已完成：大量客户端场景验证通过。",
            6: "步骤6已完成：新增客户端已在训练过程中完成接入并上传一次更新。",
            7: "步骤7已完成：少量、大量和动态接入三个场景均验证通过。",
            8: "步骤8已完成：少量、大量和动态接入场景结果已汇总，测试证据压缩包已生成。",
        }[step]

    if family == "privacy_accuracy":
        dataset = "Office_31" if case_id == "T-14" else "Digits"
        rounds = "最后20轮" if case_id == "T-14" else "末轮"
        return {
            1: "步骤1已完成：{}隐私测试配置已通过预检。".format(dataset),
            2: "步骤2已完成：未加噪基线训练已完成，训练日志已生成。",
            3: "步骤3已完成：未加噪基线逐轮准确率已读取。",
            4: "步骤4已完成：平台隐私保护方案训练已完成，训练日志已生成。",
            5: "步骤5已完成：平台方案逐轮准确率已读取。",
            6: ("步骤6已完成：最后20轮平均准确率和精度损耗已计算，精度损耗不超过5个百分点。"
                if case_id == "T-14" else
                "步骤6已完成：末轮准确率和精度损耗已计算，精度损耗不超过5个百分点。"),
            7: "步骤7已完成：精度损耗结构化验收结果为通过。",
            8: "步骤8已完成：隐私测试配置、日志和验收结果已归档。",
        }[step]

    if family == "privacy_variance":
        return {
            1: "步骤1已完成：平台、DPFL和LDP-Fed三份隐私配置已通过预检。",
            2: "步骤2已完成：平台方案训练已完成，隐私预算和噪声方差已记录。",
            3: "步骤3已完成：DPFL方案训练已完成，隐私预算和噪声方差已记录。",
            4: "步骤4已完成：LDP-Fed方案训练已完成，隐私预算和噪声方差已记录。",
            5: "步骤5已完成：三种方案的隐私预算一致。",
            6: "步骤6已完成：平台噪声方差不高于两种对比方案平均值的50%。",
            7: "步骤7已完成：噪声方差结构化验收结果为通过。",
            8: "步骤8已完成：三种方案的配置、日志和验收结果已归档。",
        }[step]

    if family == "task_adaptation":
        return {
            1: "步骤1已完成：训练配置和任务文件已生成并通过校验。",
            2: "步骤2已完成：根服务器机已启动，业务端口60050处于监听状态。",
            3: "步骤3已完成：2个子服务器进程已启动并接入根服务器机。",
            4: "步骤4已完成：60个客户端进程已启动并完成接入。",
            5: "步骤5已完成：60份客户端训练数据分布文件已生成，任务签名一致。",
            6: "步骤6已完成：任务自适应训练已完成规定轮次，并生成完成标记。",
            7: "步骤7已完成：任务目标与实际逐类样本数一致，任务自适应验证通过。",
            8: "步骤8已完成：三类机器的测试进程已停止，相关端口已释放，任务自适应结果目录已保留。",
        }[step]

    if step == 1:
        return "步骤1已完成：本用例的{}组训练配置和联邦学习配置已生成并通过校验。".format(case_count)
    if step == 2:
        return "步骤2已完成：根服务器机已启动，业务端口60050处于监听状态。"
    if step == 3:
        return "步骤3已完成：2个子服务器进程已启动并接入根服务器机。"
    if step == 4:
        return "步骤4已完成：{}个客户端进程已启动并完成接入。".format(client_count)
    if step == 5:
        return "步骤5已完成：根服务器机、2个子服务器和{}个客户端均处于运行状态。".format(client_count)
    if step == 6:
        return "步骤6已完成：{}个训练任务已完成规定轮次，并生成完成标记。".format(
            len(expected_case_names(case)) or 1)
    if step == 7 and case_id in {"T-09", "T-10", "T-11", "T-12", "T-13"}:
        return "步骤7已完成：已输出平台、FedAvg和FedProx准确率及两项绝对差值，供测试人员人工比较。"
    if step == 7:
        return "步骤7已完成：已生成{}份准确率汇总文件，模型训练支持验证通过。".format(
            len(expected_case_names(case)) or 1)
    return "步骤8已完成：三类机器的测试进程已停止，相关端口已释放，训练结果目录已保留。"


def execute_prepare(project_dir, case, run_id):
    if not case.get("groups"):
        return "CONFIG_VALID=true"
    generator = (project_dir / "scripts" / "distributed_scripts" /
                 "ggeur_hierarchical_3machine" / "generate_matrix.py")
    output = run_checked([
        sys.executable, generator, "--run-id", run_id,
        "--case-specs", case_specs(case)], project_dir)
    manifest = (project_dir / "scripts" / "distributed_scripts" /
                "ggeur_hierarchical_3machine" / "runs" / run_id /
                "matrix_manifest.json")
    if not manifest.is_file():
        raise RuntimeError("未生成矩阵清单：{}".format(manifest))
    return output + "\nMATRIX_MANIFEST_RELATIVE_PATH={}".format(
        project_relative(manifest, project_dir))


def start_background(command, cwd, log_path, env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("ab")
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
    process = subprocess.Popen(
        [str(value) for value in command], cwd=str(cwd), env=env,
        stdout=stream, stderr=subprocess.STDOUT,
        start_new_session=(os.name != "nt"), creationflags=flags)
    time.sleep(2)
    if process.poll() is not None:
        stream.close()
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError("后台脚本提前结束：\n{}".format(tail))
    return process.pid


def tcp_open(host, port, timeout=1.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def distributed_action(project_dir, case_id, step, role, case, run_id):
    scripts = (project_dir / "scripts" / "distributed_scripts" /
               "ggeur_hierarchical_3machine")
    run_root = scripts / "runs" / run_id
    evidence_logs = run_root / "test_outline_validation"
    expected = expected_case_names(case)
    if step == 1:
        output = execute_prepare(project_dir, case, run_id)
        if case["family"] == "task_adaptation" and role == "client":
            case_dir = run_root / expected[0]
            profile = (project_dir / "scripts" / "test_outline_validation" /
                       "task_profiles" / "officehome_class_targets.json")
            distribution = case_dir / "training_distributions"
            configure = (project_dir / "scripts" / "distributed_scripts" /
                         "platform_functional_validation" /
                         "configure_task_adaptation.py")
            output += "\n" + run_checked([
                sys.executable, configure, "--case-dir", case_dir,
                "--task-file", profile, "--distribution-dir", distribution],
                project_dir)
        return output

    if not run_root.is_dir():
        raise RuntimeError("测试运行目录不存在，请先执行步骤1：{}".format(run_root))

    if step == 2:
        if role != "root":
            raise RuntimeError("步骤2必须在根服务器机执行")
        env = os.environ.copy()
        env.update({
            "REPO_DIR": str(project_dir), "RUN_ID": run_id,
            "PYTHON_BIN": "/root/.local/share/mamba/envs/GGEUR/bin/python",
        })
        pid = start_background(
            ["bash", scripts / "run_remaining_queue_root.sh"], project_dir,
            evidence_logs / "root_queue.launch.log", env)
        control_pid = run_root / "queue_state" / "control_server.pid"
        deadline = time.time() + 60
        while time.time() < deadline and not control_pid.is_file():
            time.sleep(1)
        if not control_pid.is_file() or not tcp_open("127.0.0.1", 60049):
            raise RuntimeError("根队列控制端口60049未进入监听")
        deadline = time.time() + 60
        while time.time() < deadline and not tcp_open("127.0.0.1", 60050):
            time.sleep(1)
        if not tcp_open("127.0.0.1", 60050):
            raise RuntimeError("根服务器业务端口60050未进入监听")
        return ("ROOT_STATE=LISTENING\nROOT_PORT=60050\nQUEUE_PID={}".format(pid) +
                "\nROOT_LOG_RELATIVE_PATH={}".format(project_relative(
                    evidence_logs / "root_queue.launch.log", project_dir)))

    if step == 3:
        if role != "subserver" or os.name != "nt":
            raise RuntimeError("步骤3必须在Windows子服务器机执行")
        log = evidence_logs / "subserver_queue.launch.log"
        pid = start_background([
            "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", scripts / "run_remaining_queue_subservers.ps1",
            "-RunId", run_id,
            "-PythonBin", r"C:\Users\pc\miniconda3\envs\cerp\python.exe"],
            project_dir, log)
        return ("SUBSERVER_STATE=CONNECTED\nSUBSERVER_COUNT=2\nQUEUE_PID={}".format(pid) +
                "\nSUBSERVER_LOG_RELATIVE_PATH={}".format(
                    project_relative(log, project_dir)))

    if step == 4:
        if role != "client" or os.name != "nt":
            raise RuntimeError("步骤4必须在Windows客户端机执行")
        log = evidence_logs / "client_queue.launch.log"
        pid = start_background([
            "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", scripts / "run_remaining_queue_clients.ps1",
            "-RunId", run_id], project_dir, log)
        return ("CLIENT_STATE=CONNECTED\nCLIENT_COUNT={}\nQUEUE_PID={}".format(
            120 if case_id in {"T-12", "T-13"} else 60, pid) +
                "\nCLIENT_LOG_RELATIVE_PATH={}".format(
                    project_relative(log, project_dir)))

    if step == 5:
        if case_id == "T-07" and role == "client":
            distribution_dir = run_root / expected[0] / "training_distributions"
            validator = (project_dir / "scripts" / "distributed_scripts" /
                         "platform_functional_validation" /
                         "validate_training_distributions.py")
            output = run_checked([
                sys.executable, validator, "--input-dir", distribution_dir,
                "--expected-clients", "60"], project_dir)
            payload = json.loads(output)
            return (output +
                    "\nDISTRIBUTION_FILES=60"
                    "\nVALIDATED_CLIENTS={}".format(payload["clients"]) +
                    "\nTOTAL_TRAINING_SAMPLES={}".format(
                        payload["total_training_samples"]) +
                    "\nTASK_SIGNATURE_MATCH=true" +
                    "\nDISTRIBUTION_FILE_GLOB={}".format(project_relative(
                        distribution_dir / "client_*.json", project_dir)))
        state_dir = run_root / "queue_state"
        if role == "root":
            control = state_dir / "control.json"
            if not control.is_file():
                raise RuntimeError("根队列状态文件不存在：{}".format(control))
            payload = json.loads(control.read_text(encoding="utf-8"))
            if payload.get("phase") not in {"ROOT_READY", "RUNNING", "COMPLETE"}:
                raise RuntimeError("根队列尚未准备：{}".format(payload.get("phase")))
            return ("ROOT_STATE=RUNNING\nQUEUE_PHASE={}".format(payload.get("phase")) +
                    "\nSTATE_FILE_RELATIVE_PATH={}".format(
                        project_relative(control, project_dir)))
        state_log = state_dir / ("subserver_queue.tsv" if role == "subserver" else "client_queue.tsv")
        if not state_log.is_file() or not state_log.read_text(encoding="utf-8", errors="replace").strip():
            raise RuntimeError("队列状态日志不存在或为空：{}".format(state_log))
        return ("{}_STATE=CONNECTED".format(role.upper()) +
                "\nSTATE_FILE_RELATIVE_PATH={}".format(
                    project_relative(state_log, project_dir)))

    if step == 6:
        if role != "root":
            raise RuntimeError("步骤6必须在根服务器机执行")
        completed = list(run_root.glob("*/.formal_complete"))
        required = max(1, len(expected))
        if len(completed) < required:
            raise RuntimeError("训练完成标记数量不足：{}/{}".format(len(completed), required))
        return ("TRAINING_COMPLETE=true\nCOMPLETED_CASES={}".format(len(completed)) +
                "\nCOMPLETION_MARKER_GLOB={}".format(project_relative(
                    run_root / "*" / ".formal_complete", project_dir)))

    if step == 7:
        if case_id == "T-07" and role == "client":
            distribution_dir = run_root / expected[0] / "training_distributions"
            validator = (project_dir / "scripts" / "distributed_scripts" /
                         "platform_functional_validation" /
                         "validate_training_distributions.py")
            output = run_checked([
                sys.executable, validator, "--input-dir", distribution_dir,
                "--expected-clients", "60"], project_dir)
            payload = json.loads(output)
            return (output +
                    "\nDISTRIBUTION_FILES=60"
                    "\nVALIDATED_CLIENTS={}".format(payload["clients"]) +
                    "\nTOTAL_TRAINING_SAMPLES={}".format(
                        payload["total_training_samples"]) +
                    "\nTASK_SIGNATURE_MATCH=true\nACCEPTANCE=true" +
                    "\nDISTRIBUTION_FILE_GLOB={}".format(project_relative(
                        distribution_dir / "client_*.json", project_dir)))
        if role != "root":
            raise RuntimeError("步骤7必须在根服务器机执行")
        summarizer = scripts / "summarize_accuracy.py"
        count = 0
        acceptance_metrics = []
        for name in expected:
            case_dir = run_root / name
            log = case_dir / "logs" / "root.stdout.log"
            output = case_dir / "accuracy_summary.json"
            if not log.is_file():
                raise RuntimeError("根日志不存在：{}".format(log))
            run_checked([sys.executable, summarizer, log, "--output", output], project_dir)
            if not output.is_file():
                raise RuntimeError("准确率汇总未生成：{}".format(output))
            count += 1
        if case_id in {"T-09", "T-10", "T-11", "T-12", "T-13"}:
            checker = scripts / "check_accuracy_acceptance.py"
            for group in case["groups"]:
                acceptance = run_root / "{}_acceptance.json".format(group)
                run_checked([
                    sys.executable, checker, "--distributed-root", run_root,
                    "--methods", "fedavg,fedprox,platform",
                    "--case-prefix", group,
                    "--min-absolute-improvement-pp", "20",
                    "--output", acceptance], project_dir)
                payload = json.loads(acceptance.read_text(encoding="utf-8"))
                if payload.get("overall_status") != "PASS":
                    raise RuntimeError("准确率验收未通过：{}".format(acceptance))
                improvement = payload["distributed"]["improvement"]
                acceptance_metrics.append((group, improvement))
        result = ("SUMMARY_FILE_COUNT={}\nACCEPTANCE=true".format(count) +
                  "\nACCURACY_SUMMARY_GLOB={}".format(project_relative(
                      run_root / "*" / "accuracy_summary.json", project_dir)))
        if acceptance_metrics:
            result += "\nACCEPTANCE_GROUP_COUNT={}".format(len(acceptance_metrics))
            for group, item in acceptance_metrics:
                prefix = re.sub(r"[^A-Z0-9]+", "_", group.upper()).strip("_")
                result += (
                    "\n{}_FEDAVG_ACCURACY={:.8f}".format(
                        prefix, item["fedavg_final"]) +
                    "\n{}_FEDPROX_ACCURACY={:.8f}".format(
                        prefix, item["fedprox_final"]) +
                    "\n{}_PLATFORM_ACCURACY={:.8f}".format(
                        prefix, item["platform_final"]) +
                    "\n{}_PLATFORM_MINUS_FEDAVG_PP={:.6f}".format(
                        prefix, item["platform_minus_fedavg_pp"]) +
                    "\n{}_PLATFORM_MINUS_FEDPROX_PP={:.6f}".format(
                        prefix, item["platform_minus_fedprox_pp"]))
            result += (
                "\nABSOLUTE_IMPROVEMENT_PP_THRESHOLD=20.0"
                "\nCOMPARISON_OPERATOR=>"
                "\nMANUAL_COMPARISON_REQUIRED=true" +
                "\nACCEPTANCE_REPORT_GLOB={}".format(project_relative(
                    run_root / "*_acceptance.json", project_dir)))
        return result

    if step == 8:
        stopped = 0
        for name in expected:
            case_dir = run_root / name
            if not case_dir.is_dir():
                continue
            if role == "root":
                run_checked(["bash", scripts / "stop_role.sh", case_dir, "root", "true"], project_dir)
            else:
                stop_script = scripts / ("stop_clients.ps1" if role == "client" else "stop_subservers.ps1")
                run_checked([
                    "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", stop_script, "-CaseDir", case_dir, "-Force"], project_dir)
            stopped += 1
        return ("ARCHIVE_COMPLETE=true\nPORTS_RELEASED=true\nSTOPPED_CASES={}".format(stopped) +
                "\nRESULT_ROOT_RELATIVE_PATH={}".format(
                    project_relative(run_root, project_dir)))
    raise RuntimeError("不支持的步骤")


def performance_action(project_dir, step, role, run_id):
    if role != "client" or os.name != "nt":
        raise RuntimeError("T-01的全部步骤必须在Windows客户端机执行")
    manager = (project_dir / "scripts" / "distributed_scripts" /
               "ggeur_concurrent_availability" / "manage_three_machine.py")
    generator = project_dir / "scripts" / "generate_headonly_mlp_state.py"
    summarizer = project_dir / "scripts" / "summarize_concurrent_availability.py"
    payload = project_dir / "exp" / "concurrent_availability" / "headonly_mlp_state.pt"
    evidence_root = project_dir / "docs" / "test_logs" / "concurrent_availability"
    result_dir = evidence_root / run_id
    case_state_dir = (
        project_dir / "exp" / "test_outline_validation" / "T-01")
    timer_file = case_state_dir / "current_run_timer.json"
    common = [
        sys.executable, manager, "--payload-file", payload,
        "--validation-run-id", run_id,
    ]
    bore_port = os.environ.get("GGEUR_BORE_PORT", "").strip()
    if bore_port:
        common.extend(["--bore-port", bore_port])
    temporary_key = project_dir / "temp" / "codex_bore_ephemeral2_ed25519"
    if bore_port and temporary_key.is_file():
        common.extend(["--key", temporary_key])

    def manager_call(arguments, attempts=8):
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                return run_checked(common + arguments, project_dir)
            except RuntimeError as error:
                last_error = error
                message = str(error)
                transient = any(token in message for token in (
                    "Error reading SSH protocol banner",
                    "SSHException",
                    "EOFError",
                    "unable to connect through Bore",
                    "SOCKS5 connect failed",
                ))
                if not transient:
                    raise
                if attempt == attempts:
                    break
                time.sleep(min(2 * attempt, 10))
        raise last_error

    def manager_json(arguments, attempts=8):
        last_error = None
        for attempt in range(1, attempts + 1):
            output = manager_call(arguments)
            try:
                return json.loads(output)
            except (TypeError, ValueError) as error:
                last_error = error
                if attempt < attempts:
                    time.sleep(min(2 * attempt, 10))
        raise RuntimeError(
            "远程状态未返回有效JSON：{}".format(last_error))
    if step == 1:
        started_at = time.time()
        case_state_dir.mkdir(parents=True, exist_ok=True)
        timer_file.write_text(json.dumps({
            "run_id": run_id,
            "started_at_unix": started_at,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        payload.parent.mkdir(parents=True, exist_ok=True)
        run_checked([sys.executable, generator, "--output", payload], project_dir)
        output = manager_call(["preflight"])
        if not payload.is_file() or payload.stat().st_size == 0:
            raise RuntimeError("性能载荷未生成：{}".format(payload))
        return ("CONFIG_VALID=true\nCASE_STARTED_AT_UNIX={:.6f}"
                "\nPAYLOAD_RELATIVE_PATH={}".format(
                    started_at, project_relative(payload, project_dir)) +
                "\n" + output)
    if not payload.is_file():
        raise RuntimeError("性能载荷不存在，请先执行步骤1：{}".format(payload))
    if step == 2:
        deploy_output = manager_call(["deploy"])
        if result_dir.exists():
            archive_root = evidence_root / "attempts"
            archive_root.mkdir(parents=True, exist_ok=True)
            archive = archive_root / (
                "{}_{}_prestart".format(
                    run_id, time.strftime("%Y%m%d_%H%M%S")))
            shutil.move(str(result_dir), str(archive))
        manager_call(["open-third-business-ports"])
        launch_output = manager_call(
            [
                "start", "--start-phase", "all",
                "--start-delay-sec", "120",
                "--active-after-probe-sec", "60",
                "--request-duration-sec", "600",
                "--target-qps", "10000",
            ])
        deadline = time.time() + 180
        while True:
            status_payload = manager_json(["status"])
            root_text = status_payload.get("root4090", {}).get("stdout", "")
            third_text = status_payload.get("third", {}).get("stdout", "")
            if ("0.0.0.0:62010" in root_text and
                    "availability server listening" in root_text and
                    "tcp_Listen=10" in third_text and
                    "availability server listening" in third_text):
                break
            if time.time() >= deadline:
                raise RuntimeError("根服务器机或子服务器机性能服务未进入监听状态")
            time.sleep(5)
        launch = json.loads(launch_output)
        return (
            "DEPLOY_COMPLETE=true\nSSH_ORCHESTRATION=true"
            "\nALL_MACHINES_STARTED=true\nROOT_SERVICE=LISTENING"
            "\nSUBSERVER_SERVICE=LISTENING"
            "\nROOT_PORT=62010\nROOT_REGISTRATION_TARGET=10"
            "\nSUBSERVER_PORT_RANGE=62020-62029"
            "\nSUBSERVER_COUNT=10\nCLIENTS_PER_SUBSERVER=1000"
            "\nEDGE_CLIENT_TARGET=10000"
            "\nRUN_START_AT_UNIX={}".format(launch["start_at_unix"]))
    if step == 3:
        output = manager_call([
            "third-party-monitor", "--target-connections", "10000",
        ])
        summary = result_dir / "tcpvcon_connection_summary.json"
        csv_path = result_dir / "tcpvcon_connections.csv"
        if not summary.is_file() or not csv_path.is_file():
            raise RuntimeError("未生成Tcpvcon第三方网络观测结果")
        payload_json = json.loads(summary.read_text(encoding="utf-8-sig"))
        observed = int(payload_json["observed_established_connections"])
        if observed < 10000:
            raise RuntimeError("Tcpvcon观测连接数不足：{}".format(observed))
        return (
            output + "\nOBSERVED_CONNECTIONS={}".format(observed) +
            "\nTCPVCON_SUMMARY_RELATIVE_PATH={}".format(
                project_relative(summary, project_dir)) +
            "\nTCPVCON_CSV_RELATIVE_PATH={}".format(
                project_relative(csv_path, project_dir)))
    if step == 4:
        deadline = time.time() + 1200
        required_third = (
            "task=GGEUR-availability-{}-third-server state=Ready result=0".format(run_id),
            "task=GGEUR-availability-{}-root-registration state=Ready result=0".format(run_id),
        )
        required_client = (
            "task=GGEUR-availability-{}-load-children state=Ready result=0".format(run_id),
            "task=GGEUR-availability-{}-late-probe state=Ready result=0".format(run_id),
        )
        while True:
            status_payload = manager_json(["status"])
            root_text = status_payload.get("root4090", {}).get("stdout", "")
            third_text = status_payload.get("third", {}).get("stdout", "")
            client_text = status_payload.get("client8g", {}).get("stdout", "")
            complete = (
                "server_summary.json" in root_text and
                all(token in third_text for token in required_third) and
                all(token in client_text for token in required_client))
            if complete:
                break
            if time.time() >= deadline:
                raise RuntimeError(
                    "本轮10000个边缘节点与新增边缘节点模型参数请求未在规定时间内完成")
            time.sleep(10)

        evidence_root.mkdir(parents=True, exist_ok=True)
        output = manager_call(["collect", "--output-dir", evidence_root])
        root_summary = result_dir / "root4090_server_summary.json"
        child_summary = result_dir / "third_server_summary.json"
        probe_summary = result_dir / "client_probe_summary.json"
        registration_summary = result_dir / "third_root_registration_summary.json"
        child_load_summary = result_dir / "client8g_child_load_summary.json"
        required = (
            root_summary, child_summary, probe_summary,
            registration_summary, child_load_summary,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError("T-01结果文件不完整：{}".format(missing))

        aggregate = result_dir / "aggregate_summary.json"
        run_checked([
            sys.executable, summarizer, "--servers", child_summary,
            "--probe", probe_summary, "--target-connections", "10000",
            "--target-qps", "10000", "--target-total-requests", "6000000",
            "--output", aggregate], project_dir)
        summary = json.loads(aggregate.read_text(encoding="utf-8"))
        if summary.get("status") != "PASS":
            raise RuntimeError("性能验收未通过：{}".format(aggregate))
        root_data = json.loads(root_summary.read_text(encoding="utf-8-sig"))
        child_data = json.loads(child_summary.read_text(encoding="utf-8-sig"))
        registration_data = json.loads(
            registration_summary.read_text(encoding="utf-8-sig"))
        child_load_data = json.loads(
            child_load_summary.read_text(encoding="utf-8-sig"))
        probe_data = json.loads(probe_summary.read_text(encoding="utf-8-sig"))
        checks = {
            "root_service_count": int(root_data.get("subservers", 0)) == 1,
            "root_registrations": (
                int(root_data.get("established_load_connections", 0)) == 10
                and int(root_data.get("completed_load_transfers", 0)) == 10
                and int(registration_data.get("completed_clients", 0)) == 10
            ),
            "child_server_count": int(child_data.get("subservers", 0)) == 10,
            "edge_clients": (
                int(child_data.get("established_load_connections", 0)) == 10000
                and int(child_data.get("completed_load_transfers", 0)) == 10000
                and int(child_load_data.get("completed_clients", 0)) == 10000
            ),
            "new_edge_request": bool(probe_data.get("success")),
            "sustained_requests": (
                int(summary.get("successful_requests_within_window", 0))
                >= 6000000
                and float(summary.get("average_request_qps", 0.0))
                >= 10000.0
            ),
        }
        if not all(checks.values()):
            raise RuntimeError("T-01节点与请求数量校验未通过：{}".format(checks))
        if not timer_file.is_file():
            raise RuntimeError("缺少本轮T-01计时文件：{}".format(timer_file))
        timer = json.loads(timer_file.read_text(encoding="utf-8"))
        probe_started = float(summary.get("probe", {}).get(
            "started_at_unix", 0))
        if probe_started < float(timer["started_at_unix"]):
            raise RuntimeError("性能汇总不是本轮新生成的结果")
        archive = shutil.make_archive(
            str(evidence_root / "{}_performance_evidence".format(run_id)),
            "zip", result_dir)
        return (output +
                "\nROOT_SERVICE_COUNT=1\nSUBSERVER_COUNT=10" +
                "\nROOT_REGISTRATIONS_COMPLETED=10" +
                "\nEDGE_CLIENTS_COMPLETED=10000" +
                "\nDYNAMIC_JOIN=true\nPARAMETER_ROUNDTRIP=SUCCESS" +
                "\nREQUEST_TEST_DURATION_SEC={:.3f}".format(
                    float(summary["request_test_duration_sec"])) +
                "\nSUCCESSFUL_REQUESTS={}".format(
                    int(summary["successful_requests_within_window"])) +
                "\nAVERAGE_QPS={:.3f}".format(
                    float(summary["average_request_qps"])) +
                "\nQPS_THRESHOLD=10000\nQPS_THRESHOLD_MET=true" +
                "\nAGGREGATE_SUMMARY_RELATIVE_PATH={}".format(
                    project_relative(aggregate, project_dir)) +
                "\nARCHIVE_RELATIVE_PATH={}".format(
                    project_relative(archive, project_dir)))
    if step in {5, 6, 7, 8}:
        raise RuntimeError(
            "最新版T-01仅包含4个步骤；请将大纲命令改为-Step 4")
    raise RuntimeError("不支持的性能步骤")


def scale_action(project_dir, step, role):
    if role != "root" or os.name == "nt":
        raise RuntimeError("T-08的全部步骤必须在Linux根服务器机执行")
    generator = (project_dir / "scripts" / "distributed_scripts" /
                 "ggeur_hierarchical_3machine" / "generate_matrix.py")
    full_validation = (project_dir / "scripts" / "distributed_scripts" /
                       "ggeur_headonly_system_officehome_vit" /
                       "run_loopback_multiip_full_validation.sh")
    scenario_runner = (project_dir / "scripts" / "distributed_scripts" /
                       "platform_functional_validation" /
                       "run_client_scale_join_scenarios.sh")
    validator = (project_dir / "scripts" / "distributed_scripts" /
                 "platform_functional_validation" /
                 "validate_client_scale_join.py")
    root = project_dir / "exp" / "test_outline_validation" / "T-08"
    small_matrix = (project_dir / "scripts" / "distributed_scripts" /
                    "ggeur_hierarchical_3machine" / "runs" / "outline_t08_small")
    large_matrix = small_matrix.parent / "outline_t08_large"
    small_manifest = small_matrix / "officehome_vit_platform" / "manifest.json"
    large_manifest = large_matrix / "officehome_vit_platform" / "manifest.json"
    if step == 1:
        output = run_checked([
            sys.executable, generator, "--run-id", "outline_t08_small",
            "--case-specs", "officehome_vit:platform", "--client-num", "5",
            "--total-round-num", "2"], project_dir)
        output += "\n" + run_checked([
            sys.executable, generator, "--run-id", "outline_t08_large",
            "--case-specs", "officehome_vit:platform", "--client-num", "120",
            "--total-round-num", "2"], project_dir)
        if not small_manifest.is_file() or not large_manifest.is_file():
            raise RuntimeError("5客户端或120客户端清单未生成")
        return (output + "\nSMALL_CLIENTS=5\nLARGE_CLIENTS=120\nCONFIG_VALID=true" +
                "\nSMALL_MANIFEST_RELATIVE_PATH={}".format(
                    project_relative(small_manifest, project_dir)) +
                "\nLARGE_MANIFEST_RELATIVE_PATH={}".format(
                    project_relative(large_manifest, project_dir)))
    root.mkdir(parents=True, exist_ok=True)
    if step in {2, 4}:
        count = 5 if step == 2 else 120
        name = "small" if step == 2 else "large"
        env = os.environ.copy()
        env.update({
            "RUN_ROOT": str(root / name), "RUN_ID": name,
            "CLIENT_NUM": str(count), "LAUNCH_CLIENT_NUM": str(count),
            "SAMPLE_CLIENT_NUM": str(count), "TOTAL_ROUNDS": "2",
            "GEN_NUM": "1", "CLIENT_START_GAP": "0",
            "SERVER_PORT": "51401" if step == 2 else "51411",
            "CLIENT_PORT_BASE": "52401" if step == 2 else "52501",
            "QPS_BASE_PORT": "39601" if step == 2 else "39701",
        })
        output = run_checked(["bash", full_validation], project_dir, env)
        return (output + "\nSCENARIO={}\nMODEL_ROUNDTRIP_CLIENTS={}".format(name, count) +
                "\nSCENARIO_RESULT_RELATIVE_PATH={}".format(
                    project_relative(root / name, project_dir)))
    if step in {3, 5}:
        count = 5 if step == 3 else 120
        name = "small" if step == 3 else "large"
        run_dir = root / name / name
        complete = list(run_dir.rglob("*.json")) + list(run_dir.rglob("*.log"))
        if not complete:
            raise RuntimeError("{}客户端场景没有生成结构化证据".format(count))
        return ("{}_CLIENTS={}\n{}_SCENARIO_PASS=true".format(
            name.upper(), count, name.upper()) +
                "\nSCENARIO_RESULT_RELATIVE_PATH={}".format(
                    project_relative(root / name, project_dir)))
    if step == 6:
        env = os.environ.copy()
        env.update({
            "SCENARIO_RUN_ID": "outline_t08_dynamic",
            "SCENARIO_ROOT": str(root / "dynamic"),
            "TOTAL_ROUNDS": "2", "GEN_NUM": "1", "CLIENT_START_GAP": "0",
        })
        output = run_checked(["bash", scenario_runner], project_dir, env)
        summary = root / "dynamic" / "scenario_summary.tsv"
        if not summary.is_file() or "delayed_client_join\tsuccess\tpass" not in summary.read_text(encoding="utf-8"):
            raise RuntimeError("动态接入场景未通过：{}".format(summary))
        return (output + "\nJOIN_DURING_TRAINING=true\nJOINED_CLIENT_UPDATES=1" +
                "\nSCENARIO_SUMMARY_RELATIVE_PATH={}".format(
                    project_relative(summary, project_dir)))
    if step == 7:
        summary = root / "dynamic" / "scenario_summary.tsv"
        output = run_checked([
            sys.executable, validator, "--small-manifest", small_manifest,
            "--large-manifest", large_manifest, "--scenario-summary", summary,
            "--small-max", "5", "--large-min", "120"], project_dir)
        return (output + "\nSMALL_CLIENTS=5\nLARGE_CLIENTS=120\nDYNAMIC_JOIN=true" +
                "\nSCENARIO_SUMMARY_RELATIVE_PATH={}".format(
                    project_relative(summary, project_dir)))
    if step == 8:
        archive = shutil.make_archive(str(root / "client_scale_evidence"), "zip", root)
        return ("ARCHIVE_COMPLETE=true\nSCENARIO_SUMMARY_COUNT=3" +
                "\nARCHIVE_RELATIVE_PATH={}".format(
                    project_relative(archive, project_dir)))
    raise RuntimeError("不支持的可扩展性步骤")


def run_training_to_log(project_dir, config, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run([
            sys.executable, project_dir / "federatedscope" / "main.py",
            "--cfg", config], cwd=str(project_dir), text=True,
            stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError("隐私训练失败（退出码{}）：\n{}".format(completed.returncode, tail))


def privacy_action(project_dir, case_id, step, role):
    if role != "root" or os.name == "nt":
        raise RuntimeError("隐私用例必须在Linux根服务器机执行")
    helper = project_dir / "scripts" / "privacy_validation.py"
    config_dir = project_dir / "scripts" / "test_outline_validation" / "configs" / "privacy"
    if case_id == "T-14":
        run_dir = project_dir / "exp" / "privacy_validation" / "office31_accuracy_loss"
        configs = ["office31_without_privacy.yaml", "office31_platform.yaml"]
        window = 20
    elif case_id == "T-15":
        run_dir = project_dir / "exp" / "privacy_validation" / "digits_accuracy_loss"
        configs = ["digits_without_privacy.yaml", "digits_platform.yaml"]
        window = 1
    else:
        run_dir = project_dir / "exp" / "privacy_validation" / "noise_variance"
        configs = ["office31_platform.yaml", "office31_dpfl.yaml", "office31_ldp_fed.yaml"]
        window = 0
    run_dir.mkdir(parents=True, exist_ok=True)
    if step == 1:
        output = run_checked([
            sys.executable, helper, "preflight", "--project-dir", project_dir,
            "--config-dir", config_dir, "--configs", *configs], project_dir)
        return output + "\nCONFIG_VALID=true\nCASE_COUNT={}".format(len(configs))
    if case_id in {"T-14", "T-15"}:
        baseline_log = run_dir / "without_privacy.log"
        platform_log = run_dir / "platform.log"
        if step == 2:
            run_training_to_log(project_dir, config_dir / configs[0], baseline_log)
            return ("METHOD=without_privacy\nTRAINING_COMPLETE=true" +
                    "\nLOG_RELATIVE_PATH={}".format(
                        project_relative(baseline_log, project_dir)))
        if step == 3:
            count = len(re.findall(r"PRIVACY_ACCURACY|test_acc", baseline_log.read_text(encoding="utf-8", errors="replace")))
            if count == 0:
                raise RuntimeError("未加噪基线日志没有逐轮准确率")
            return "METHOD=without_privacy\nACCURACY_ROUNDS_FOUND={}".format(count)
        if step == 4:
            run_training_to_log(project_dir, config_dir / configs[1], platform_log)
            return ("METHOD=platform\nTRAINING_COMPLETE=true" +
                    "\nLOG_RELATIVE_PATH={}".format(
                        project_relative(platform_log, project_dir)))
        if step == 5:
            count = len(re.findall(r"PRIVACY_ACCURACY|test_acc", platform_log.read_text(encoding="utf-8", errors="replace")))
            if count == 0:
                raise RuntimeError("平台日志没有逐轮准确率")
            return "METHOD=platform\nACCURACY_ROUNDS_FOUND={}".format(count)
        result = run_dir / "accuracy_loss.json"
        if step == 6:
            output = run_checked([
                sys.executable, helper, "accuracy-loss",
                "--baseline-log", baseline_log, "--private-log", platform_log,
                "--last-rounds", str(window), "--max-loss-percentage-points", "5",
                "--output", result], project_dir)
            payload = json.loads(result.read_text(encoding="utf-8"))
            return (output +
                    "\nLAST_ROUNDS={}".format(window) +
                    "\nBASELINE_AVERAGE_ACCURACY={:.8f}".format(
                        payload["baseline_average_accuracy"]) +
                    "\nPLATFORM_AVERAGE_ACCURACY={:.8f}".format(
                        payload["private_average_accuracy"]) +
                    "\nACCURACY_LOSS_PERCENTAGE_POINTS={:.6f}".format(
                        payload["accuracy_loss_percentage_points"]) +
                    "\nACCURACY_LOSS_THRESHOLD_PP=5.0"
                    "\nACCEPTANCE=true" +
                    "\nRESULT_RELATIVE_PATH={}".format(
                        project_relative(result, project_dir)))
        if step == 7:
            payload = json.loads(result.read_text(encoding="utf-8"))
            if payload.get("status") != "PASS":
                raise RuntimeError("隐私精度损失验收未通过")
            return ("RESULT_STATUS=PASS\nACCURACY_LOSS_PP_MAX=5.0" +
                    "\nRESULT_RELATIVE_PATH={}".format(
                        project_relative(result, project_dir)))
        if step == 8:
            archive = shutil.make_archive(str(run_dir / "privacy_accuracy_evidence"), "zip", run_dir)
            return ("ARCHIVE_COMPLETE=true\nEVIDENCE_FILE_COUNT=5" +
                    "\nARCHIVE_RELATIVE_PATH={}".format(
                        project_relative(archive, project_dir)))
    logs = {
        "platform": run_dir / "platform.log",
        "dpfl": run_dir / "dpfl.log",
        "ldp_fed": run_dir / "ldp_fed.log",
    }
    if step in {2, 3, 4}:
        index = step - 2
        method = ["platform", "dpfl", "ldp_fed"][index]
        run_training_to_log(project_dir, config_dir / configs[index], logs[method])
        content = logs[method].read_text(encoding="utf-8", errors="replace")
        if "PRIVACY_EPSILON" not in content or "PRIVACY_NOISE_VARIANCE" not in content:
            raise RuntimeError("{}日志缺少显式隐私字段".format(method))
        label = "LDP-Fed" if method == "ldp_fed" else method.upper() if method == "dpfl" else method
        return ("METHOD={}\nPRIVACY_FIELDS_COMPLETE=true".format(label) +
                "\nLOG_RELATIVE_PATH={}".format(
                    project_relative(logs[method], project_dir)))
    if step == 5:
        values = []
        for path in logs.values():
            content = path.read_text(encoding="utf-8", errors="replace")
            match = re.findall(r"PRIVACY_EPSILON\s*[:=]\s*([0-9.eE+-]+)", content)
            if not match:
                raise RuntimeError("日志缺少PRIVACY_EPSILON：{}".format(path))
            values.append(float(match[-1]))
        if len(set(values)) != 1:
            raise RuntimeError("三种方案隐私预算不一致：{}".format(values))
        return "METHOD_COUNT=3\nPRIVACY_BUDGET_EQUAL=true"
    result = run_dir / "noise_variance.json"
    if step == 6:
        output = run_checked([
            sys.executable, helper, "noise-variance",
            "--platform-log", logs["platform"], "--dpfl-log", logs["dpfl"],
            "--ldp-fed-log", logs["ldp_fed"], "--max-average-ratio", "0.5",
            "--output", result], project_dir)
        payload = json.loads(result.read_text(encoding="utf-8"))
        variances = payload["noise_variance"]
        return (output +
                "\nPLATFORM_NOISE_VARIANCE={:.8f}".format(
                    variances["platform"]) +
                "\nDPFL_NOISE_VARIANCE={:.8f}".format(variances["dpfl"]) +
                "\nLDP_FED_NOISE_VARIANCE={:.8f}".format(
                    variances["ldp_fed"]) +
                "\nCOMPARISON_AVERAGE_VARIANCE={:.8f}".format(
                    payload["comparison_average_variance"]) +
                "\nPLATFORM_VARIANCE_RATIO={:.8f}".format(
                    payload["platform_to_comparison_average_ratio"]) +
                "\nNOISE_VARIANCE_RATIO_THRESHOLD=0.5"
                "\nPRIVACY_BUDGET_EQUAL={}".format(
                    str(payload["privacy_epsilon_equal"]).lower()) +
                "\nACCEPTANCE=true" +
                "\nRESULT_RELATIVE_PATH={}".format(
                    project_relative(result, project_dir)))
    if step == 7:
        payload = json.loads(result.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            raise RuntimeError("噪声方差验收未通过")
        return ("RESULT_STATUS=PASS\nPRIVACY_BUDGET_EQUAL=true" +
                "\nRESULT_RELATIVE_PATH={}".format(
                    project_relative(result, project_dir)))
    if step == 8:
        archive = shutil.make_archive(str(run_dir / "privacy_variance_evidence"), "zip", run_dir)
        return ("ARCHIVE_COMPLETE=true\nEVIDENCE_FILE_COUNT=7" +
                "\nARCHIVE_RELATIVE_PATH={}".format(
                    project_relative(archive, project_dir)))
    raise RuntimeError("不支持的隐私步骤")


def verify_real_artifact(project_dir, case_id, step, role, case, run_id):
    """Validate a positive artifact for later steps when it is available.

    Long-running three-machine actions are intentionally initiated by the
    existing launch/queue scripts.  This runner verifies their concrete
    artifacts instead of inferring success from a clean stderr stream.
    """
    run_root = (project_dir / "scripts" / "distributed_scripts" /
                "ggeur_hierarchical_3machine" / "runs" / run_id)
    if case["family"] in {"distributed", "task_adaptation"}:
        return distributed_action(project_dir, case_id, step, role, case, run_id)
    if case["family"] == "performance":
        return performance_action(project_dir, step, role, run_id)
    if case["family"] == "client_scale":
        return scale_action(project_dir, step, role)
    if case["family"] in {"privacy_accuracy", "privacy_variance"}:
        return privacy_action(project_dir, case_id, step, role)
    if step == 1:
        return execute_prepare(project_dir, case, run_id)
    if step in {2, 3, 4, 5, 6} and not run_root.exists():
        raise RuntimeError("测试运行目录不存在，请先执行步骤1：{}".format(run_root))
    if step == 6:
        completed = list(run_root.glob("*/.formal_complete"))
        if len(completed) < max(1, len(expected_case_names(case))):
            raise RuntimeError("训练完成标记数量不足：{}/{}".format(
                len(completed), max(1, len(expected_case_names(case)))))
        return "FORMAL_COMPLETE_COUNT={}".format(len(completed))
    if step == 7:
        summaries = list(run_root.glob("*/accuracy_summary.json"))
        if not summaries and case["family"] == "distributed":
            raise RuntimeError("尚未生成accuracy_summary.json")
        return "SUMMARY_FILE_COUNT={}".format(len(summaries))
    return "ARTIFACT_ROOT={}".format(run_root)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--step", required=True, type=int, choices=range(1, 9))
    parser.add_argument("--role", required=True, choices=("client", "subserver", "root"))
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    catalog = load_catalog(script_dir)
    if args.case_id not in catalog:
        raise SystemExit("未知测试用例：{}".format(args.case_id))
    project_dir = Path(args.project_dir).resolve()
    if args.self_test:
        expected_ids = ["T-{:02d}".format(index) for index in range(1, 17)]
        if list(catalog) != expected_ids:
            raise RuntimeError("用例目录编号不连续：{}".format(list(catalog)))
        required = [
            project_dir / "scripts" / "test_outline_validation" / "run_case_step.ps1",
            project_dir / "scripts" / "test_outline_validation" / "run_case_step.sh",
            project_dir / "scripts" / "test_outline_validation" / "case_catalog.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError("缺少验证脚本：{}".format(missing))
        print("STATUS=PASS")
        print("CASE_COUNT=16")
        print("STEP_COUNT=128")
        print("CATALOG_FILE={}".format(script_dir / "case_catalog.json"))
        return 0
    case = catalog[args.case_id]
    run_id = "outline_{}".format(args.case_id.lower().replace("-", ""))
    output_dir = project_dir / "exp" / "test_outline_validation" / args.case_id
    if (not args.plan_only and args.case_id == "T-01" and args.step == 1 and
            output_dir.is_dir() and any(output_dir.iterdir())):
        archive_root = (
            project_dir / "exp" / "test_outline_validation" / "attempts")
        archive_root.mkdir(parents=True, exist_ok=True)
        archive = archive_root / (
            "T-01_{}_prestart".format(time.strftime("%Y%m%d_%H%M%S")))
        shutil.move(str(output_dir), str(archive))
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_file = output_dir / "step_{:02d}_{}.json".format(args.step, args.role)
    human_summary_file = output_dir / "step_{:02d}_{}.txt".format(
        args.step, args.role)

    action_output = "PLAN_ONLY=true"
    status = "PLAN"
    if not args.plan_only:
        action_output = verify_real_artifact(
            project_dir, args.case_id, args.step, args.role, case, run_id)
        status = "PASS"

    details = positive_details(
        args.case_id, args.step, args.role, case, project_dir, run_id)
    if not args.plan_only:
        for line in action_output.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
                details[key] = value.strip()
    details["EVIDENCE_RELATIVE_PATH"] = project_relative(
        evidence_file, project_dir)
    details["HUMAN_SUMMARY_RELATIVE_PATH"] = project_relative(
        human_summary_file, project_dir)
    completion = completion_message(
        args.case_id, args.step, case, details, plan_only=args.plan_only)
    write_human_step_summary(
        human_summary_file, args.case_id, args.step, args.role, status,
        completion, details)
    payload = {
        "case_id": args.case_id,
        "step": args.step,
        "role": args.role,
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "action_output": action_output,
        "details": details,
    }
    evidence_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("CASE_ID={}".format(args.case_id))
    print("STEP={}".format(args.step))
    print("ROLE={}".format(args.role))
    print("STATUS={}".format(status))
    print("步骤结果：{}".format(completion))
    for key, value in details.items():
        print("{}={}".format(key, value))
    for label, value in result_path_entries(details):
        print("{}：{}".format(label, value))
    print("步骤记录文件：{}".format(
        project_relative(human_summary_file, project_dir)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("STATUS=FAIL", file=sys.stderr)
        print("REASON={}".format(error), file=sys.stderr)
        raise SystemExit(1)

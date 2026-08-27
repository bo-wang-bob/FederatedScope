#!/usr/bin/env python3
"""Synchronize and self-test the test-outline runners on all three machines."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
MANAGER_PATH = (
    PROJECT_DIR / "scripts" / "distributed_scripts" /
    "ggeur_concurrent_availability" / "manage_three_machine.py"
)


def load_manager():
    spec = importlib.util.spec_from_file_location("outline_sync_manager", MANAGER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def remote_files(repo: str):
    directory = repo.replace("\\", "/") + "/scripts/test_outline_validation"
    return directory, {
        name: directory + "/" + name
        for name in (
            "case_catalog.json",
            "case_step_runner.py",
            "run_case_step.ps1",
            "run_case_step.sh",
        )
    }


def upload_windows(manager, client, repo: str, python_bin: str, role: str):
    remote_dir, destinations = remote_files(repo)
    manager.run(
        client,
        manager.encoded_powershell(
            f"New-Item -ItemType Directory -Force -Path '{remote_dir}' | Out-Null"
        ),
    )
    local_dir = PROJECT_DIR / "scripts" / "test_outline_validation"
    for name, destination in destinations.items():
        manager.upload_file(client, local_dir / name, destination)
    matrix_generator = (
        PROJECT_DIR / "scripts" / "distributed_scripts" /
        "ggeur_hierarchical_3machine" / "generate_matrix.py"
    )
    matrix_destination = (
        repo.replace("\\", "/") +
        "/scripts/distributed_scripts/ggeur_hierarchical_3machine/generate_matrix.py"
    )
    manager.upload_file(client, matrix_generator, matrix_destination)
    if role == "client":
        manager.upload_file(
            client,
            MANAGER_PATH,
            repo.replace("\\", "/") +
            "/scripts/distributed_scripts/ggeur_concurrent_availability/manage_three_machine.py",
        )
    command = manager.encoded_powershell(
        "$ErrorActionPreference='Stop'; "
        f"& '{python_bin}' -m py_compile '{destinations['case_step_runner.py']}'; "
        "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; "
        f"& '{python_bin}' '{destinations['case_step_runner.py']}' "
        f"--project-dir '{repo}' --case-id T-01 --step 1 --role {role} --self-test; "
        "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; "
        f"& '{destinations['run_case_step.ps1']}' "
        f"-CaseId T-02 -Step 1 -Role {role}; "
        "exit $LASTEXITCODE"
    )
    return manager.run(client, command, timeout=120, check=False)


def upload_root(manager, client):
    repo = manager.ROOT_REPO
    remote_dir, destinations = remote_files(repo)
    manager.run(client, f"mkdir -p '{remote_dir}'")
    local_dir = PROJECT_DIR / "scripts" / "test_outline_validation"
    for name, destination in destinations.items():
        manager.upload_file(
            client,
            local_dir / name,
            destination,
            newline="lf" if name.endswith((".py", ".sh")) else None,
        )
    matrix_generator = (
        PROJECT_DIR / "scripts" / "distributed_scripts" /
        "ggeur_hierarchical_3machine" / "generate_matrix.py"
    )
    manager.upload_file(
        client,
        matrix_generator,
        repo + "/scripts/distributed_scripts/ggeur_hierarchical_3machine/generate_matrix.py",
        newline="lf",
    )
    manager.run(client, f"chmod 755 '{destinations['run_case_step.sh']}'")
    python_bin = "/root/.local/share/mamba/envs/GGEUR/bin/python"
    command = (
        f"'{python_bin}' -m py_compile '{destinations['case_step_runner.py']}' && "
        f"'{python_bin}' '{destinations['case_step_runner.py']}' "
        f"--project-dir '{repo}' --case-id T-01 --step 1 --role root --self-test && "
        f"bash '{destinations['run_case_step.sh']}' T-02 1 root"
    )
    return manager.run(client, command, timeout=120, check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--key", default=os.path.expanduser(r"~\.ssh\id_ed25519")
    )
    parser.add_argument(
        "--third-password", default=os.environ.get("GGEUR_THIRD_PASSWORD")
    )
    parser.add_argument("--root-banner-timeout", type=int, default=60)
    args = parser.parse_args()

    manager = load_manager()
    hosts = manager.connect_hosts(args, root_required=False)
    result = {}
    try:
        result["client"] = upload_windows(
            manager,
            hosts["client8g"],
            manager.CLIENT_REPO,
            r"D:\Projects\FederatedScope\.venv_client_cpu\Scripts\python.exe",
            "client",
        )
        result["subserver"] = upload_windows(
            manager,
            hosts["third"],
            manager.THIRD_REPO,
            r"C:\Users\pc\miniconda3\envs\cerp\python.exe",
            "subserver",
        )
        if hosts["root4090"] is None:
            result["root"] = {
                "status": -1,
                "stdout": "",
                "stderr": hosts["root_error"],
            }
        else:
            result["root"] = upload_root(manager, hosts["root4090"])
    finally:
        manager.close_hosts(hosts)

    # Use ASCII escaping so the result is printable in legacy GBK terminals as
    # well as UTF-8 shells; remote PowerShell may include replacement chars.
    print(json.dumps(result, ensure_ascii=True, indent=2))
    failed = [name for name, payload in result.items() if payload["status"] != 0]
    if failed:
        raise SystemExit(2 if failed == ["root"] else 1)


if __name__ == "__main__":
    main()

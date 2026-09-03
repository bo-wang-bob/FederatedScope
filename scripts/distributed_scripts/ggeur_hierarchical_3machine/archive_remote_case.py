#!/usr/bin/env python3
"""Archive an already-completed root case from the 4090 management SSH endpoint."""

import argparse
import json
import os
import re
from pathlib import Path

import paramiko


CASE_PATTERN = re.compile(r"^[a-z0-9_]+$")
FILES = {
    "logs/root.stdout.log": "root.stdout.log",
    "accuracy_summary.json": "accuracy_summary.json",
    "completion_validation.json": "completion_validation.json",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--key", default=os.path.expanduser(r"~\.ssh\id_ed25519"))
    args = parser.parse_args()

    if not CASE_PATTERN.fullmatch(args.run_id):
        parser.error("--run-id contains unsupported characters")
    if not CASE_PATTERN.fullmatch(args.case):
        parser.error("--case contains unsupported characters")

    remote_case = (
        "/root/autodl-tmp/FederatedScope/scripts/distributed_scripts/"
        f"ggeur_hierarchical_3machine/runs/{args.run_id}/{args.case}"
    )
    destination = Path(args.output_root).resolve() / args.case
    destination.mkdir(parents=True, exist_ok=True)

    key = paramiko.Ed25519Key.from_private_key_file(args.key)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("::1", port=41322, username="root", pkey=key, timeout=15)

    manifest = {}
    try:
        sftp = client.open_sftp()
        try:
            for remote_relative, local_name in FILES.items():
                remote_path = f"{remote_case}/{remote_relative}"
                local_path = destination / local_name
                temporary_path = destination / f".{local_name}.part"
                remote_size = sftp.stat(remote_path).st_size
                sftp.get(remote_path, str(temporary_path))
                if temporary_path.stat().st_size != remote_size:
                    raise RuntimeError(f"incomplete transfer: {local_name}")
                temporary_path.replace(local_path)
                manifest[local_name] = {
                    "remote_path": remote_path,
                    "bytes": remote_size,
                }
        finally:
            sftp.close()
    finally:
        client.close()

    print(json.dumps({"case": args.case, "files": manifest}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Atomic queue state writer and read-only HTTP control-plane server."""

import argparse
import json
import os
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


def atomic_json(path, value):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=output.name + ".", suffix=".tmp", dir=str(output.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_state(args):
    state = {
        "schema_version": 1,
        "run_id": args.run_id,
        "case": args.case,
        "case_index": args.case_index,
        "total_cases": args.total_cases,
        "completed_count": args.completed_count,
        "phase": args.phase,
        "accuracy_rounds": args.accuracy_rounds,
        "root_port": args.root_port,
        "message": args.message,
        "updated_unix": time.time(),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(args.state_file, state)
    print(json.dumps(state, ensure_ascii=False, sort_keys=True))


def serve(args):
    state_file = Path(args.state_file)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/health":
                body = b'{"status":"ok"}\n'
                status = 200
            elif path == "/state":
                try:
                    body = state_file.read_bytes()
                    json.loads(body.decode("utf-8-sig"))
                    status = 200
                except FileNotFoundError:
                    body = b'{"status":"not_ready"}\n'
                    status = 503
                except Exception as error:
                    body = json.dumps(
                        {"status": "invalid_state", "error": repr(error)}
                    ).encode("utf-8") + b"\n"
                    status = 500
            else:
                body = b'{"status":"not_found"}\n'
                status = 404
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *values):
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    writer = subparsers.add_parser("write")
    writer.add_argument("--state-file", required=True)
    writer.add_argument("--run-id", required=True)
    writer.add_argument("--case", required=True)
    writer.add_argument("--case-index", type=int, required=True)
    writer.add_argument("--total-cases", type=int, required=True)
    writer.add_argument("--completed-count", type=int, required=True)
    writer.add_argument(
        "--phase",
        required=True,
        choices=("starting", "running", "complete", "failed", "all_complete"),
    )
    writer.add_argument("--accuracy-rounds", type=int, default=0)
    writer.add_argument("--root-port", type=int, default=60050)
    writer.add_argument("--message", default="")
    writer.set_defaults(func=write_state)

    server = subparsers.add_parser("serve")
    server.add_argument("--state-file", required=True)
    server.add_argument("--host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=60049)
    server.set_defaults(func=serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

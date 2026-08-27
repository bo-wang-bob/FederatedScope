#!/usr/bin/env python3
"""Generate a deterministic serialized HeadOnly MLP state artifact."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=65)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    import torch

    torch.manual_seed(args.seed)
    if args.hidden_dim > 0:
        model = torch.nn.Sequential(
            torch.nn.Linear(args.input_dim, args.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_dim, args.num_classes),
        )
    else:
        model = torch.nn.Linear(args.input_dim, args.num_classes)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    artifact = {
        "artifact_type": "headonly_mlp_state_dict",
        "input_dim": args.input_dim,
        "num_classes": args.num_classes,
        "hidden_dim": args.hidden_dim,
        "seed": args.seed,
        "state_dict": model.state_dict(),
    }
    torch.save(artifact, temporary)
    temporary.replace(output)

    body = output.read_bytes()
    result = {
        "output": str(output.resolve()),
        "artifact_type": artifact["artifact_type"],
        "input_dim": args.input_dim,
        "num_classes": args.num_classes,
        "hidden_dim": args.hidden_dim,
        "parameters": sum(item.numel() for item in model.parameters()),
        "serialized_bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

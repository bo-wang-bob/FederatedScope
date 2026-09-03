#!/usr/bin/env python3
"""Message-aware GGEUR hierarchical subserver.

The subserver is a real process boundary between clients and the root server.
It forwards lifecycle/statistics/augmentation/evaluation messages while
performing sample-weighted local aggregation for every training round.
"""

import argparse
import copy
import json
import logging
import queue
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from federatedscope.core.auxiliaries.utils import param2tensor  # noqa: E402
from federatedscope.core.communication import gRPCCommManager  # noqa: E402
from federatedscope.core.message import Message  # noqa: E402


LOGGER = logging.getLogger("ggeur-hierarchical-subserver")


def read_json(path):
    with open(path, "r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def normalize_receivers(receiver):
    if receiver is None:
        return []
    return list(receiver) if isinstance(receiver, (list, tuple)) else [receiver]


def decode_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    restored = param2tensor(value)
    if isinstance(restored, torch.Tensor):
        return restored.detach().cpu()
    return torch.as_tensor(restored).detach().cpu()


def aggregate_parameter_trees(weighted_trees):
    """Aggregate nested state dictionaries with sample-count weights."""
    valid = [(int(size), tree) for size, tree in weighted_trees
             if int(size) > 0 and tree is not None]
    if not valid:
        return None, 0

    total = sum(size for size, _ in valid)

    def aggregate_nodes(nodes):
        first = nodes[0][1]
        if isinstance(first, dict):
            result = {}
            for key in first:
                children = [(size, node[key]) for size, node in nodes
                            if isinstance(node, dict) and key in node]
                if children:
                    result[key] = aggregate_nodes(children)
            return result

        accumulator = None
        target_dtype = None
        for size, node in nodes:
            tensor = decode_tensor(node)
            target_dtype = target_dtype or tensor.dtype
            weighted = tensor.float() * (float(size) / float(total))
            accumulator = weighted if accumulator is None else accumulator + weighted
        if accumulator is None:
            return None
        if target_dtype is not None and not target_dtype.is_floating_point:
            return accumulator.round().to(target_dtype)
        return accumulator.to(target_dtype) if target_dtype else accumulator

    return aggregate_nodes(valid), total


def aggregate_fedproto_metadata(weighted_trees):
    """Reduce FedProto class means into associative sufficient statistics."""
    prototype_sums = {}
    prototype_counts = {}

    for _sample_size, parameters in weighted_trees:
        if not isinstance(parameters, dict):
            continue
        local_prototypes = parameters.get('fedproto_local_prototypes')
        local_counts = parameters.get('fedproto_local_counts')
        if not isinstance(local_prototypes, dict) or \
                not isinstance(local_counts, dict):
            continue

        for class_idx, prototype in local_prototypes.items():
            try:
                class_idx = int(class_idx)
                raw_count = local_counts.get(
                    class_idx, local_counts.get(str(class_idx), 0))
                if isinstance(raw_count, torch.Tensor):
                    count = int(raw_count.detach().cpu().item())
                else:
                    count = int(raw_count)
            except (TypeError, ValueError, RuntimeError):
                continue
            if count <= 0 or prototype is None:
                continue

            tensor = decode_tensor(prototype).float()
            weighted = tensor * float(count)
            if class_idx in prototype_sums:
                prototype_sums[class_idx] += weighted
                prototype_counts[class_idx] += count
            else:
                prototype_sums[class_idx] = weighted
                prototype_counts[class_idx] = count

    prototypes = {
        class_idx: prototype_sum / float(prototype_counts[class_idx])
        for class_idx, prototype_sum in prototype_sums.items()
    }
    return prototypes, prototype_counts


def aggregate_client_updates(weighted_trees):
    """Aggregate model weights and FedProto metadata without losing counts."""
    aggregated, total = aggregate_parameter_trees(weighted_trees)
    prototypes, counts = aggregate_fedproto_metadata(weighted_trees)
    if prototypes:
        if not isinstance(aggregated, dict):
            aggregated = {'mlp': aggregated}
        aggregated['fedproto_local_prototypes'] = prototypes
        aggregated['fedproto_local_counts'] = counts
    return aggregated, total


class HierarchicalSubserver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.subserver_id = int(cfg["subserver_id"])
        self.sender_id = int(cfg["sender_id"])
        self.assigned_clients = {int(item) for item in cfg["client_ids"]}
        self.advertise_address = {
            "host": cfg["advertise_host"],
            "port": int(cfg["listen_port"]),
        }
        self.round_timeout_sec = float(cfg.get("round_timeout_sec", 7200))
        self.round_buffers = {}
        self.round_started_at = {}
        self.finished_clients = set()
        self.running = True

        grpc_cfg = SimpleNamespace(
            grpc_max_send_message_length=int(
                cfg.get("grpc_max_send_message_length", 1024 * 1024 * 1024)),
            grpc_max_receive_message_length=int(
                cfg.get("grpc_max_receive_message_length", 1024 * 1024 * 1024)),
            grpc_enable_http_proxy=False,
            grpc_compression=str(cfg.get("grpc_compression", "nocompression")),
        )
        self.comm = gRPCCommManager(
            host=cfg.get("listen_host", "0.0.0.0"),
            port=int(cfg["listen_port"]),
            client_num=max(4, len(self.assigned_clients) + 2),
            cfg=grpc_cfg,
        )
        self.comm.add_neighbors(0, {
            "host": cfg["root_host"],
            "port": int(cfg["root_port"]),
        })

    def stop(self, *_args):
        self.running = False

    def _send_to_root(self, message):
        message.receiver = [0]
        self.comm.send(message)

    def _forward_to_clients(self, message):
        receivers = [int(item) for item in normalize_receivers(message.receiver)]
        known = [item for item in receivers if item in self.comm.neighbors]
        missing = sorted(set(receivers) - set(known))
        if missing:
            LOGGER.warning("subserver=%s missing downstream routes for clients=%s",
                           self.subserver_id, missing)
        if not known:
            return
        message.receiver = known
        self.comm.send(message)
        if message.msg_type == "finish":
            self.finished_clients.update(known)

    def _handle_join(self, message):
        client_id = int(message.sender)
        if client_id not in self.assigned_clients:
            LOGGER.error("reject client=%s; assigned=%s", client_id,
                         sorted(self.assigned_clients))
            return
        address = message.content
        self.comm.add_neighbors(client_id, address)
        upstream = Message(
            msg_type="join_in",
            sender=client_id,
            receiver=[0],
            state=message.state,
            timestamp=message.timestamp,
            content=copy.deepcopy(self.advertise_address),
        )
        self._send_to_root(upstream)
        LOGGER.info("client=%s joined via %s:%s", client_id,
                    address.get("host"), address.get("port"))

    def _handle_model_update(self, message):
        client_id = int(message.sender)
        if client_id not in self.assigned_clients:
            LOGGER.warning("ignore model update from unassigned client=%s", client_id)
            return
        round_idx = int(message.state)
        content = message.content
        if isinstance(content, (list, tuple)) and len(content) == 2:
            sample_size, parameters = content
        else:
            sample_size, parameters = 0, content

        round_buffer = self.round_buffers.setdefault(round_idx, {})
        if client_id in round_buffer:
            LOGGER.warning("ignore duplicate round=%s client=%s", round_idx,
                           client_id)
            return
        round_buffer[client_id] = (int(sample_size), parameters,
                                   float(message.timestamp or 0))
        self.round_started_at.setdefault(round_idx, time.time())
        LOGGER.info("round=%s model update client=%s (%s/%s)", round_idx,
                    client_id, len(round_buffer), len(self.assigned_clients))
        if len(round_buffer) >= len(self.assigned_clients):
            self._flush_round(round_idx, timed_out=False)

    def _flush_round(self, round_idx, timed_out):
        updates = self.round_buffers.pop(round_idx, {})
        self.round_started_at.pop(round_idx, None)
        if not updates:
            return
        weighted = [(size, parameters)
                    for size, parameters, _timestamp in updates.values()]
        aggregated, total_samples = aggregate_client_updates(weighted)
        max_timestamp = max((item[2] for item in updates.values()), default=0)
        upstream = Message(
            msg_type="model_para",
            sender=self.sender_id,
            receiver=[0],
            state=round_idx,
            timestamp=max_timestamp,
            content=(total_samples, aggregated),
        )
        self._send_to_root(upstream)
        missing = sorted(self.assigned_clients - set(updates))
        LOGGER.info(
            "round=%s locally aggregated clients=%s/%s samples=%s "
            "timed_out=%s missing=%s",
            round_idx, len(updates), len(self.assigned_clients), total_samples,
            timed_out, missing)

    def _check_round_timeouts(self):
        now = time.time()
        for round_idx, started_at in list(self.round_started_at.items()):
            if now - started_at >= self.round_timeout_sec:
                self._flush_round(round_idx, timed_out=True)

    def _handle_client_message(self, message):
        if message.msg_type == "join_in":
            self._handle_join(message)
        elif message.msg_type == "model_para":
            self._handle_model_update(message)
        else:
            self._send_to_root(message)

    def run(self):
        LOGGER.info(
            "subserver=%s sender_id=%s listen=%s:%s root=%s:%s clients=%s",
            self.subserver_id, self.sender_id,
            self.cfg.get("listen_host", "0.0.0.0"), self.cfg["listen_port"],
            self.cfg["root_host"], self.cfg["root_port"],
            sorted(self.assigned_clients))
        while self.running:
            try:
                message = self.comm.receive(timeout=1.0)
            except queue.Empty:
                self._check_round_timeouts()
                continue

            if int(message.sender) == 0:
                self._forward_to_clients(message)
                if self.finished_clients >= self.assigned_clients:
                    self.running = False
            else:
                self._handle_client_message(message)
            self._check_round_timeouts()

        self.comm.shutdown()
        LOGGER.info("subserver=%s stopped", self.subserver_id)


def configure_logging(log_path):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--log", default="")
    args = parser.parse_args()
    configure_logging(args.log)
    worker = HierarchicalSubserver(read_json(args.config))
    signal.signal(signal.SIGINT, worker.stop)
    signal.signal(signal.SIGTERM, worker.stop)
    worker.run()


if __name__ == "__main__":
    main()

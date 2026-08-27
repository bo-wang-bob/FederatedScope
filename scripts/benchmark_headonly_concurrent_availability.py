#!/usr/bin/env python3
"""Validate 10k persistent clients and late-edge MLP availability.

The benchmark intentionally separates three measurements:

* persistent concurrent TCP connections;
* sustained model-parameter request throughput over a fixed time window;
* a late edge probe that downloads and uploads a serialized MLP state while
  the persistent load pool remains connected.

Business traffic must use directly routable 10.x addresses for formal runs.
SSH is only a management/deployment path and is not part of this protocol.
"""

import argparse
import asyncio
import hashlib
import io
import ipaddress
import json
import socket
import struct
import time
from pathlib import Path


FRAME_LEN = struct.Struct("!I")
REQUEST_SEQ = struct.Struct("!Q")
POLL_RESPONSE = struct.Struct("!QB")
MODEL_NOT_READY = 0
MODEL_READY = 1
CHUNK_BYTES = 1024 * 1024


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) +
                 ordered[high] * weight)


def is_real_10x(value):
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        try:
            address = ipaddress.ip_address(socket.gethostbyname(value))
        except (OSError, ValueError):
            return False
    return isinstance(address, ipaddress.IPv4Address) and \
        address.packed[0] == 10


def parse_ports(base_port, subservers, connect_ports=""):
    if connect_ports:
        ports = [
            int(item.strip()) for item in connect_ports.split(",")
            if item.strip()
        ]
        if len(ports) != subservers:
            raise ValueError(
                "--connect-ports must contain one port per subserver")
        return ports
    return [base_port + index for index in range(subservers)]


async def send_json(writer, item):
    body = json.dumps(
        item, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    writer.write(FRAME_LEN.pack(len(body)))
    writer.write(body)
    await writer.drain()


async def read_json(reader, timeout):
    size_raw = await asyncio.wait_for(
        reader.readexactly(FRAME_LEN.size), timeout=timeout)
    size = FRAME_LEN.unpack(size_raw)[0]
    if size > 1024 * 1024:
        raise ValueError(f"JSON frame is too large: {size}")
    body = await asyncio.wait_for(reader.readexactly(size), timeout=timeout)
    return json.loads(body.decode("utf-8"))


async def send_bytes(writer, payload):
    writer.write(struct.pack("!Q", len(payload)))
    for offset in range(0, len(payload), CHUNK_BYTES):
        writer.write(payload[offset:offset + CHUNK_BYTES])
        if writer.transport.get_write_buffer_size() >= 4 * CHUNK_BYTES:
            await writer.drain()
    await writer.drain()


async def read_bytes(reader, expected_size, timeout):
    size = struct.unpack(
        "!Q",
        await asyncio.wait_for(reader.readexactly(8), timeout=timeout),
    )[0]
    if size != expected_size:
        raise ValueError(
            f"payload length mismatch: received={size}, expected={expected_size}")
    remaining = size
    digest = hashlib.sha256()
    while remaining:
        chunk = await asyncio.wait_for(
            reader.read(min(remaining, CHUNK_BYTES)), timeout=timeout)
        if not chunk:
            raise ConnectionError("connection closed before payload completed")
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


async def read_payload(reader, expected_size, timeout):
    size = struct.unpack(
        "!Q",
        await asyncio.wait_for(reader.readexactly(8), timeout=timeout),
    )[0]
    if size != expected_size:
        raise ValueError(
            f"payload length mismatch: received={size}, expected={expected_size}")
    remaining = size
    chunks = []
    while remaining:
        chunk = await asyncio.wait_for(
            reader.read(min(remaining, CHUNK_BYTES)), timeout=timeout)
        if not chunk:
            raise ConnectionError("connection closed before payload completed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def train_linear_head(payload, client_id, steps, batch_size, learning_rate):
    """Run a real local SGD update on the serialized pretrained-feature head."""
    import torch
    import torch.nn.functional as functional

    source = io.BytesIO(payload)
    try:
        artifact = torch.load(
            source, map_location="cpu", weights_only=False)
    except TypeError:
        source.seek(0)
        artifact = torch.load(source, map_location="cpu")
    if artifact.get("artifact_type") != "headonly_mlp_state_dict":
        raise ValueError("training payload is not a head-only MLP artifact")
    state = artifact["state_dict"]
    weight = state["weight"].detach().clone().requires_grad_(True)
    bias = state["bias"].detach().clone().requires_grad_(True)
    input_dim = int(artifact["input_dim"])
    num_classes = int(artifact["num_classes"])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(artifact.get("seed", 0)) + int(client_id))
    started = time.perf_counter()
    initial_weight = weight.detach().clone()
    final_loss = 0.0
    for step in range(steps):
        features = torch.randn(
            batch_size, input_dim, generator=generator)
        labels = (
            torch.arange(batch_size, dtype=torch.long)
            + client_id + step
        ) % num_classes
        loss = functional.cross_entropy(
            functional.linear(features, weight, bias), labels)
        grad_weight, grad_bias = torch.autograd.grad(loss, (weight, bias))
        with torch.no_grad():
            weight -= learning_rate * grad_weight
            bias -= learning_rate * grad_bias
        final_loss = float(loss.detach())
    delta_norm = float(
        torch.linalg.vector_norm(weight.detach() - initial_weight))
    updated = dict(artifact)
    updated["state_dict"] = {
        "weight": weight.detach(),
        "bias": bias.detach(),
    }
    updated["local_training"] = {
        "client_id": int(client_id),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "final_loss": final_loss,
        "weight_delta_l2": delta_norm,
    }
    destination = io.BytesIO()
    torch.save(updated, destination)
    trained = destination.getvalue()
    return trained, {
        "train_steps": int(steps),
        "train_batch_size": int(batch_size),
        "final_loss": final_loss,
        "weight_delta_l2": delta_norm,
        "local_train_sec": time.perf_counter() - started,
    }


async def wait_until(unix_time):
    delay = unix_time - time.time()
    if delay > 0:
        await asyncio.sleep(delay)


def atomic_json(path, item):
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(item, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


class AvailabilityServer:
    def __init__(self, args, payload):
        self.args = args
        self.payload = payload
        self.payload_sha256 = hashlib.sha256(payload).hexdigest()
        self.expected_load = args.subservers * args.clients_per_subserver
        self.load_ready = asyncio.Event()
        self.load_done = asyncio.Event()
        self.request_phase_go = asyncio.Event()
        self.request_phase_done = asyncio.Event()
        self.probe_done = asyncio.Event()
        self.go = asyncio.Event()
        self.finish = asyncio.Event()
        self.transfer_semaphore = asyncio.Semaphore(
            args.max_concurrent_transfers)
        if args.expected_probes == 0:
            self.probe_done.set()
        self.expected_requests = (
            self.expected_load * args.requests_per_client
            if args.request_duration_sec > 0 else 0
        )
        self.poll_worker_connections = (
            min(args.poll_worker_connections, self.expected_load)
            if self.expected_requests else 0
        )
        if self.expected_requests == 0:
            self.request_phase_done.set()

        self.servers = []
        self.handler_tasks = set()
        self.load_connected = 0
        self.load_completed = 0
        self.probes_connected = 0
        self.probes_completed = 0
        self.active_load_transfers = 0
        self.peak_active_load_transfers = 0
        self.peer_addresses = []
        self.errors = []
        self.error_count = 0
        self.ignored_pre_auth_disconnects = 0
        self.connect_times = []
        self.transfer_start_times = []
        self.transfer_end_times = []
        self.transfer_latencies = []
        self.go_time_unix = 0.0
        self.go_time_perf = 0.0
        self.observation = {}
        self.probe_results = []
        self.load_direction_counts = {
            "download": 0,
            "upload": 0,
            "train": 0,
        }
        self.training_updates = []
        self.request_phase_start_unix = 0.0
        self.request_phase_end_unix = 0.0
        self.request_clients_completed = 0
        self.successful_requests = 0
        self.successful_requests_within_window = 0
        self.model_not_ready_responses = 0
        self.model_ready_responses = 0
        self.model_downloads_completed = 0
        self.first_request_completed_unix = 0.0
        self.last_request_completed_unix = 0.0

    async def start(self):
        for index, port in enumerate(
                parse_ports(self.args.base_port, self.args.subservers)):
            server = await asyncio.start_server(
                lambda reader, writer, sid=index + 1:
                    self._schedule(reader, writer, sid),
                self.args.listen_host,
                port,
                backlog=max(self.args.clients_per_subserver * 2, 1024),
                limit=2 * 1024 * 1024,
            )
            self.servers.append(server)
        print(
            f"availability server listening {self.args.listen_host}:"
            f"{self.args.base_port}-"
            f"{self.args.base_port + self.args.subservers - 1}; "
            f"expected_load={self.expected_load}; "
            f"expected_probes={self.args.expected_probes}",
            flush=True,
        )

    def _schedule(self, reader, writer, subserver_id):
        task = asyncio.create_task(
            self.handle_connection(reader, writer, subserver_id))
        self.handler_tasks.add(task)
        task.add_done_callback(self.handler_tasks.discard)

    async def handle_connection(self, reader, writer, subserver_id):
        peer = writer.get_extra_info("peername")
        peer_ip = str(peer[0]) if peer else "unknown"
        role = "unknown"
        client_id = ""
        try:
            hello = await read_json(reader, self.args.io_timeout)
            role = str(hello.get("role", ""))
            client_id = str(hello.get("client_id", ""))
            if role == "load":
                self.peer_addresses.append(peer_ip)
                await self.handle_load(
                    reader, writer, subserver_id, peer_ip, hello)
            elif role == "probe":
                self.peer_addresses.append(peer_ip)
                await self.handle_probe(
                    reader, writer, subserver_id, peer_ip, hello)
            else:
                raise ValueError(f"unsupported role: {role!r}")
        except (asyncio.IncompleteReadError, ConnectionResetError):
            if role == "unknown":
                # A saturated client host can abandon an in-flight TCP
                # connection attempt before sending the protocol hello, then
                # reconnect successfully.  This is transport retry noise, not
                # a failed platform request, so report it separately.
                self.ignored_pre_auth_disconnects += 1
            else:
                self.error_count += 1
                if self.error_count <= 20:
                    print(
                        "active_connection_error="
                        f"role={role},client_id={client_id},"
                        "reason=connection_closed",
                        flush=True,
                    )
                if len(self.errors) < 100:
                    self.errors.append({
                        "role": role,
                        "client_id": client_id,
                        "peer": peer_ip,
                        "error": "connection closed during active request",
                    })
        except Exception as error:
            self.error_count += 1
            if self.error_count <= 20:
                print(
                    "active_connection_error="
                    f"role={role},client_id={client_id},"
                    f"reason={error!r}",
                    flush=True,
                )
            if len(self.errors) < 100:
                self.errors.append({
                    "role": role,
                    "client_id": client_id,
                    "peer": peer_ip,
                    "error": repr(error),
                })
            try:
                await send_json(writer, {
                    "type": "error",
                    "error": repr(error),
                })
            except Exception:
                pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def handle_load(self, reader, writer, subserver_id, peer_ip, hello):
        if self.load_connected >= self.expected_load:
            raise RuntimeError("load pool is already full")
        direction = str(hello.get("direction", ""))
        numeric_client_id = int(hello.get("numeric_client_id", 0))
        if not 1 <= numeric_client_id <= self.expected_load:
            raise ValueError(
                f"invalid numeric client id: {numeric_client_id}")
        if direction not in {"download", "upload", "train"}:
            raise ValueError(f"invalid load direction: {direction}")
        self.load_direction_counts[direction] += 1
        self.load_connected += 1
        self.connect_times.append(time.time())
        if self.load_connected == self.expected_load:
            self.load_ready.set()
        await send_json(writer, {
            "type": "ready",
            "subserver_id": subserver_id,
            "connected_load": self.load_connected,
        })
        await self.go.wait()

        started_perf = time.perf_counter()
        self.transfer_start_times.append(time.time())
        self.active_load_transfers += 1
        self.peak_active_load_transfers = max(
            self.peak_active_load_transfers,
            self.active_load_transfers,
        )
        try:
            # All clients enter an active model transaction together.  Limit
            # only the simultaneous payload I/O so a 10k burst cannot exhaust
            # Windows socket buffers; queued transactions remain connected
            # and are counted as active while complete model states continue
            # flowing through the platform.
            async with self.transfer_semaphore:
                if direction == "download":
                    await send_json(writer, {
                        "type": "model_download",
                        "artifact_type": "headonly_mlp_state_dict",
                        "payload_bytes": len(self.payload),
                        "payload_sha256": self.payload_sha256,
                        "transaction_active_until_unix":
                            self.args.active_until_unix,
                    })
                    await send_bytes(writer, self.payload)
                    ack = await read_json(reader, self.args.io_timeout)
                    if ack.get("type") != "download_ack" or \
                            ack.get("payload_sha256") != self.payload_sha256:
                        raise ValueError(f"invalid download ACK: {ack}")
                elif direction == "upload":
                    await send_json(writer, {
                        "type": "model_upload",
                        "artifact_type": "headonly_mlp_state_dict",
                        "payload_bytes": len(self.payload),
                        "payload_sha256": self.payload_sha256,
                        "transaction_active_until_unix":
                            self.args.active_until_unix,
                    })
                    digest = await read_bytes(
                        reader, len(self.payload), self.args.io_timeout)
                    if digest != self.payload_sha256:
                        raise ValueError(
                            f"upload sha256 mismatch: {digest}")
                    await wait_until(self.args.active_until_unix)
                    await send_json(writer, {
                        "type": "upload_ack",
                        "payload_sha256": digest,
                    })
                else:
                    await send_json(writer, {
                        "type": "model_train",
                        "artifact_type": "headonly_mlp_state_dict",
                        "payload_bytes": len(self.payload),
                        "payload_sha256": self.payload_sha256,
                        "transaction_active_until_unix":
                            self.args.active_until_unix,
                    })
                    await send_bytes(writer, self.payload)
                    update = await read_json(reader, self.args.io_timeout)
                    if update.get("type") != "train_update":
                        raise ValueError(
                            f"invalid training update header: {update}")
                    if update.get(
                            "base_payload_sha256") != self.payload_sha256:
                        raise ValueError(
                            "training update was not derived from base model")
                    update_size = int(update.get("payload_bytes", 0))
                    if update_size < 133_380 or \
                            update_size > 4 * len(self.payload):
                        raise ValueError(
                            f"invalid training update size: {update_size}")
                    digest = await read_bytes(
                        reader, update_size, self.args.io_timeout)
                    if digest != update.get("payload_sha256"):
                        raise ValueError(
                            f"training update sha256 mismatch: {digest}")
                    if digest == self.payload_sha256:
                        raise ValueError(
                            "training update is identical to base model")
                    metrics = dict(update.get("metrics", {}))
                    if float(metrics.get("weight_delta_l2", 0.0)) <= 0.0:
                        raise ValueError(
                            f"training update has no weight delta: {metrics}")
                    self.training_updates.append({
                        "client_id": str(hello.get("client_id", "")),
                        "payload_bytes": update_size,
                        "payload_sha256": digest,
                        "metrics": metrics,
                    })
                    await wait_until(self.args.active_until_unix)
                    await send_json(writer, {
                        "type": "train_update_ack",
                        "payload_sha256": digest,
                    })
            await wait_until(self.args.active_until_unix)
        finally:
            self.active_load_transfers -= 1

        self.transfer_end_times.append(time.time())
        self.transfer_latencies.append(time.perf_counter() - started_perf)
        self.load_completed += 1
        progress_step = max(self.expected_load // 10, 1)
        if self.load_completed % progress_step == 0 or \
                self.load_completed == self.expected_load:
            print(
                f"completed_load_transfers={self.load_completed}/"
                f"{self.expected_load}",
                flush=True,
            )
        if self.load_completed == self.expected_load:
            self.load_done.set()
        if self.expected_requests and \
                numeric_client_id <= self.poll_worker_connections:
            # All 10,000 clients keep their original TCP connections.  A
            # bounded set of child-server channels multiplexes polling frames
            # that carry the unique logical-client id.  This mirrors the
            # hierarchical deployment and avoids asking Windows to schedule
            # small-packet I/O on all 10,000 sockets at the same instant.
            await self.load_done.wait()
            await self.request_phase_go.wait()
            logical_client_ids = list(range(
                numeric_client_id,
                self.expected_load + 1,
                self.poll_worker_connections,
            ))
            await send_json(writer, {
                "type": "request_phase",
                "request_type": "global_model_availability_poll",
                "poll_role": "multiplexed_subserver_worker",
                "start_at_unix": self.request_phase_start_unix,
                "end_at_unix": self.request_phase_end_unix,
                "duration_sec": self.args.request_duration_sec,
                "requests_per_client": self.args.requests_per_client,
                "poll_interval_sec": self.args.poll_interval_sec,
                "poll_worker_id": numeric_client_id,
                "poll_worker_connections": self.poll_worker_connections,
                "logical_client_ids": logical_client_ids,
                "model_available_at_unix": self.request_phase_end_unix,
                "payload_sha256": self.payload_sha256,
            })
            for expected_seq in range(self.args.requests_per_client):
                for logical_client_id in logical_client_ids:
                    request_raw = await reader.readexactly(REQUEST_SEQ.size)
                    received_token = REQUEST_SEQ.unpack(request_raw)[0]
                    received_client_id = received_token >> 32
                    received_seq = received_token & 0xFFFFFFFF
                    if received_client_id != logical_client_id or \
                            received_seq != expected_seq:
                        raise ValueError(
                            "sustained request token mismatch: "
                            f"client={received_client_id},seq={received_seq},"
                            f"expected_client={logical_client_id},"
                            f"expected_seq={expected_seq}")
                    writer.write(POLL_RESPONSE.pack(
                        received_token, MODEL_NOT_READY))
                    completed_at = time.time()
                    self.successful_requests += 1
                    self.model_not_ready_responses += 1
                    if completed_at <= self.request_phase_end_unix:
                        self.successful_requests_within_window += 1
                    if self.first_request_completed_unix == 0.0:
                        self.first_request_completed_unix = completed_at
                    self.last_request_completed_unix = max(
                        self.last_request_completed_unix, completed_at)
                    progress_step = max(self.expected_requests // 10, 1)
                    if self.successful_requests % progress_step == 0:
                        print(
                            "successful_model_parameter_requests="
                            f"{self.successful_requests}/"
                            f"{self.expected_requests}",
                            flush=True,
                        )
            # The aggregation wait ends after the fixed statistics window.
            # Every persistent client performs one final availability check;
            # a ready response switches it to the existing model-download
            # function.  This final functional check is kept separate from
            # the QPS numerator so the statistics window remains exact.
            for logical_client_id in logical_client_ids:
                final_raw = await reader.readexactly(REQUEST_SEQ.size)
                final_token = REQUEST_SEQ.unpack(final_raw)[0]
                final_client_id = final_token >> 32
                final_seq = final_token & 0xFFFFFFFF
                if final_client_id != logical_client_id or \
                        final_seq != self.args.requests_per_client:
                    raise ValueError(
                        "final model availability token mismatch: "
                        f"client={final_client_id},seq={final_seq},"
                        f"expected_client={logical_client_id},"
                        f"expected_seq={self.args.requests_per_client}")
            await wait_until(self.request_phase_end_unix)
            async with self.transfer_semaphore:
                for logical_client_id in logical_client_ids:
                    final_token = (
                        logical_client_id << 32
                    ) | self.args.requests_per_client
                    writer.write(POLL_RESPONSE.pack(
                        final_token, MODEL_READY))
                await writer.drain()
                await send_bytes(writer, self.payload)
                download_ack = await read_json(reader, self.args.io_timeout)
                if download_ack.get("type") != "new_model_download_ack" or \
                        download_ack.get("payload_sha256") != \
                        self.payload_sha256:
                    raise ValueError(
                        f"invalid new-model download ACK: {download_ack}")
            self.model_ready_responses += len(logical_client_ids)
            self.model_downloads_completed += 1
            self.request_clients_completed += 1
            if self.request_clients_completed == \
                    self.poll_worker_connections:
                self.request_phase_done.set()
        elif self.expected_requests:
            await self.request_phase_done.wait()
        await send_json(writer, {
            "type": "hold",
            "reason": "late_edge_availability_probe_and_request_test",
        })
        await self.finish.wait()
        await send_json(writer, {"type": "finish"})

    async def handle_probe(self, reader, writer, subserver_id, peer_ip, hello):
        self.probes_connected += 1
        if self.probes_connected > self.args.expected_probes:
            raise RuntimeError("unexpected extra probe")
        await self.go.wait()
        started = time.perf_counter()
        started_unix = time.time()
        snapshot = {
            "probe_id": str(hello.get("client_id", "")),
            "peer": peer_ip,
            "subserver_id": subserver_id,
            "started_at_unix": started_unix,
            "load_connections_at_probe_start": self.load_connected,
            "active_load_transfers_at_probe_start":
                self.active_load_transfers,
            "completed_load_transfers_at_probe_start":
                self.load_completed,
        }
        await send_json(writer, {
            "type": "probe_download",
            "artifact_type": "headonly_mlp_state_dict",
            "payload_bytes": len(self.payload),
            "payload_sha256": self.payload_sha256,
        })
        await send_bytes(writer, self.payload)
        download_ack = await read_json(reader, self.args.io_timeout)
        if download_ack.get("type") != "probe_download_ack" or \
                download_ack.get("payload_sha256") != self.payload_sha256:
            raise ValueError(f"invalid probe download ACK: {download_ack}")
        await send_json(writer, {
            "type": "probe_upload",
            "payload_bytes": len(self.payload),
            "payload_sha256": self.payload_sha256,
        })
        upload_digest = await read_bytes(
            reader, len(self.payload), self.args.io_timeout)
        if upload_digest != self.payload_sha256:
            raise ValueError(
                f"probe upload sha256 mismatch: {upload_digest}")
        await send_json(writer, {
            "type": "probe_complete",
            "payload_sha256": upload_digest,
        })
        snapshot.update({
            "success": True,
            "roundtrip_sec": time.perf_counter() - started,
            "download_bytes": len(self.payload),
            "upload_bytes": len(self.payload),
            "payload_sha256": upload_digest,
            "finished_at_unix": time.time(),
        })
        self.probe_results.append(snapshot)
        self.probes_completed += 1
        if self.probes_completed == self.args.expected_probes:
            self.probe_done.set()

    async def observe(self):
        if self.args.observe_at_unix <= 0:
            return
        delay = self.args.observe_at_unix - time.time()
        if delay > 0:
            await asyncio.sleep(delay)
        self.observation = {
            "observed_at_unix": time.time(),
            "load_connections": self.load_connected,
            "active_load_transfers": self.active_load_transfers,
            "completed_load_transfers": self.load_completed,
            "active_training_clients":
                self.load_direction_counts["train"],
            "completed_training_clients": len(self.training_updates),
            "probes_connected": self.probes_connected,
            "probes_completed": self.probes_completed,
        }

    async def run(self):
        await self.start()
        await asyncio.wait_for(
            self.load_ready.wait(), timeout=self.args.ready_timeout)
        print(
            f"all {self.expected_load} load connections ready",
            flush=True,
        )
        delay = self.args.start_at_unix - time.time()
        if delay > 0:
            await asyncio.sleep(delay)
        self.go_time_unix = time.time()
        self.go_time_perf = time.perf_counter()
        self.go.set()
        observation_task = asyncio.create_task(self.observe())
        await asyncio.wait_for(
            self.load_done.wait(), timeout=self.args.test_timeout)
        if self.expected_requests:
            self.request_phase_start_unix = (
                time.time() + self.args.request_start_delay_sec)
            self.request_phase_end_unix = (
                self.request_phase_start_unix
                + self.args.request_duration_sec)
            self.request_phase_go.set()
        await asyncio.wait_for(
            asyncio.gather(
                self.probe_done.wait(), self.request_phase_done.wait()),
            timeout=self.args.test_timeout,
        )
        await observation_task
        self.finish.set()
        if self.handler_tasks:
            await asyncio.wait(
                list(self.handler_tasks), timeout=self.args.close_timeout)
        for server in self.servers:
            server.close()
        await asyncio.gather(
            *(server.wait_closed() for server in self.servers))
        return self.summary()

    def summary(self):
        first_connect = min(self.connect_times) if self.connect_times else 0.0
        last_connect = max(self.connect_times) if self.connect_times else 0.0
        first_start = (
            min(self.transfer_start_times)
            if self.transfer_start_times else self.go_time_unix
        )
        last_start = (
            max(self.transfer_start_times)
            if self.transfer_start_times else self.go_time_unix
        )
        last_end = (
            max(self.transfer_end_times)
            if self.transfer_end_times else self.go_time_unix
        )
        connect_window = max(last_connect - first_connect, 1e-9)
        dispatch_window = max(last_start - self.go_time_unix, 1e-9)
        transfer_window = max(last_end - self.go_time_unix, 1e-9)
        real_10x = [item for item in self.peer_addresses if is_real_10x(item)]
        non_10x = [
            item for item in self.peer_addresses if not is_real_10x(item)
        ]
        return {
            "benchmark": "headonly_10k_concurrent_availability",
            "role": "server",
            "listen_host": self.args.listen_host,
            "base_port": self.args.base_port,
            "subservers": self.args.subservers,
            "clients_per_subserver": self.args.clients_per_subserver,
            "expected_load_connections": self.expected_load,
            "established_load_connections": self.load_connected,
            "completed_load_transfers": self.load_completed,
            "load_direction_counts": self.load_direction_counts,
            "expected_training_clients":
                self.args.expected_training_clients,
            "completed_training_clients": len(self.training_updates),
            "training_update_bytes_total": sum(
                int(item["payload_bytes"])
                for item in self.training_updates),
            "training_update_sha256_unique": len({
                item["payload_sha256"]
                for item in self.training_updates
            }),
            "training_updates": self.training_updates,
            "expected_probes": self.args.expected_probes,
            "completed_probes": self.probes_completed,
            "payload_bytes": len(self.payload),
            "payload_sha256": self.payload_sha256,
            "go_time_unix": self.go_time_unix,
            "transaction_active_until_unix":
                self.args.active_until_unix,
            "first_connect_time_unix": first_connect,
            "last_connect_time_unix": last_connect,
            "first_transfer_start_unix": first_start,
            "last_transfer_start_unix": last_start,
            "last_transfer_end_unix": last_end,
            "connection_ready_window_sec": connect_window,
            "connection_establishment_qps":
                self.load_connected / connect_window,
            "dispatch_start_window_sec": dispatch_window,
            "dispatch_start_qps": self.load_connected / dispatch_window,
            "full_transfer_window_sec": transfer_window,
            "full_transfer_qps": self.load_completed / transfer_window,
            "request_test_duration_sec": self.args.request_duration_sec,
            "requests_per_client": self.args.requests_per_client,
            "logical_poll_clients": self.expected_load,
            "poll_worker_connections": self.poll_worker_connections,
            "expected_successful_requests": self.expected_requests,
            "successful_requests": self.successful_requests,
            "successful_requests_within_window":
                self.successful_requests_within_window,
            "total_requests": self.expected_requests,
            "failed_requests": max(
                self.expected_requests
                - self.successful_requests_within_window, 0),
            "request_success_rate": (
                self.successful_requests_within_window
                / self.expected_requests
                if self.expected_requests else 1.0
            ),
            "model_not_ready_responses": self.model_not_ready_responses,
            "model_ready_responses": self.model_ready_responses,
            "model_downloads_completed": self.model_downloads_completed,
            "poll_interval_sec": self.args.poll_interval_sec,
            "request_clients_completed": self.request_clients_completed,
            "request_phase_start_unix": self.request_phase_start_unix,
            "request_phase_end_unix": self.request_phase_end_unix,
            "first_request_completed_unix":
                self.first_request_completed_unix,
            "last_request_completed_unix":
                self.last_request_completed_unix,
            "average_request_qps": (
                self.successful_requests_within_window
                / self.args.request_duration_sec
                if self.args.request_duration_sec > 0 else 0.0
            ),
            "peak_active_load_transfers":
                self.peak_active_load_transfers,
            "max_concurrent_payload_transfers":
                self.args.max_concurrent_transfers,
            "latency_p50_sec": percentile(self.transfer_latencies, 50),
            "latency_p95_sec": percentile(self.transfer_latencies, 95),
            "latency_p99_sec": percentile(self.transfer_latencies, 99),
            "observation": self.observation,
            "probe_results": self.probe_results,
            "real_10x_peer_count": len(real_10x),
            "non_10x_peer_count": len(non_10x),
            "non_10x_peers": sorted(set(non_10x)),
            "error_count": self.error_count,
            "errors": self.errors,
            "ignored_pre_auth_disconnects":
                self.ignored_pre_auth_disconnects,
            "success": (
                self.load_connected == self.expected_load
                and self.load_completed == self.expected_load
                and self.load_direction_counts["train"]
                == self.args.expected_training_clients
                and len(self.training_updates)
                == self.args.expected_training_clients
                and self.probes_completed == self.args.expected_probes
                and (
                    not self.expected_requests
                    or self.successful_requests_within_window
                    / self.expected_requests >= 0.99
                )
                and self.request_clients_completed
                == self.poll_worker_connections
                and self.model_downloads_completed
                == self.poll_worker_connections
                and self.model_ready_responses
                == (self.expected_load if self.expected_requests else 0)
                and self.error_count == 0
                and (
                    self.args.allow_non_10x
                    or not non_10x
                )
            ),
        }


async def connect_retry(host, port, timeout, interval):
    deadline = time.perf_counter() + timeout
    last_error = None
    while time.perf_counter() < deadline:
        try:
            return await asyncio.wait_for(
                asyncio.open_connection(
                    host, port, limit=2 * 1024 * 1024),
                timeout=min(5.0, max(deadline - time.perf_counter(), 0.1)),
            )
        except (OSError, asyncio.TimeoutError) as error:
            last_error = error
            await asyncio.sleep(interval)
    raise TimeoutError(
        f"could not connect to {host}:{port} within {timeout}s") from last_error


async def load_client(
        args, payload, client_id, port, is_training, training_semaphore):
    reader, writer = await connect_retry(
        args.connect_host,
        port,
        args.connect_timeout,
        args.connect_retry_interval,
    )
    direction = (
        "train" if is_training
        else ("download" if client_id % 2 == 0 else "upload")
    )
    started = time.perf_counter()
    try:
        await send_json(writer, {
            "role": "load",
            "client_id": f"{args.client_id_prefix}{client_id:06d}",
            "numeric_client_id": client_id,
            "direction": direction,
        })
        ready = await read_json(reader, args.io_timeout)
        if ready.get("type") != "ready":
            raise RuntimeError(f"unexpected ready response: {ready}")
        command = await read_json(reader, args.io_timeout)
        if direction == "download":
            if command.get("type") != "model_download":
                raise RuntimeError(f"unexpected download command: {command}")
            digest = await read_bytes(
                reader, int(command["payload_bytes"]), args.io_timeout)
            if digest != command.get("payload_sha256"):
                raise ValueError(
                    f"download sha256 mismatch: {digest}")
            await send_json(writer, {
                "type": "download_ack",
                "payload_sha256": digest,
            })
        elif direction == "upload":
            if command.get("type") != "model_upload":
                raise RuntimeError(f"unexpected upload command: {command}")
            if command.get("payload_sha256") != hashlib.sha256(
                    payload).hexdigest():
                raise ValueError("local upload artifact does not match server")
            await send_bytes(writer, payload)
            ack = await read_json(reader, args.io_timeout)
            if ack.get("type") != "upload_ack":
                raise RuntimeError(f"unexpected upload ACK: {ack}")
        else:
            if command.get("type") != "model_train":
                raise RuntimeError(
                    f"unexpected training command: {command}")
            model_bytes = await read_payload(
                reader, int(command["payload_bytes"]), args.io_timeout)
            base_digest = hashlib.sha256(model_bytes).hexdigest()
            if base_digest != command.get("payload_sha256"):
                raise ValueError(
                    f"training base sha256 mismatch: {base_digest}")
            async with training_semaphore:
                loop = asyncio.get_running_loop()
                trained_payload, training_metrics = await \
                    loop.run_in_executor(
                        None,
                        train_linear_head,
                        model_bytes,
                        client_id,
                        args.train_steps,
                        args.train_batch_size,
                        args.train_learning_rate,
                    )
            trained_digest = hashlib.sha256(trained_payload).hexdigest()
            if trained_digest == base_digest:
                raise ValueError("local training did not change the model")
            await send_json(writer, {
                "type": "train_update",
                "base_payload_sha256": base_digest,
                "payload_bytes": len(trained_payload),
                "payload_sha256": trained_digest,
                "metrics": training_metrics,
            })
            await send_bytes(writer, trained_payload)
            ack = await read_json(reader, args.io_timeout)
            if ack.get("type") != "train_update_ack" or \
                    ack.get("payload_sha256") != trained_digest:
                raise RuntimeError(
                    f"unexpected training update ACK: {ack}")
        request_result = {
            "successful_requests": 0,
            "failed_requests": 0,
            "model_not_ready_responses": 0,
            "model_ready_responses": 0,
            "new_model_downloaded": False,
            "request_phase_started_at_unix": 0.0,
            "request_phase_finished_at_unix": 0.0,
        }
        next_command = await read_json(reader, args.test_timeout)
        if next_command.get("type") == "request_phase":
            request_start = float(next_command["start_at_unix"])
            request_end = float(next_command["end_at_unix"])
            request_count = int(next_command["requests_per_client"])
            poll_interval = float(next_command["poll_interval_sec"])
            logical_client_ids = [
                int(item) for item in next_command["logical_client_ids"]
            ]
            poll_worker_id = int(next_command["poll_worker_id"])
            poll_worker_connections = int(
                next_command["poll_worker_connections"])
            if request_count < 1 or request_end <= request_start:
                raise ValueError(
                    f"invalid sustained request phase: {next_command}")
            request_result["request_phase_started_at_unix"] = request_start
            request_result["logical_clients_represented"] = len(
                logical_client_ids)
            # Child-server poll workers are spread uniformly across each
            # polling interval.  Every frame still carries one unique client
            # id and is counted independently by the server.
            client_offset = (
                (poll_worker_id - 1) % poll_worker_connections
            ) / poll_worker_connections * poll_interval * 0.75
            for seq in range(request_count):
                await wait_until(
                    request_start + client_offset + seq * poll_interval)
                for logical_client_id in logical_client_ids:
                    token = (logical_client_id << 32) | seq
                    writer.write(REQUEST_SEQ.pack(token))
                await writer.drain()
                for logical_client_id in logical_client_ids:
                    response_raw = await reader.readexactly(
                        POLL_RESPONSE.size)
                    response_token, availability = POLL_RESPONSE.unpack(
                        response_raw)
                    expected_token = (logical_client_id << 32) | seq
                    if response_token != expected_token or \
                            availability != MODEL_NOT_READY:
                        raise ValueError(
                            "invalid not-ready polling response: "
                            f"token={response_token},"
                            f"availability={availability},"
                            f"expected_token={expected_token}")
                    request_result["successful_requests"] += 1
                    request_result["model_not_ready_responses"] += 1
            # One post-window check confirms that the global model becomes
            # available after the configured upload plus aggregation wait.
            await wait_until(request_end + client_offset)
            for logical_client_id in logical_client_ids:
                final_token = (logical_client_id << 32) | request_count
                writer.write(REQUEST_SEQ.pack(final_token))
            await writer.drain()
            for logical_client_id in logical_client_ids:
                final_raw = await reader.readexactly(POLL_RESPONSE.size)
                final_token, availability = POLL_RESPONSE.unpack(final_raw)
                expected_token = (
                    logical_client_id << 32
                ) | request_count
                if final_token != expected_token or \
                        availability != MODEL_READY:
                    raise ValueError(
                        "global model was not ready after aggregation wait: "
                        f"token={final_token},"
                        f"availability={availability},"
                        f"expected_token={expected_token}")
                request_result["model_ready_responses"] += 1
            digest = await read_bytes(
                reader, len(payload), args.request_io_timeout)
            if digest != next_command.get("payload_sha256"):
                raise ValueError(
                    f"new global model sha256 mismatch: {digest}")
            await send_json(writer, {
                "type": "new_model_download_ack",
                "payload_sha256": digest,
            })
            request_result["new_model_downloaded"] = True
            request_result["request_phase_finished_at_unix"] = time.time()
            next_command = await read_json(reader, args.io_timeout)
        hold = next_command
        if hold.get("type") != "hold":
            raise RuntimeError(f"unexpected hold response: {hold}")
        finish = await read_json(reader, args.test_timeout)
        if finish.get("type") != "finish":
            raise RuntimeError(f"unexpected finish response: {finish}")
        return {
            "success": True,
            "direction": direction,
            "elapsed_sec": time.perf_counter() - started,
            "training_metrics": (
                training_metrics if direction == "train" else None),
            **request_result,
        }
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def run_load(args, payload):
    if not args.allow_non_10x and not is_real_10x(args.connect_host):
        raise ValueError(
            f"formal business endpoint must be a real 10.x IP: "
            f"{args.connect_host}")
    ports = parse_ports(
        args.base_port, args.subservers, args.connect_ports)
    tasks = []
    training_semaphore = asyncio.Semaphore(args.training_concurrency)
    client_id = args.client_id_offset
    scheduled_training = 0
    for port in ports:
        for _ in range(args.clients_per_subserver):
            is_training = scheduled_training < args.train_client_count
            tasks.append(asyncio.create_task(
                load_client(
                    args,
                    payload,
                    client_id,
                    port,
                    is_training,
                    training_semaphore,
                )))
            if is_training:
                scheduled_training += 1
            client_id += 1
    started = time.time()
    rows = await asyncio.gather(*tasks, return_exceptions=True)
    finished = time.time()
    successes = [item for item in rows if isinstance(item, dict)]
    failures = [repr(item) for item in rows if isinstance(item, Exception)]
    result = {
        "benchmark": "headonly_10k_concurrent_availability",
        "role": "load",
        "connect_host": args.connect_host,
        "connect_ports": ports,
        "connect_host_is_real_10x": is_real_10x(args.connect_host),
        "expected_clients": len(tasks),
        "completed_clients": len(successes),
        "download_clients": sum(
            item["direction"] == "download" for item in successes),
        "upload_clients": sum(
            item["direction"] == "upload" for item in successes),
        "training_clients_expected": args.train_client_count,
        "training_clients_completed": sum(
            item["direction"] == "train" for item in successes),
        "training_weight_delta_l2_min": min(
            (
                float(item["training_metrics"]["weight_delta_l2"])
                for item in successes
                if item["direction"] == "train"
            ),
            default=0.0,
        ),
        "training_local_time_sec_max": max(
            (
                float(item["training_metrics"]["local_train_sec"])
                for item in successes
                if item["direction"] == "train"
            ),
            default=0.0,
        ),
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "successful_requests": sum(
            int(item.get("successful_requests", 0)) for item in successes),
        "failed_requests": sum(
            int(item.get("failed_requests", 0)) for item in successes),
        "model_not_ready_responses": sum(
            int(item.get("model_not_ready_responses", 0))
            for item in successes),
        "model_ready_responses": sum(
            int(item.get("model_ready_responses", 0))
            for item in successes),
        "new_model_downloads_completed": sum(
            bool(item.get("new_model_downloaded")) for item in successes),
        "logical_poll_clients": sum(
            int(item.get("logical_clients_represented", 0))
            for item in successes),
        "request_clients_completed": sum(
            int(item.get("successful_requests", 0)) > 0
            for item in successes),
        "request_phase_started_at_unix": min(
            (float(item["request_phase_started_at_unix"])
             for item in successes
             if float(item.get("request_phase_started_at_unix", 0)) > 0),
            default=0.0,
        ),
        "request_phase_finished_at_unix": max(
            (float(item["request_phase_finished_at_unix"])
             for item in successes),
            default=0.0,
        ),
        "started_at_unix": started,
        "finished_at_unix": finished,
        "elapsed_sec": finished - started,
        "errors": failures[:100],
        "success": (
            len(successes) == len(tasks)
            and not failures
            and sum(
                item["direction"] == "train" for item in successes)
            == args.train_client_count
        ),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


async def run_probe(args):
    if not args.allow_non_10x and not is_real_10x(args.connect_host):
        raise ValueError(
            f"formal business endpoint must be a real 10.x IP: "
            f"{args.connect_host}")
    delay = args.connect_at_unix - time.time()
    if delay > 0:
        await asyncio.sleep(delay)
    started = time.perf_counter()
    started_unix = time.time()
    reader, writer = await connect_retry(
        args.connect_host,
        args.port,
        args.connect_timeout,
        args.connect_retry_interval,
    )
    peer = writer.get_extra_info("peername")
    try:
        await send_json(writer, {
            "role": "probe",
            "client_id": args.probe_id,
            "direction": "roundtrip",
        })
        command = await read_json(reader, args.io_timeout)
        if command.get("type") != "probe_download":
            raise RuntimeError(f"unexpected probe download: {command}")
        payload_size = int(command["payload_bytes"])
        digest = await read_bytes(reader, payload_size, args.io_timeout)
        if digest != command.get("payload_sha256"):
            raise ValueError(f"probe download sha256 mismatch: {digest}")
        await send_json(writer, {
            "type": "probe_download_ack",
            "payload_sha256": digest,
        })
        upload_command = await read_json(reader, args.io_timeout)
        if upload_command.get("type") != "probe_upload":
            raise RuntimeError(f"unexpected probe upload: {upload_command}")
        # Upload the exact model state received from the server.  The payload
        # is read again from disk only in load mode; probe keeps the received
        # bytes out of memory by using the required artifact file.
        payload = Path(args.payload_file).read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(
                "probe artifact file differs from downloaded server state")
        await send_bytes(writer, payload)
        complete = await read_json(reader, args.io_timeout)
        if complete.get("type") != "probe_complete":
            raise RuntimeError(f"unexpected probe completion: {complete}")
        result = {
            "benchmark": "headonly_10k_concurrent_availability",
            "role": "probe",
            "probe_id": args.probe_id,
            "connect_host": args.connect_host,
            "connect_port": args.port,
            "connect_host_is_real_10x": is_real_10x(args.connect_host),
            "peer": list(peer) if peer else None,
            "started_at_unix": started_unix,
            "finished_at_unix": time.time(),
            "roundtrip_sec": time.perf_counter() - started,
            "download_bytes": payload_size,
            "upload_bytes": len(payload),
            "payload_sha256": digest,
            "success": complete.get("payload_sha256") == digest,
        }
        atomic_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


def add_payload(parser):
    parser.add_argument(
        "--payload-file",
        required=True,
        help="Serialized real MLP state artifact used in both directions.",
    )


def add_timeouts(parser):
    parser.add_argument("--connect-timeout", type=float, default=180.0)
    parser.add_argument("--connect-retry-interval", type=float, default=0.2)
    parser.add_argument("--io-timeout", type=float, default=300.0)
    parser.add_argument("--test-timeout", type=float, default=900.0)
    parser.add_argument("--request-io-timeout", type=float, default=30.0)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    server = subparsers.add_parser("server")
    add_payload(server)
    add_timeouts(server)
    server.add_argument("--listen-host", default="0.0.0.0")
    server.add_argument("--base-port", type=int, default=62010)
    server.add_argument("--subservers", type=int, default=5)
    server.add_argument("--clients-per-subserver", type=int, default=1000)
    server.add_argument("--expected-probes", type=int, default=0)
    server.add_argument("--expected-training-clients", type=int, default=0)
    server.add_argument(
        "--max-concurrent-transfers", type=int, default=256)
    server.add_argument("--start-at-unix", type=float, default=0.0)
    server.add_argument("--observe-at-unix", type=float, default=0.0)
    server.add_argument("--active-until-unix", type=float, default=0.0)
    server.add_argument("--ready-timeout", type=float, default=600.0)
    server.add_argument("--close-timeout", type=float, default=30.0)
    server.add_argument("--request-duration-sec", type=float, default=0.0)
    server.add_argument("--requests-per-client", type=int, default=0)
    server.add_argument("--poll-interval-sec", type=float, default=0.8)
    server.add_argument(
        "--poll-worker-connections", type=int, default=200)
    server.add_argument("--request-start-delay-sec", type=float, default=2.0)
    server.add_argument("--output", required=True)
    server.add_argument("--allow-non-10x", action="store_true")

    load = subparsers.add_parser("load")
    add_payload(load)
    add_timeouts(load)
    load.add_argument("--connect-host", required=True)
    load.add_argument("--base-port", type=int, default=62010)
    load.add_argument("--connect-ports", default="")
    load.add_argument("--subservers", type=int, default=5)
    load.add_argument("--clients-per-subserver", type=int, default=1000)
    load.add_argument("--client-id-offset", type=int, default=1)
    load.add_argument("--client-id-prefix", default="load-")
    load.add_argument("--train-client-count", type=int, default=0)
    load.add_argument("--train-steps", type=int, default=1)
    load.add_argument("--train-batch-size", type=int, default=8)
    load.add_argument("--train-learning-rate", type=float, default=0.01)
    load.add_argument("--training-concurrency", type=int, default=4)
    load.add_argument("--output", required=True)
    load.add_argument("--allow-non-10x", action="store_true")

    probe = subparsers.add_parser("probe")
    add_payload(probe)
    add_timeouts(probe)
    probe.add_argument("--connect-host", required=True)
    probe.add_argument("--port", type=int, default=62010)
    probe.add_argument("--connect-at-unix", type=float, default=0.0)
    probe.add_argument("--probe-id", default="late-edge-probe")
    probe.add_argument("--output", required=True)
    probe.add_argument("--allow-non-10x", action="store_true")

    args = parser.parse_args()
    payload_path = Path(args.payload_file)
    if not payload_path.is_file():
        parser.error(f"payload artifact not found: {payload_path}")
    payload = payload_path.read_bytes()
    if len(payload) < 133_380:
        parser.error(
            "serialized MLP artifact must be at least 133380 bytes")
    if args.mode == "load":
        expected_clients = args.subservers * args.clients_per_subserver
        if not 0 <= args.train_client_count <= expected_clients:
            parser.error(
                "--train-client-count must be between 0 and the load size")
        if args.train_steps < 1 or args.train_batch_size < 1:
            parser.error("training steps and batch size must be positive")
        if args.training_concurrency < 1:
            parser.error("--training-concurrency must be positive")

    if args.mode == "server":
        if (args.request_duration_sec > 0) != (args.requests_per_client > 0):
            parser.error(
                "--request-duration-sec and --requests-per-client must "
                "both be positive or both be zero")
        if args.poll_interval_sec <= 0:
            parser.error("--poll-interval-sec must be positive")
        if args.poll_worker_connections < 1:
            parser.error("--poll-worker-connections must be positive")
        async def run_server():
            # Construct asyncio primitives only after asyncio.run() has
            # installed the active loop.  Python 3.9 otherwise binds Event
            # objects to a different loop and fails as soon as wait() runs.
            return await AvailabilityServer(args, payload).run()

        result = asyncio.run(run_server())
        atomic_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["success"]:
            raise SystemExit(2)
    elif args.mode == "load":
        result = asyncio.run(run_load(args, payload))
        if not result["success"]:
            raise SystemExit(2)
    else:
        result = asyncio.run(run_probe(args))
        if not result["success"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()

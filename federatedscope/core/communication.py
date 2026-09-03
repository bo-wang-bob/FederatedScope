import grpc
from concurrent import futures
import logging
import time
import torch.distributed as dist

from collections import deque

from federatedscope.core.proto import gRPC_comm_manager_pb2, \
    gRPC_comm_manager_pb2_grpc
from federatedscope.core.gRPC_server import gRPCComServeFunc
from federatedscope.core.message import Message

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StandaloneCommManager(object):
    """
    The communicator used for standalone mode
    """
    def __init__(self, comm_queue, monitor=None):
        self.comm_queue = comm_queue
        self.neighbors = dict()
        self.monitor = monitor  # used to track the communication related
        # metrics

    def receive(self):
        # we don't need receive() in standalone
        pass

    def add_neighbors(self, neighbor_id, address=None):
        self.neighbors[neighbor_id] = address

    def get_neighbors(self, neighbor_id=None):
        address = dict()
        if neighbor_id:
            if isinstance(neighbor_id, list):
                for each_neighbor in neighbor_id:
                    address[each_neighbor] = self.get_neighbors(each_neighbor)
                return address
            else:
                return self.neighbors[neighbor_id]
        else:
            # Get all neighbors
            return self.neighbors

    def send(self, message):
        # All the workers share one comm_queue
        self.comm_queue.append(message)

    def shutdown(self, grace=0):
        return


class StandaloneDDPCommManager(StandaloneCommManager):
    """
    The communicator used for standalone mode with multigpu
    """
    def __init__(self, comm_queue, monitor=None, id2comm=None):
        super().__init__(comm_queue, monitor)
        self.id2comm = id2comm
        self.device = "cuda:{}".format(dist.get_rank())

    def _send_model_para(self, model_para, dst_rank):
        for v in model_para.values():
            t = v.to(self.device)
            dist.send(tensor=t, dst=dst_rank)

    def send(self, message):
        is_model_para = message.msg_type == 'model_para'
        is_evaluate = message.msg_type == 'evaluate'
        if self.id2comm is None:
            # client to server
            if is_model_para:
                model_para = message.content[1]
                message.content = (message.content[0], {})
                self.comm_queue.append(message) if isinstance(
                    self.comm_queue, deque) else self.comm_queue.put(message)
                self._send_model_para(model_para, 0)
            else:
                self.comm_queue.append(message) if isinstance(
                    self.comm_queue, deque) else self.comm_queue.put(message)
        else:
            receiver = message.receiver
            if not isinstance(receiver, list):
                receiver = [receiver]
            if is_model_para or is_evaluate:
                model_para = message.content
                message.content = {}
            for idx, each_comm in enumerate(self.comm_queue):
                for each_receiver in receiver:
                    if each_receiver in self.neighbors and \
                            self.id2comm[each_receiver] == idx:
                        each_comm.put(message)
                        break
                if is_model_para or is_evaluate:
                    for each_receiver in receiver:
                        if each_receiver in self.neighbors and \
                                self.id2comm[each_receiver] == idx:
                            self._send_model_para(model_para, idx + 1)
                            break
        download_bytes, upload_bytes = message.count_bytes()
        self.monitor.track_upload_bytes(upload_bytes)

    def shutdown(self, grace=0):
        return


class gRPCCommManager(object):
    """
        The implementation of gRPCCommManager is referred to the tutorial on
        https://grpc.io/docs/languages/python/
    """
    def __init__(self, host='0.0.0.0', port='50050', client_num=2, cfg=None):
        self.host = host
        self.port = port
        options = [
            ("grpc.max_send_message_length", cfg.grpc_max_send_message_length),
            ("grpc.max_receive_message_length",
             cfg.grpc_max_receive_message_length),
            ("grpc.enable_http_proxy", cfg.grpc_enable_http_proxy),
            # A CNN covariance upload can exceed 10 minutes when 30 remote
            # clients concurrently send ~300 MB each.  Do not issue a
            # keepalive ping during such an active, back-pressured RPC: gRPC
            # may otherwise report GOAWAY/ping_timeout even though payload
            # bytes are still making progress.  The distributed-stage
            # timeout remains the authoritative liveness bound.
            ("grpc.keepalive_time_ms", 86400000),
            ("grpc.keepalive_timeout_ms", 3600000),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.keepalive_permit_without_calls", 1),
            # gRPC's HTTP/2 bandwidth-delay-product probe uses PING frames
            # independently of keepalive.  With many concurrent, compressed
            # 200-300 MB unary uploads the peer can fail to ACK that probe in
            # time and the C-core closes an otherwise progressing RPC with
            # GOAWAY/ping_timeout.  Disable the probe for these bounded RPCs.
            ("grpc.http2.bdp_probe", 0),
        ]

        if cfg.grpc_compression.lower() == 'deflate':
            self.comp_method = grpc.Compression.Deflate
        elif cfg.grpc_compression.lower() == 'gzip':
            self.comp_method = grpc.Compression.Gzip
        else:
            self.comp_method = grpc.Compression.NoCompression

        self.server_funcs = gRPCComServeFunc()
        self.grpc_server = self.serve(max_workers=client_num,
                                      host=host,
                                      port=port,
                                      options=options)
        self.neighbors = dict()
        self.monitor = None  # used to track the communication related metrics

    def serve(self, max_workers, host, port, options):
        """
        This function is referred to
        https://grpc.io/docs/languages/python/basics/#starting-the-server
        """
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            compression=self.comp_method,
            options=options)
        gRPC_comm_manager_pb2_grpc.add_gRPCComServeFuncServicer_to_server(
            self.server_funcs, server)
        bound_port = server.add_insecure_port("{}:{}".format(host, port))
        if bound_port == 0:
            raise OSError(
                "gRPC could not bind to {}:{}; the address may already be "
                "in use or unavailable".format(host, port))
        server.start()

        return server

    def add_neighbors(self, neighbor_id, address):
        if isinstance(address, dict):
            self.neighbors[neighbor_id] = '{}:{}'.format(
                address['host'], address['port'])
        elif isinstance(address, str):
            self.neighbors[neighbor_id] = address
        else:
            raise TypeError(f"The type of address ({type(address)}) is not "
                            "supported yet")

    def get_neighbors(self, neighbor_id=None):
        address = dict()
        if neighbor_id:
            if isinstance(neighbor_id, list):
                for each_neighbor in neighbor_id:
                    address[each_neighbor] = self.get_neighbors(each_neighbor)
                return address
            else:
                return self.neighbors[neighbor_id]
        else:
            # Get all neighbors
            return self.neighbors

    def _send_request(self, receiver_address, request):
        def _create_stub(receiver_address):
            """
            This part is referred to
            https://grpc.io/docs/languages/python/basics/#creating-a-stub
            """
            channel = grpc.insecure_channel(
                receiver_address,
                compression=self.comp_method,
                options=(('grpc.enable_http_proxy', 0),
                         ('grpc.keepalive_time_ms', 86400000),
                         ('grpc.keepalive_timeout_ms', 3600000),
                         ('grpc.http2.max_pings_without_data', 0),
                         ('grpc.keepalive_permit_without_calls', 1),
                         ('grpc.http2.bdp_probe', 0)))
            stub = gRPC_comm_manager_pb2_grpc.gRPCComServeFuncStub(channel)
            return stub, channel

        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            stub, channel = _create_stub(receiver_address)
            try:
                stub.sendMessage(request)
                return
            except grpc.RpcError as error:
                code = error.code()
                transient = code in {
                    grpc.StatusCode.UNAVAILABLE,
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                }
                if not transient or attempt >= max_attempts:
                    logger.error(
                        "gRPC send to %s failed after %s attempt(s): %s",
                        receiver_address, attempt, error)
                    raise
                backoff = 2 ** attempt
                logger.warning(
                    "Transient gRPC send failure to %s (%s/%s, code=%s); "
                    "retrying in %ss",
                    receiver_address, attempt, max_attempts, code, backoff)
                time.sleep(backoff)
            finally:
                channel.close()

    def _send(self, receiver_address, message):
        request = message.transform(to_list=True)
        self._send_request(receiver_address, request)

    def _send_many(self, receiver_addresses, message):
        """Send one immutable request to several peers concurrently.

        Hierarchical broadcasts can carry tens of millions of covariance
        values.  Building that protobuf once per logical client made a
        30-client fan-out take hours on the Windows subservers.  Convert the
        message once and bound concurrency so the shared request does not
        create one additional in-memory copy per client.
        """
        request = message.transform(to_list=True)
        worker_num = min(8, len(receiver_addresses))
        with futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
            tasks = [
                executor.submit(self._send_request, address, request)
                for address in receiver_addresses
            ]
            # Resolve in submission order so any transport exception remains
            # visible to the caller instead of being lost in a worker thread.
            for task in tasks:
                task.result()

    def send(self, message):
        receiver = message.receiver
        if receiver is not None:
            if not isinstance(receiver, list):
                receiver = [receiver]
            receiver_addresses = []
            sent_addresses = set()
            for each_receiver in receiver:
                if each_receiver in self.neighbors:
                    receiver_address = self.neighbors[each_receiver]
                    # A hierarchical proxy can represent multiple logical
                    # workers at one network endpoint. Keep the full receiver
                    # list for proxy-side fan-out, but transmit the identical
                    # message only once per endpoint.
                    if receiver_address in sent_addresses:
                        continue
                    receiver_addresses.append(receiver_address)
                    sent_addresses.add(receiver_address)
        else:
            receiver_addresses = []
            sent_addresses = set()
            for each_receiver in self.neighbors:
                receiver_address = self.neighbors[each_receiver]
                if receiver_address in sent_addresses:
                    continue
                receiver_addresses.append(receiver_address)
                sent_addresses.add(receiver_address)
        if len(receiver_addresses) == 1:
            self._send(receiver_addresses[0], message)
        elif receiver_addresses:
            self._send_many(receiver_addresses, message)

    def receive(self, timeout=None):
        received_msg = self.server_funcs.receive(timeout=timeout)
        message = Message()
        message.parse(received_msg.msg)
        return message

    def shutdown(self, grace=0):
        grpc_server = getattr(self, 'grpc_server', None)
        if grpc_server is None:
            return
        try:
            stopped = grpc_server.stop(grace)
            stopped.wait(timeout=grace if grace else 1)
        finally:
            self.grpc_server = None

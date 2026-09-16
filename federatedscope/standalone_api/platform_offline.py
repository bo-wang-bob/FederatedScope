"""Offline guard for cache-only workers, not for the HTTP control service."""
import os
import socket
import sys

_installed = False


def _deny_network(event, args):
    if event == 'socket.getaddrinfo':
        raise RuntimeError('离线模式禁止网络解析；请提供完整本地资源')
    if event in {'socket.connect', 'socket.sendto'} and args:
        if args[0].family in {socket.AF_INET, socket.AF_INET6}:
            raise RuntimeError('离线模式禁止训练进程联网；请检查本地缓存和依赖')


def configure_offline_worker():
    """Fail closed on accidental downloads without affecting browser/API traffic."""
    global _installed
    if os.environ.get('FS_PLATFORM_OFFLINE') != '1':
        return
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled',
                      HF_HUB_DISABLE_TELEMETRY='1')
    if not _installed:
        sys.addaudithook(_deny_network)
        _installed = True

import os

from federatedscope.core.workers.base_worker import Worker
from federatedscope.core.workers.base_server import BaseServer
from federatedscope.core.workers.base_client import BaseClient
from federatedscope.core.workers.server import Server
from federatedscope.core.workers.client import Client

if os.environ.get('FEDERATEDSCOPE_GGEUR_LIGHTWEIGHT') != '1':
    # Import FedLSA workers to trigger registration in the full runtime.
    try:
        from federatedscope.core.workers.client_FedLSA import FedLSAClient
        from federatedscope.core.workers.server_FedLSA import FedLSAServer
    except ImportError as e:
        import logging
        logging.getLogger(__name__).warning(
            f'FedLSA workers not available: {e}'
        )
        FedLSAClient = None
        FedLSAServer = None
else:
    FedLSAClient = None
    FedLSAServer = None

__all__ = ['Worker', 'BaseServer', 'BaseClient', 'Server', 'Client',
           'FedLSAClient', 'FedLSAServer']



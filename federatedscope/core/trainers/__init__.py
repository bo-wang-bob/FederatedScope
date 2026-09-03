import os

from federatedscope.core.trainers.base_trainer import BaseTrainer
from federatedscope.core.trainers.trainer import Trainer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.trainers.context import Context

if os.environ.get('FEDERATEDSCOPE_GGEUR_LIGHTWEIGHT') == '1':
    # FedProx is part of the final GGEUR baseline matrix. Its wrapper only
    # uses the torch trainer stack already loaded above, so keep it available
    # without importing the unrelated heavyweight trainers below.
    from federatedscope.core.trainers.trainer_fedprox import \
        wrap_fedprox_trainer

    __all__ = [
        'Trainer', 'Context', 'GeneralTorchTrainer', 'BaseTrainer',
        'wrap_fedprox_trainer'
    ]
else:
    from federatedscope.core.trainers.tf_trainer import GeneralTFTrainer
    from federatedscope.core.trainers.trainer_multi_model import \
        GeneralMultiModelTrainer
    from federatedscope.core.trainers.trainer_pFedMe import wrap_pFedMeTrainer
    from federatedscope.core.trainers.trainer_Ditto import wrap_DittoTrainer
    from federatedscope.core.trainers.trainer_FedEM import FedEMTrainer
    from federatedscope.core.trainers.trainer_FedRep import wrap_FedRepTrainer
    from federatedscope.core.trainers.trainer_simple_tuning import \
        wrap_Simple_tuning_Trainer
    from federatedscope.core.trainers.trainer_fedprox import wrap_fedprox_trainer
    from federatedscope.core.trainers.trainer_nbafl import \
        wrap_nbafl_trainer, wrap_nbafl_server

    try:
        from federatedscope.core.trainers.trainer_FedLSA import FedLSATrainer
    except ImportError as e:
        import logging
        logging.getLogger(__name__).warning(
            f'FedLSA trainer not available: {e}')
        FedLSATrainer = None
    try:
        from federatedscope.core.trainers.trainer_FedProto import \
            FedProtoTrainer
    except ImportError as e:
        import logging
        logging.getLogger(__name__).warning(
            f'FedProto trainer not available: {e}')
        FedProtoTrainer = None

    __all__ = [
        'Trainer', 'Context', 'GeneralTorchTrainer',
        'GeneralMultiModelTrainer', 'wrap_pFedMeTrainer',
        'wrap_DittoTrainer', 'FedEMTrainer', 'wrap_fedprox_trainer',
        'wrap_nbafl_trainer', 'wrap_nbafl_server',
        'wrap_Simple_tuning_Trainer', 'wrap_FedRepTrainer', 'BaseTrainer',
        'GeneralTFTrainer', 'FedLSATrainer', 'FedProtoTrainer'
    ]





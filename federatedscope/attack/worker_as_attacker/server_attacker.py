from federatedscope.core.workers import Server
from federatedscope.core.message import Message
from federatedscope.attack.auxiliary.a3fl_utils import \
    get_a3fl_active_attacker_ids

from federatedscope.core.auxiliaries.criterion_builder import get_criterion
import copy
from federatedscope.attack.auxiliary.utils import get_data_sav_fn, \
    get_reconstructor

import logging
import json
import math
from pathlib import Path

import torch
import numpy as np
from federatedscope.attack.privacy_attacks.passive_PIA import \
    PassivePropertyInference

logger = logging.getLogger(__name__)


class BackdoorServer(Server):
    '''
    For backdoor attacks, we will choose different sampling stratergies.
    fix-frequency, all-round ,or random sampling.
    '''
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 unseen_clients_id=None,
                 **kwargs):
        super(BackdoorServer, self).__init__(ID=ID,
                                             state=state,
                                             data=data,
                                             model=model,
                                             config=config,
                                             client_num=client_num,
                                             total_round_num=total_round_num,
                                             device=device,
                                             strategy=strategy,
                                             **kwargs)

    def _sample_a3fl_receiver(self, sample_client_num):
        active_attackers = get_a3fl_active_attacker_ids(
            self._cfg, self.state, sample_client_num)
        if len(active_attackers) == 0:
            return self.sampler.sample(size=sample_client_num)

        candidate_clients = np.arange(1, self.client_num + 1)
        benign_pool = np.array(
            [x for x in candidate_clients if x not in active_attackers])
        benign_num = max(0, sample_client_num - len(active_attackers))
        benign_receivers = np.random.choice(benign_pool,
                                            size=benign_num,
                                            replace=False).tolist()
        return active_attackers + benign_receivers

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        """
        To broadcast the message to all clients or sampled clients

        Arguments:
            msg_type: 'model_para' or other user defined msg_type
            sample_client_num: the number of sampled clients in the broadcast
                behavior. And sample_client_num = -1 denotes to broadcast to
                all the clients.
            filter_unseen_clients: whether filter out the unseen clients that
                do not contribute to FL process by training on their local
                data and uploading their local model update. The splitting is
                useful to check participation generalization gap in [ICLR'22,
                What Do We Mean by Generalization in Federated Learning?]
                You may want to set it to be False when in evaluation stage
        """

        if filter_unseen_clients:
            # to filter out the unseen clients when sampling
            self.sampler.change_state(self.unseen_clients_id, 'unseen')

        if sample_client_num > 0:  # only activated at training process
            if self._cfg.attack.attack_method.lower() == 'a3fl':
                receiver = self._sample_a3fl_receiver(sample_client_num)
            else:
                attacker_id = self._cfg.attack.attacker_id
                setting = self._cfg.attack.setting
                insert_round = self._cfg.attack.insert_round

                if attacker_id == -1 or self._cfg.attack.attack_method == '':

                    receiver = np.random.choice(np.arange(1, self.client_num + 1),
                                                size=sample_client_num,
                                                replace=False).tolist()

                elif setting == 'fix':
                    if self.state % self._cfg.attack.freq == 0:
                        client_list = np.delete(np.arange(1, self.client_num + 1),
                                                self._cfg.attack.attacker_id - 1)
                        receiver = np.random.choice(client_list,
                                                    size=sample_client_num - 1,
                                                    replace=False).tolist()
                        receiver.insert(0, self._cfg.attack.attacker_id)
                        logger.info('starting the fix-frequency poisoning attack')
                        logger.info(
                            'starting poisoning round: {:d}, the attacker ID: {:d}'
                            .format(self.state, self._cfg.attack.attacker_id))
                    else:
                        client_list = np.delete(np.arange(1, self.client_num + 1),
                                                self._cfg.attack.attacker_id - 1)
                        receiver = np.random.choice(client_list,
                                                    size=sample_client_num,
                                                    replace=False).tolist()

                elif setting == 'single' and self.state == insert_round:
                    client_list = np.delete(np.arange(1, self.client_num + 1),
                                            self._cfg.attack.attacker_id - 1)
                    receiver = np.random.choice(client_list,
                                                size=sample_client_num - 1,
                                                replace=False).tolist()
                    receiver.insert(0, self._cfg.attack.attacker_id)
                    logger.info('starting the single-shot poisoning attack')
                    logger.info(
                        'starting poisoning round: {:d}, the attacker ID: {:d}'.
                        format(self.state, self._cfg.attack.attacker_id))

                elif self._cfg.attack.setting == 'all':

                    client_list = np.delete(np.arange(1, self.client_num + 1),
                                            self._cfg.attack.attacker_id - 1)
                    receiver = np.random.choice(client_list,
                                                size=sample_client_num - 1,
                                                replace=False).tolist()
                    receiver.insert(0, self._cfg.attack.attacker_id)
                    logger.info('starting the all-round poisoning attack')
                    logger.info(
                        'starting poisoning round: {:d}, the attacker ID: {:d}'.
                        format(self.state, self._cfg.attack.attacker_id))

                else:
                    receiver = np.random.choice(np.arange(1, self.client_num + 1),
                                                size=sample_client_num,
                                                replace=False).tolist()

        else:
            # broadcast to all clients
            receiver = list(self.comm_manager.neighbors.keys())

        if self._noise_injector is not None and msg_type == 'model_para':
            # Inject noise only when broadcast parameters
            for model_idx_i in range(len(self.models)):
                num_sample_clients = [
                    v["num_sample"] for v in self.join_in_info.values()
                ]
                self._noise_injector(self._cfg, num_sample_clients,
                                     self.models[model_idx_i])

        skip_broadcast = self._cfg.federate.method in ["local", "global"]
        if self.model_num > 1:
            model_para = [{} if skip_broadcast else model.state_dict()
                          for model in self.models]
        else:
            model_para = {} if skip_broadcast else self.model.state_dict()

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=min(self.state, self.total_round_num),
                    content=model_para))
        if self._cfg.federate.online_aggr:
            for idx in range(self.model_num):
                self.aggregators[idx].reset()

        if filter_unseen_clients:
            # restore the state of the unseen clients within sampler
            self.sampler.change_state(self.unseen_clients_id, 'seen')


class PassiveServer(Server):
    '''
    In passive attack, the server store the model and the message collected
    from the client,and perform the optimization based reconstruction,
    such as DLG, InvertGradient.
    '''
    def __init__(self,
                 ID=-1,
                 state=0,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 state_to_reconstruct=None,
                 client_to_reconstruct=None,
                 **kwargs):
        super(PassiveServer, self).__init__(ID=ID,
                                            state=state,
                                            data=data,
                                            model=model,
                                            client_num=client_num,
                                            total_round_num=total_round_num,
                                            device=device,
                                            strategy=strategy,
                                            **kwargs)

        # self.offline_reconstruct = offline_reconstruct
        self.atk_method = self._cfg.attack.attack_method
        self.state_to_reconstruct = state_to_reconstruct
        self.client_to_reconstruct = client_to_reconstruct
        self.reconstruct_data = dict()
        self.data = data
        self.full_msg_buffer = dict()

        # the loss function of the global model; the global model can be
        # obtained in self.aggregator.model
        self.model_criterion = get_criterion(self._cfg.criterion.type,
                                             device=self.device)

        if self.atk_method.lower() in ['fedmia', 'ggeur_fedmia']:
            self.data_dim, self.num_class, self.is_one_hot_label = None, None, None
            self.reconstructor = None
            self.reconstructed_data_sav_fn = None
        else:
            from federatedscope.attack.auxiliary.utils import get_data_info
            self.data_dim, self.num_class, self.is_one_hot_label = get_data_info(
                self._cfg.data.type)
            self.reconstructor = self._get_reconstructor()
            self.reconstructed_data_sav_fn = get_data_sav_fn(self._cfg.data.type)

        self.reconstruct_data_summary = dict()
        self.grnn_psnr_threshold = float(getattr(
            self._cfg.attack, 'grnn_psnr_threshold', 0.9))
        self.grnn_psnr_values = []

    def _record_grnn_success(self, reconstruction, original):
        """Record PSNR-based GRNN success as soon as a reference is available."""
        if self.atk_method.lower() != 'grnn' or original is None:
            return
        try:
            recon = reconstruction.detach().float().cpu()
            ref = original.detach().float().cpu()
            if recon.ndim == 3:
                recon = recon.unsqueeze(0)
            if ref.ndim == 3:
                ref = ref.unsqueeze(0)
            count = min(recon.shape[0], ref.shape[0])
            if count == 0:
                return
            ref = ref[:count].clamp(0.0, 1.0)
            recon = recon[:count].clamp(0.0, 1.0)
            if ref.shape[-2:] != recon.shape[-2:]:
                ref = torch.nn.functional.interpolate(
                    ref, size=recon.shape[-2:], mode='bilinear',
                    align_corners=False)
            mse = (ref - recon).pow(2).flatten(1).mean(1)
            psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1.0e-12))
            self.grnn_psnr_values.extend(float(v) for v in psnr.tolist()
                                         if math.isfinite(float(v)))
            total = len(self.grnn_psnr_values)
            successful = sum(v > self.grnn_psnr_threshold
                             for v in self.grnn_psnr_values)
            summary = {
                'metric': 'psnr_db',
                'success_rule': f'PSNR > {self.grnn_psnr_threshold}',
                'threshold': self.grnn_psnr_threshold,
                'total_reconstructions': total,
                'successful_reconstructions': successful,
                'failed_reconstructions': total - successful,
                'attack_success_rate': successful / total,
                'attack_success_rate_percent': successful / total * 100.0,
                'mean_psnr_db': sum(self.grnn_psnr_values) / total,
                'psnr_values_db': self.grnn_psnr_values,
            }
            path = Path(self._cfg.outdir) / 'grnn_attack_success_summary.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(summary, indent=2) + '\\n',
                            encoding='utf-8')
            logger.info('[GRNN] PSNR success: %s/%s (%.2f%%), threshold=%.3f',
                        successful, total, summary['attack_success_rate_percent'],
                        self.grnn_psnr_threshold)
        except Exception as exc:
            logger.warning('[GRNN] Failed to record PSNR success: %s', exc)

    def _get_reconstructor(self):

        if self.atk_method.lower() in ['fedmia', 'ggeur_fedmia']:
            return None

        reconstructor_kwargs = {
            'max_ite': self._cfg.attack.max_ite,
            'lr': self._cfg.attack.reconstruct_lr,
            'federate_loss_fn': self.model_criterion,
            'device': self.device,
            'federate_lr': self._cfg.train.optimizer.lr,
            'optim': self._cfg.attack.reconstruct_optim,
            'info_diff_type': self._cfg.attack.info_diff_type,
            'federate_method': self._cfg.federate.method,
            'alpha_TV': self._cfg.attack.alpha_TV
        }
        if self.atk_method.lower() == 'grnn':
            reconstructor_kwargs.update({
                'g_in': self._cfg.attack.get('grnn_g_in', 128),
                'tv_weight': self._cfg.attack.get('grnn_tv_weight', 1e-6),
                'use_wd': self._cfg.attack.get('grnn_use_wd', False),
                'dataset_name': self._cfg.data.type
            })

        return get_reconstructor(self.atk_method, **reconstructor_kwargs)

    def _reconstruct(self, model_para, batch_size, state, sender,
                     gradients=None):
        logger.info('-------- reconstruct round:{}, client:{}---------'.format(
            state, sender))
        if gradients is not None:
            logger.info('Using real gradients from client for reconstruction')
        else:
            logger.info('No real gradients available; using parameter diff')

        reconstruct_kwargs = {
            'model': copy.deepcopy(self.model).to(torch.device(self.device)),
            'original_info': model_para,
            'data_feature_dim': self.data_dim,
            'num_class': self.num_class,
            'batch_size': batch_size
        }
        if self.atk_method.lower() == 'grnn':
            reconstruct_kwargs['real_gradients'] = gradients

        dummy_data, dummy_label = self.reconstructor.reconstruct(
            **reconstruct_kwargs)
        if state not in self.reconstruct_data.keys():
            self.reconstruct_data[state] = dict()
        self.reconstruct_data[state][sender] = [
            dummy_data.cpu(), dummy_label.cpu()
        ]

    def run_reconstruct(self, state_list=None, sender_list=None):

        if state_list is None:
            state_list = self.msg_buffer['train'].keys()

        # After FL running, using gradient based reconstruction method to
        # recover client's private training data
        for state in state_list:
            if sender_list is None:
                sender_list = self.msg_buffer['train'][state].keys()
            for sender in sender_list:
                if state in self.full_msg_buffer and \
                        sender in self.full_msg_buffer[state]:
                    full_content = self.full_msg_buffer[state][sender]
                    sample_size = full_content[0]
                    model_para = full_content[1]
                    gradients = full_content[2] if len(full_content) > 2 else None
                else:
                    content = self.msg_buffer['train'][state][sender]
                    sample_size = content[0]
                    model_para = content[1]
                    gradients = None

                if self.atk_method.lower() == 'grnn':
                    batch_size = min(sample_size, self._cfg.dataloader.batch_size)
                else:
                    batch_size = sample_size

                self._reconstruct(model_para=model_para,
                                  batch_size=batch_size,
                                  gradients=gradients,
                                  state=state,
                                  sender=sender)

    def callback_funcs_model_para(self, message: Message):
        if self.is_finish:
            return 'finish'

        round, sender, content = message.state, message.sender, message.content
        self.sampler.change_state(sender, 'idle')
        if round not in self.msg_buffer['train']:
            self.msg_buffer['train'][round] = dict()
        if round not in self.full_msg_buffer:
            self.full_msg_buffer[round] = dict()
        self.full_msg_buffer[round][sender] = content

        if isinstance(content, (tuple, list)) and len(content) >= 2:
            self.msg_buffer['train'][round][sender] = content[:2]
        else:
            self.msg_buffer['train'][round][sender] = content

        # run reconstruction before the clear of self.msg_buffer
        if 'DLG_loss' not in self.best_results.keys():
            self.best_results['DLG_loss'] = {}
        if round not in self.best_results['DLG_loss'].keys():
            self.best_results['DLG_loss'][round] = {}

        if self.state_to_reconstruct is None or message.state in \
                self.state_to_reconstruct:
            if self.client_to_reconstruct is None or message.sender in \
                    self.client_to_reconstruct:
                self.run_reconstruct(state_list=[message.state],
                                     sender_list=[message.sender])
                logger.info(
                    'Finish {} attack; Final reconstruction loss: {}'.
                    format(self.atk_method.upper(),
                           self.reconstructor.dlg_recover_loss))
                self.best_results['DLG_loss'][round][
                    sender] = self.reconstructor.dlg_recover_loss
                if self.reconstructed_data_sav_fn is not None:
                    dummy_data, dummy_label = self.reconstruct_data[message.state][message.sender]

                    # 将标签转换成字符串（支持单个或多个标签）
                    if dummy_label.numel() == 1:
                        label_str = str(dummy_label.item())
                    else:
                        label_str = '_'.join(map(str, dummy_label.cpu().numpy()))

                    filename = f'image_state_{message.state}_client_{message.sender}_label_{label_str}.png'

                    # 尝试从客户端发送的数据中获取原始训练批次
                    original_data = None
                    try:
                        # content格式: (sample_size, shared_model_para, gradients, last_batch_data)
                        # last_batch_data是tuple: (data, labels) 或 None
                        if len(content) >= 4 and content[3] is not None:
                            last_batch_data = content[3]
                            if isinstance(last_batch_data, tuple) and len(last_batch_data) == 2:
                                original_data, original_labels = last_batch_data
                                logger.info(f"[Original Data] Retrieved from client: shape={original_data.shape}, labels={original_labels}")
                            else:
                                logger.warning(f"[Original Data] Invalid format from client: {type(last_batch_data)}")
                        else:
                            logger.info(f"[Original Data] Client did not send original batch data, falling back to dataset search")

                            # 后备方案：从数据集中查找同标签样本
                            target_label = dummy_label.item() if dummy_label.numel() == 1 else dummy_label[0].item()

                            if self.data is not None and sender in self.data:
                                client_data = self.data[sender]

                                if 'train' in client_data and client_data['train'] is not None:
                                    train_loader = client_data['train']

                                    # Try to find a sample with matching label
                                    found_match = False
                                    for batch in train_loader:
                                        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                                            batch_data, batch_labels = batch[0], batch[1]

                                            # Look for matching label in this batch
                                            for i in range(len(batch_labels)):
                                                if batch_labels[i].item() == target_label:
                                                    original_data = batch_data[i:i+1]
                                                    found_match = True
                                                    logger.info(f"[Original Data] Found matching label {target_label} in dataset")
                                                    break

                                            if found_match:
                                                break

                                    # Fallback to first sample if no match found
                                    if not found_match and original_data is None:
                                        # 尝试获取第一个batch的第一个样本
                                        for batch in train_loader:
                                            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                                                batch_data, batch_labels = batch[0], batch[1]
                                                if len(batch_data) > 0:
                                                    original_data = batch_data[:1]
                                                    logger.warning(f"[Original Data] No matching label {target_label}, using first sample")
                                                    break

                                    if original_data is not None:
                                        logger.info(f"[Original Data] From dataset: shape={original_data.shape}")
                    except Exception as e:
                        logger.warning(f"[Original Data] Failed to retrieve: {e}")
                        import traceback
                        logger.warning(traceback.format_exc())
                        original_data = None

                    logger.info(f"[Save Image] original_data: {'provided' if original_data is not None else 'None'}")

                    self._record_grnn_success(dummy_data, original_data)

                    self.reconstructed_data_sav_fn(
                        data=dummy_data,
                        sav_pth=self._cfg.outdir,
                        name=filename,
                        original_data=original_data)

                    # 保存后立即清理，释放内存
                    del dummy_data, dummy_label
                    if original_data is not None:
                        del original_data
                    # 清理重构数据缓存
                    if message.state in self.reconstruct_data:
                        if message.sender in self.reconstruct_data[message.state]:
                            del self.reconstruct_data[message.state][message.sender]
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        self.check_and_move_on()


class PassivePIAServer(Server):
    '''
    The implementation of the batch property classifier, the algorithm 3 in
    paper: Exploiting Unintended Feature Leakage in Collaborative Learning

    References:

    Melis, Luca, Congzheng Song, Emiliano De Cristofaro and Vitaly
    Shmatikov. “Exploiting Unintended Feature Leakage in Collaborative
    Learning.” 2019 IEEE Symposium on Security and Privacy (SP) (2019): 691-706
    '''
    def __init__(self,
                 ID=-1,
                 state=0,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):
        super(PassivePIAServer, self).__init__(ID=ID,
                                               state=state,
                                               data=data,
                                               model=model,
                                               client_num=client_num,
                                               total_round_num=total_round_num,
                                               device=device,
                                               strategy=strategy,
                                               **kwargs)

        # self.offline_reconstruct = offline_reconstruct
        self.atk_method = self._cfg.attack.attack_method
        self.pia_attacker = PassivePropertyInference(
            classier=self._cfg.attack.classifier_PIA,
            fl_model_criterion=get_criterion(self._cfg.criterion.type,
                                             device=self.device),
            device=self.device,
            grad_clip=self._cfg.grad.grad_clip,
            dataset_name=self._cfg.data.type,
            fl_local_update_num=self._cfg.train.local_update_steps,
            fl_type_optimizer=self._cfg.train.optimizer.type,
            fl_lr=self._cfg.train.optimizer.lr,
            batch_size=100)

        # self.optimizer = get_optimizer(
        # type=self._cfg.fedopt.type_optimizer, model=self.model,
        # lr=self._cfg.fedopt.optimizer.lr)
        # print(self.optimizer)
    def callback_funcs_model_para(self, message: Message):
        if self.is_finish:
            return 'finish'

        round, sender, content = message.state, message.sender, message.content
        self.sampler.change_state(sender, 'idle')
        if round not in self.msg_buffer['train']:
            self.msg_buffer['train'][round] = dict()
        self.msg_buffer['train'][round][sender] = content

        # collect the updates
        self.pia_attacker.collect_updates(
            previous_para=self.model.state_dict(),
            updated_parameter=content[1],
            round=round,
            client_id=sender)
        self.pia_attacker.get_data_for_dataset_prop_classifier(
            model=self.model)

        if self._cfg.federate.online_aggr:
            # TODO: put this line to `check_and_move_on`
            # currently, no way to know the latest `sender`
            self.aggregator.inc(content)
        self.check_and_move_on()

        if self.state == self.total_round_num:
            self.pia_attacker.train_property_classifier()
            self.pia_results = self.pia_attacker.infer_collected()
            print(self.pia_results)
